#!/usr/bin/env python3
"""
GR00T Inference Service Management Tool

Manages GR00T policy inference services running in Docker containers.
Uses Isaac-GR00T's native inference service for proper ZMQ/HTTP communication.

Container lifecycle (``build_image`` / ``download_checkpoint`` /
``start_container`` / ``lifecycle="full"``) wraps the four manual setup
steps so an LLM driving the AgentTool can fully orchestrate a GR00T eval
from a single prompt - see #148 for the motivation.
"""

import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from strands import tool

from strands_robots.utils import get_base_dir, require_optional

# Default cache layout for the lifecycle helpers. Mirrors the layout
# documented in the Isaac-GR00T README so users moving from the manual
# four-step setup don't have to relocate their existing artefacts.
_DEFAULT_REPO_URL = "https://github.com/NVIDIA/Isaac-GR00T"
_DEFAULT_REPO_TAG = "n1.7-release"
_DEFAULT_IMAGE_NAME = "gr00t:latest"
_DEFAULT_CONTAINER_COMMAND = "tail -f /dev/null"


# --- container hardening: image allowlist + dangerous-mount guard -------
#
# ``_start_container`` builds a ``docker run`` argv. The agent-facing
# ``gr00t_inference`` tool deliberately does NOT expose ``volumes``,
# ``image_name``, or ``container_command`` as parameters -- a prompt-
# injected agent must never be able to mount the host filesystem, pick an
# arbitrary image, or inject a container command. Container topology is
# operator-config-driven (env vars below), not agent-driven.
#
# The image the container runs is resolved from the operator environment
# (``STRANDS_GR00T_IMAGE``) and validated against an allowlist. The default
# allowlist covers the canonical GR00T images; operators extend it via
# ``STRANDS_GR00T_IMAGE_ALLOW`` (comma-separated; supports a trailing ``*``
# tag wildcard, e.g. ``myregistry/gr00t:*``).

# Built-in image-name allowlist patterns. A trailing ``*`` is a tag/suffix
# wildcard (matches any characters); everything else is matched literally.
_DEFAULT_IMAGE_ALLOW: tuple[str, ...] = (
    "gr00t:*",
    "nvcr.io/nvidia/isaac-gr00t:*",
)

# Built-in repo-URL allowlist for ``build_image`` source clones. Unlike the
# image allowlist (a tag wildcard), repo URLs are matched EXACTLY against the
# canonical set (with/without the ``.git`` suffix): a trailing-``*`` wildcard
# on a URL would let ``https://github.com/NVIDIA/Isaac-GR00T-evil`` slip past a
# ``...Isaac-GR00T*`` pattern. Operators add private mirrors via
# ``STRANDS_GR00T_REPO_URL_ALLOW`` (comma-separated, each entry exact-matched).
_DEFAULT_REPO_URL_ALLOW: tuple[str, ...] = (
    _DEFAULT_REPO_URL,
    _DEFAULT_REPO_URL + ".git",
)

# Host paths that must never be bind-mounted into a container. Mounting any
# of these hands the container (and anything that can influence its command)
# control over the host: root fs, the docker socket (daemon takeover),
# credential/identity dirs, and kernel/proc/sys pseudo-filesystems.
# NOTE (#384, item 1): ``/home`` is blocked wholesale, not narrowed to the
# sensitive subpaths (~/.ssh, ~/.aws, ~/.config). Rationale: this guard is
# defence-in-depth for an untrusted/prompt-injected caller, and any home
# directory may hold credentials, tokens, or dotfiles whose names we cannot
# enumerate ahead of time. Operators who need a checkpoint bind-mount must
# place it OUTSIDE ``/home`` (e.g. ``/data/checkpoints`` or ``/opt/...``); the
# auto-derived default (``~/.cache/huggingface``) is never agent-controlled
# and reaches docker only via the curated ``effective_volumes`` set. See the
# README Configuration section for the operator-facing guidance.
_BLOCKED_VOLUME_HOST_PATHS: tuple[str, ...] = (
    "/",
    "/etc",
    "/root",
    "/home",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/var",
    "/var/run",
    "/run",
)
# Exact-match files/sockets that must never be mounted regardless of dir rules.
_BLOCKED_VOLUME_EXACT: tuple[str, ...] = (
    "/var/run/docker.sock",
    "/run/docker.sock",
)


def _image_allowlist() -> tuple[str, ...]:
    """Return the configured image allowlist (defaults + env extras)."""
    raw = os.getenv("STRANDS_GR00T_IMAGE_ALLOW", "")
    extras = tuple(t.strip() for t in raw.split(",") if t.strip())
    return _DEFAULT_IMAGE_ALLOW + extras


# Image-name charset gate: docker image refs use [A-Za-z0-9._:/@-] only.
# Anything outside that (whitespace, ``;$()`` `\`` etc.) is rejected before
# the value reaches argv, defence in depth even though the container is
# started with subprocess in argv-mode (no shell).
_IMAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$")


def _is_allowed_image(image_name: str) -> bool:
    """True iff *image_name* matches an allowlist pattern.

    A pattern ending in ``*`` matches any image whose name starts with the
    pattern prefix (tag wildcard). Other patterns match literally. The image
    name itself must also pass a charset gate so shell metacharacters in the
    tag (``gr00t:;rm``, ``gr00t:$(x)``) cannot ride through even if they
    matched a wildcard prefix.
    """
    if not isinstance(image_name, str) or not image_name:
        return False
    if not _IMAGE_NAME_RE.match(image_name):
        return False
    for pattern in _image_allowlist():
        if pattern.endswith("*"):
            if image_name.startswith(pattern[:-1]):
                return True
        elif image_name == pattern:
            return True
    return False


def _resolve_image_name() -> str:
    """Resolve the container image from operator config.

    The agent has no say in the image. Operators set ``STRANDS_GR00T_IMAGE``
    (defaulting to the canonical ``gr00t:latest``); the value must pass the
    allowlist or resolution fails closed.
    """
    return os.getenv("STRANDS_GR00T_IMAGE", _DEFAULT_IMAGE_NAME)


def _repo_url_allowlist() -> tuple[str, ...]:
    """Return the configured repo-URL allowlist (defaults + env extras)."""
    raw = os.getenv("STRANDS_GR00T_REPO_URL_ALLOW", "")
    extras = tuple(u.strip() for u in raw.split(",") if u.strip())
    return _DEFAULT_REPO_URL_ALLOW + extras


def _is_allowed_repo_url(repo_url: str) -> bool:
    """True iff *repo_url* is an exact match for an allowlisted URL.

    Exact match only (no wildcard): a substring/prefix test would let an
    attacker-controlled host (``...Isaac-GR00T-evil``, ``...Isaac-GR00T.evil``)
    slip past. A leading ``-`` is rejected outright so the value can never be
    consumed as a ``git`` option (argument injection).
    """
    if not isinstance(repo_url, str) or not repo_url or repo_url.startswith("-"):
        return False
    return repo_url in _repo_url_allowlist()


# git refs may legitimately contain letters, digits, and ``._/-``; anything
# else (whitespace, shell metacharacters, a leading ``-`` that git would read
# as an option) is rejected before the value reaches ``git --branch``/checkout.
_REPO_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def _is_allowed_repo_tag(repo_tag: str) -> bool:
    """True iff *repo_tag* is a safe git ref (no option/metachar injection)."""
    return isinstance(repo_tag, str) and bool(_REPO_TAG_RE.match(repo_tag))


def _resolve_repo_url() -> str:
    """Resolve the Isaac-GR00T clone URL from operator config.

    The agent has no say in the source repo. Operators set
    ``STRANDS_GR00T_REPO_URL`` (defaulting to the canonical NVIDIA repo); the
    value must pass the allowlist or ``build_image`` fails closed.
    """
    return os.getenv("STRANDS_GR00T_REPO_URL", _DEFAULT_REPO_URL)


def _resolve_repo_tag() -> str:
    """Resolve the Isaac-GR00T clone tag/branch from operator config."""
    return os.getenv("STRANDS_GR00T_REPO_TAG", _DEFAULT_REPO_TAG)


def _resolve_build_source() -> tuple[str, str] | dict[str, Any]:
    """Resolve+validate the operator-configured clone URL and tag.

    Returns ``(repo_url, repo_tag)`` on success, or a structured error dict if
    the configured URL is off-allowlist or the tag is not a safe git ref. The
    agent cannot influence either value (both removed from the tool signature);
    this guard catches a misconfigured operator env and fails closed.
    """
    repo_url = _resolve_repo_url()
    repo_tag = _resolve_repo_tag()
    if not _is_allowed_repo_url(repo_url):
        return {
            "status": "error",
            "message": (
                f"configured repo URL {repo_url!r} is not in the allowlist "
                f"{list(_repo_url_allowlist())}. Set STRANDS_GR00T_REPO_URL to "
                "an allowed URL or extend STRANDS_GR00T_REPO_URL_ALLOW."
            ),
        }
    if not _is_allowed_repo_tag(repo_tag):
        return {
            "status": "error",
            "message": (
                f"configured repo tag {repo_tag!r} is not a valid git ref "
                "(allowed: letters, digits, and '._/-', no leading '-')."
            ),
        }
    return repo_url, repo_tag


def _normalize_host_path(host_path: str) -> str:
    """Normalize a bind-mount host path for prefix comparison.

    POSIX trap: ``os.path.normpath('//etc')`` returns ``'//etc'`` (preserves
    a leading double slash, per POSIX § 4.13). The Linux kernel collapses
    the leading ``//`` to ``/`` at path-lookup, so ``docker -v //etc:/x``
    really mounts ``/etc``. We collapse runs of leading slashes ourselves
    BEFORE normpath so the blocklist comparison matches reality.
    """
    expanded = os.path.expanduser(host_path)
    # Collapse any run of leading slashes to a single '/'
    if expanded.startswith("/"):
        expanded = "/" + expanded.lstrip("/")
    return os.path.normpath(expanded)


def _resolve_host_path(host_path: str) -> str:
    """Resolve a bind-mount host path through symlinks for blocklist comparison.

    NOTE (#384, item 2): ``_normalize_host_path`` only canonicalises slashes
    and runs ``normpath`` -- it does NOT follow symlinks. A pre-existing host
    symlink pointing at a protected dir (e.g. ``/data/ckpt -> /etc``) would
    pass the blocklist while docker mounts the resolved target. We additionally
    compare the ``realpath`` so the symlink target is also checked.

    Residual TOCTOU: ``realpath`` resolves at check time; a symlink swapped
    between check and ``docker run`` could still differ. That race is not
    closable at this layer (it is a host-fs-mutation primitive the gr00t tool
    does not expose to the agent), so we resolve best-effort and accept the
    residual gap. ``os.path.realpath`` does not raise on missing paths -- it
    resolves as far as it can -- so this is safe to call on not-yet-created
    mount sources.
    """
    return os.path.realpath(os.path.expanduser(host_path))


def _check_volume_safety(volumes: dict[str, str] | None) -> str | None:
    """Return None if all bind-mount host paths are safe, else a reason.

    Defence in depth for the internal/operator/test entry point into
    ``_start_container`` (the agent tool no longer supplies volumes at all).
    Rejects mounts of the host root, system dirs, credential dirs, and the
    docker socket.
    """
    if not volumes:
        return None
    blocked_dirs = {os.path.normpath(p) for p in _BLOCKED_VOLUME_HOST_PATHS}
    blocked_exact = {os.path.normpath(p) for p in _BLOCKED_VOLUME_EXACT}
    for host_path in volumes:
        norm = _normalize_host_path(str(host_path))
        # #384 item 2: also evaluate the symlink-resolved path so a host symlink
        # pointing into a protected dir cannot slip past the prefix check.
        resolved = _resolve_host_path(str(host_path))
        candidates = {norm, resolved}
        if blocked_exact & candidates:
            return f"refusing to mount {host_path!r}: docker socket / sensitive path"
        # Prefix check: reject the protected dir itself AND any child of it, so
        # mounting /etc/shadow, /root/.ssh/id_rsa, /home/<u>/.aws/credentials,
        # /proc/1/environ, /var/run/docker.sock.bak, etc. is blocked too. Root
        # ("/") is matched exactly only -- a prefix test on "/" would reject
        # every absolute path, including legitimate operator mounts.
        for blocked in blocked_dirs:
            if blocked == os.sep:
                if os.sep in candidates:
                    return f"refusing to mount {host_path!r}: host root filesystem"
                continue
            for cand in candidates:
                if cand == blocked or cand.startswith(blocked + os.sep):
                    return f"refusing to mount {host_path!r}: under protected host path {blocked!r}"
    return None


def _check_hf_local_dir_safety(hf_local_dir: str | None) -> str | None:
    """Return None if an agent-supplied ``hf_local_dir`` is safe, else a reason.

    ``hf_local_dir`` is an untrusted agent string that reaches two host-fs
    sinks: ``_download_checkpoint`` writes the snapshot to it directly on the
    host (no docker mediation), and ``_start_container`` bind-mounts it into
    the container. Both must reject a prompt-injected path like ``/etc`` or
    ``/root/.ssh``. Validating once at the agent dispatch boundary closes every
    sink (including ``lifecycle``, which downloads before it starts the
    container) rather than guarding each call site. Reuses the same
    expand + prefix-match blocklist as bind-mount validation.
    """
    if not hf_local_dir:
        return None
    return _check_volume_safety({str(Path(hf_local_dir).expanduser()): "/data/checkpoints"})


def _isaac_gr00t_dir() -> Path:
    """Default clone destination for the Isaac-GR00T source tree."""
    return get_base_dir() / "Isaac-GR00T"


def _checkpoints_dir() -> Path:
    """Default download destination for HuggingFace checkpoints."""
    return get_base_dir() / "checkpoints"


@tool
def gr00t_inference(
    action: str,
    checkpoint_path: str | None = None,
    policy_name: str | None = None,
    port: int = 5555,
    data_config: str = "fourier_gr1_arms_only",
    embodiment_tag: str = "gr1",
    denoising_steps: int = 4,
    host: str = "0.0.0.0",
    container_name: str | None = None,
    timeout: int = 60,
    use_tensorrt: bool = False,
    trt_engine_path: str = "gr00t_engine",
    vit_dtype: str = "fp8",
    llm_dtype: str = "nvfp4",
    dit_dtype: str = "fp8",
    http_server: bool = False,
    api_token: str | None = None,
    protocol: str = "n1.5",
    use_sim_policy_wrapper: bool = False,
    hf_repo: str | None = None,
    hf_subfolder: str | None = None,
    hf_local_dir: str | None = None,
    hf_token: str | None = None,
    lifecycle: str = "full",
    remove_volumes: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Manage GR00T N1 inference services in Docker containers.

    Starts, stops, and monitors Isaac-GR00T inference services running inside
    Docker containers. Supports both ZMQ (low-latency) and HTTP (REST API)
    protocols, with optional TensorRT acceleration.

    Prerequisites:
        - Docker installed and running
        - An Isaac-GR00T container pulled and started (e.g., ``nvcr.io/nvidia/isaac-gr00t``)
        - A GR00T N1 checkpoint (fine-tuned or pre-trained)
        - NVIDIA GPU with sufficient VRAM (8GB+ recommended)

    Actions:
        - ``start``: Launch an inference service with a checkpoint. Requires ``checkpoint_path``.
        - ``stop``: Terminate a running service on the specified ``port``.
        - ``status``: Check whether a service is running on the specified ``port``.
        - ``list``: Discover all running services across common ports (5555-5558, 8000-8003).
        - ``restart``: Stop and re-start a service (e.g., to swap checkpoints). Requires ``checkpoint_path``.
        - ``find_containers``: List available Isaac-GR00T Docker containers.
        - ``build_image``: Clone Isaac-GR00T at ``repo_tag`` and run ``bash docker/build.sh``.
          Idempotent - skips the build when ``image_name`` already exists in the local
          docker daemon. Pass ``force=True`` to rebuild.
        - ``download_checkpoint``: Download a HuggingFace checkpoint to a local cache
          directory using ``huggingface_hub``. Requires ``hf_repo``;
          ``hf_subfolder`` filters to a single sub-checkpoint (e.g.
          ``"libero_spatial"``). Idempotent - skips when the local directory is
          already populated unless ``force=True``.
        - ``start_container``: ``docker run -d --gpus all --ipc=host`` on the
          operator-configured image with ``container_name``, the default
          checkpoint + HF-cache volume mounts,
          ``HF_TOKEN`` env passthrough, and ``-p {port}:{port}``. Idempotent -
          skips when a running container with the same name exists; reuses a
          stopped container when ``force=True``.
        - ``lifecycle``: One-call orchestration. ``lifecycle="full"`` runs
          ``build_image`` → ``download_checkpoint`` → ``start_container`` → ``start`` and
          waits for the inference port. ``lifecycle="teardown"`` removes the
          container (and its volumes when ``remove_volumes=True``). Each sub-step
          stays idempotent so re-running ``lifecycle="full"`` after a crash
          resumes from where it stopped.

    Protocol selection:
        - **ZMQ** (default, ``http_server=False``): Low-latency binary protocol on port 5555.
          Best for real-time robot control loops.
        - **HTTP** (``http_server=True``): REST API on port 8000 (auto-switched from 5555).
          Best for remote access, debugging, or multi-client scenarios.
          Endpoint: ``http://<host>:<port>/act``

    Data configs:
        The ``data_config`` parameter selects the embodiment-specific observation/action schema.
        Available configs (defined in ``data_configs.json``):

        **SO-100/101 arms:**
          ``so100``, ``so100_dualcam``, ``so100_4cam``,
          ``so101``, ``so101_dualcam``, ``so101_tricam``

        **Fourier GR1 humanoid:**
          ``fourier_gr1_arms_only``, ``fourier_gr1_arms_waist``,
          ``fourier_gr1_full_upper_body``

        **Unitree G1 humanoid:**
          ``unitree_g1_real`` (N1.7 REAL_G1 embodiment - locomotion + bimanual
          manipulation; PRETRAIN - works directly with the base model),
          ``unitree_g1`` [posttrain], ``unitree_g1_full_body`` [posttrain],
          ``unitree_g1_locomanip``,
          ``unitree_g1_sonic`` [posttrain] (SONIC whole-body controller - the
          VLA emits 64-dim SONIC motion-token latents, NOT executable joint
          commands; they must be decoded by the SONIC runtime from
          NVlabs/GR00T-WholeBodyControl. Requires a finetuned checkpoint, not
          the base nvidia/GR00T-N1.7-3B)

        **Franka Panda manipulators:**
          ``single_panda_gripper``, ``bimanual_panda_gripper``, ``bimanual_panda_hand``

        **Open X-Embodiment:**
          ``oxe_droid``, ``oxe_google``, ``oxe_widowx``

        **Simulation:**
          ``libero_panda`` [posttrain], ``libero_sim`` [posttrain],
          ``simpler_env_google`` [posttrain], ``simpler_env_widowx`` [posttrain]

        **AgiBOT:**
          ``agibot_genie1``, ``agibot_dual_arm_gripper`` (alias: ``agibot_dual_arm``),
          ``agibot_dual_arm_dexhand``, ``agibot_dual_arm_full``

        **Galaxea:**
          ``galaxea_r1_pro``

        .. note::
           Entries marked ``[posttrain]`` correspond to upstream
           ``POSTTRAIN_TAGS`` (Isaac-GR00T ``gr00t/data/embodiment_tags.py``):
           ``unitree_g1``, ``unitree_g1_sonic``, ``libero_panda`` / ``libero_sim``,
           ``simpler_env_google``, ``simpler_env_widowx``. They REQUIRE a
           finetuned checkpoint - pointing the base ``nvidia/GR00T-N1.7-3B`` at a
           posttrain tag silently emits garbage actions. Unmarked entries are
           pretrain tags baked into the base model and work directly.

    TensorRT acceleration:
        Set ``use_tensorrt=True`` to enable TensorRT inference. This compiles the model
        into an optimized engine on first run (may take several minutes). Subsequent runs
        load from ``trt_engine_path``. Dtype flags (``vit_dtype``, ``llm_dtype``, ``dit_dtype``)
        control precision - lower precision (fp8/nvfp4) trades accuracy for speed.

    Authentication:
        The ``api_token`` parameter authenticates with the inference service. If omitted,
        falls back to the ``GROOT_API_TOKEN`` environment variable.

    Server protocol versions:
        Isaac-GR00T's inference-service entrypoint and flag set changed between
        N1.6 and N1.7. The ``protocol`` parameter selects which command to
        ``docker exec``:

        - ``"n1.5"`` (default) and ``"n1.6"``: ``python /opt/Isaac-GR00T/scripts/inference_service.py``
          with ``--data-config`` + ``--denoising-steps`` flags. Matches the
          script that ships with images built before the N1.7 release.
        - ``"n1.7"``: ``python -m gr00t.eval.run_gr00t_server``. Drops
          ``--data-config`` and ``--denoising-steps`` (the server reads them
          from the model's metadata.json instead). Adds optional
          ``--use-sim-policy-wrapper`` for sim eval (LIBERO, RoboCasa, …)
          - pass ``use_sim_policy_wrapper=True`` to enable.

        The default stays ``"n1.5"`` for back-compat. N1.7 users must opt in
        explicitly: ``gr00t_inference(action="start", ..., protocol="n1.7")``.

    Args:
        action: Action to perform (see Actions above).
        checkpoint_path: Path to model checkpoint directory (required for ``start``/``restart``).
        policy_name: Optional name for the policy service (for registration/tracking).
        port: Port for the inference service. Defaults to 5555 (ZMQ) or auto-switches
            to 8000 when ``http_server=True``.
        data_config: Embodiment data config name (see Data configs above). N1.5/N1.6 only.
        embodiment_tag: Embodiment tag for the model (e.g., ``gr1``, ``so100``,
            ``libero_sim``).
        denoising_steps: Number of denoising steps for action generation (default: 4).
            N1.5/N1.6 only - the N1.7 server reads this from the checkpoint.
        host: Host address to bind the service to (default: ``0.0.0.0``).
        container_name: Specific Docker container name. Auto-detected if omitted.
        timeout: Seconds to wait for service startup (default: 60).
        use_tensorrt: Enable TensorRT acceleration (default: False).
        trt_engine_path: Directory for TensorRT engine cache (default: ``gr00t_engine``).
        vit_dtype: ViT precision with TensorRT - ``fp16`` or ``fp8`` (default: ``fp8``).
        llm_dtype: LLM precision with TensorRT - ``fp16``, ``nvfp4``, or ``fp8`` (default: ``nvfp4``).
        dit_dtype: DiT precision with TensorRT - ``fp16`` or ``fp8`` (default: ``fp8``).
        http_server: Use HTTP REST API instead of ZMQ (default: False).
        api_token: API token for authentication. Falls back to ``GROOT_API_TOKEN`` env var.
        protocol: Server protocol version - ``"n1.5"`` (default), ``"n1.6"``, or ``"n1.7"``.
            Determines which inference-service entrypoint and flag set is exec'd in
            the container. See "Server protocol versions" above.
        use_sim_policy_wrapper: When ``protocol="n1.7"``, append
            ``--use-sim-policy-wrapper`` to the server command. Required for sim
            evaluation (LIBERO, RoboCasa, …) - the wrapper translates
            simulator-side observations into the format the policy expects.
            Ignored for N1.5 / N1.6 (no equivalent flag).

    Container lifecycle args (used by ``build_image``, ``download_checkpoint``,
    ``start_container``, ``lifecycle``):
        (The build source repo/tag/clone-dir, the container image, bind-mount
        volumes, and the container command are operator-config-driven, NOT
        agent parameters: an agent-supplied git URL would clone an attacker
        tree and ``bash docker/build.sh`` it (host RCE). ``build_image`` clones
        ``$STRANDS_GR00T_REPO_URL`` (allowlisted; default the canonical NVIDIA
        repo) at ``$STRANDS_GR00T_REPO_TAG`` into ``$STRANDS_BASE_DIR/Isaac-GR00T``;
        the image is resolved from ``STRANDS_GR00T_IMAGE`` and validated against
        ``STRANDS_GR00T_IMAGE_ALLOW``. Extend the URL allowlist for private
        mirrors via ``STRANDS_GR00T_REPO_URL_ALLOW``.)
        hf_repo: HuggingFace dataset/model id (e.g., ``"nvidia/GR00T-N1.7-LIBERO"``).
            Required for ``download_checkpoint``.
        hf_subfolder: Subfolder pattern within the HF repo (e.g.,
            ``"libero_spatial"``). When set, only files matching
            ``<subfolder>/*`` are downloaded.
        hf_local_dir: Where to download the checkpoint. Defaults to
            ``$STRANDS_BASE_DIR/checkpoints/<basename(hf_repo)>``.
        hf_token: HuggingFace API token (gated repos). Falls back to
            ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env vars.
            Defaults to mounting ``hf_local_dir`` → ``/data/checkpoints`` and
            ``~/.cache/huggingface`` → ``/root/.cache/huggingface``.
        lifecycle: ``"full"`` (default - chain build → download → start_container
            → start) or ``"teardown"`` (rm container + volumes).
        remove_volumes: When ``lifecycle="teardown"``, also remove docker volumes
            (default: ``False`` to preserve checkpoint mounts).
        force: For idempotent steps - rebuild image, redownload checkpoint, or
            recreate container even when the artefact is already present.

    Returns:
        Dict with operation results. Common fields:

        - ``status``: ``"success"`` or ``"error"``
        - ``message``: Human-readable description

        For ``start``/``restart``:
          ``port``, ``checkpoint_path``, ``container_name``, ``protocol``,
          ``data_config``, ``embodiment_tag``, ``denoising_steps``,
          ``endpoint`` (HTTP only), ``tensorrt`` (if enabled)

        For ``status``:
          ``port``, ``service_status`` (``"running"`` or ``"not_running"``), ``protocol``

        For ``list``:
          ``services`` (list of ``{port, protocol, status}``)

        For ``find_containers``:
          ``containers`` (list of ``{name, image, status, ports}``)

    Examples:
        Start a ZMQ service for SO-100 dual-camera setup::

            gr00t_inference(
                action="start",
                checkpoint_path="/data/checkpoints/so100_model",
                data_config="so100_dualcam",
                embodiment_tag="so100",
            )

        Start an HTTP service with TensorRT::

            gr00t_inference(
                action="start",
                checkpoint_path="/data/checkpoints/gr1_model",
                http_server=True,
                use_tensorrt=True,
                data_config="fourier_gr1_arms_only",
            )

        Check service status and list running services::

            gr00t_inference(action="status", port=5555)
            gr00t_inference(action="list")

        Restart with a different checkpoint::

            gr00t_inference(
                action="restart",
                checkpoint_path="/data/checkpoints/gr1_model_v2",
                port=5555,
            )
    """
    # Resolve api_token from env var if not provided as parameter
    if api_token is None:
        api_token = os.environ.get("GROOT_API_TOKEN")

    # Validate protocol up-front so users get a friendly error rather than
    # an opaque docker-exec failure inside _start_service.
    valid_protocols = ("n1.5", "n1.6", "n1.7")
    if protocol not in valid_protocols:
        return {
            "status": "error",
            "message": f"Unknown protocol {protocol!r}. Valid: {list(valid_protocols)}",
        }

    # Boundary guard: hf_local_dir is an untrusted agent string that reaches
    # two host-fs sinks (checkpoint download writes to it directly; container
    # start bind-mounts it). Validate once here -- before any action branch
    # forwards it -- so a prompt-injected path is rejected for download,
    # start_container, AND lifecycle (which downloads before it starts).
    _hf_local_dir_reason = _check_hf_local_dir_safety(hf_local_dir)
    if _hf_local_dir_reason is not None:
        return {"status": "error", "message": _hf_local_dir_reason}

    if action == "find_containers":
        return _find_gr00t_containers()
    elif action == "list":
        return _list_running_services()
    elif action == "status":
        return _check_service_status(port)
    elif action == "stop":
        return _stop_service(port)
    elif action == "build_image":
        image_name = _resolve_image_name()
        if not _is_allowed_image(image_name):
            return {
                "status": "error",
                "message": (
                    f"configured image {image_name!r} is not in the allowlist "
                    f"{list(_image_allowlist())}. Set STRANDS_GR00T_IMAGE to an "
                    "allowed image or extend STRANDS_GR00T_IMAGE_ALLOW."
                ),
            }
        _build_source = _resolve_build_source()
        if isinstance(_build_source, dict):
            return _build_source
        _repo_url, _repo_tag = _build_source
        return _build_image(
            repo_url=_repo_url,
            repo_tag=_repo_tag,
            image_name=image_name,
            force=force,
        )
    elif action == "download_checkpoint":
        if hf_repo is None:
            return {"status": "error", "message": "'hf_repo' is required for action='download_checkpoint'"}
        return _download_checkpoint(
            hf_repo=hf_repo,
            hf_subfolder=hf_subfolder,
            hf_local_dir=hf_local_dir,
            hf_token=hf_token,
            force=force,
        )
    elif action == "start_container":
        image_name = _resolve_image_name()
        if not _is_allowed_image(image_name):
            return {
                "status": "error",
                "message": (
                    f"configured image {image_name!r} is not in the allowlist "
                    f"{list(_image_allowlist())}. Set STRANDS_GR00T_IMAGE to an "
                    "allowed image or extend STRANDS_GR00T_IMAGE_ALLOW."
                ),
            }
        # Container topology is not agent-controllable: the agent cannot
        # supply bind-mount volumes or a container command. _start_container
        # computes the default checkpoint + HF-cache mounts and runs the
        # keep-alive command so subsequent ``start`` actions can docker exec.
        return _start_container(
            image_name=image_name,
            container_name=container_name,
            port=port,
            volumes=None,
            hf_token=hf_token,
            container_command=_DEFAULT_CONTAINER_COMMAND,
            hf_local_dir=hf_local_dir,
            force=force,
        )
    elif action == "lifecycle":
        image_name = _resolve_image_name()
        if not _is_allowed_image(image_name):
            return {
                "status": "error",
                "message": (
                    f"configured image {image_name!r} is not in the allowlist "
                    f"{list(_image_allowlist())}. Set STRANDS_GR00T_IMAGE to an "
                    "allowed image or extend STRANDS_GR00T_IMAGE_ALLOW."
                ),
            }
        _build_source = _resolve_build_source()
        if isinstance(_build_source, dict):
            return _build_source
        _repo_url, _repo_tag = _build_source
        return _lifecycle(
            phase=lifecycle,
            # Phase-specific kwargs (most reused from start/start_container).
            repo_url=_repo_url,
            repo_tag=_repo_tag,
            image_name=image_name,
            hf_repo=hf_repo,
            hf_subfolder=hf_subfolder,
            hf_local_dir=hf_local_dir,
            hf_token=hf_token,
            container_name=container_name,
            volumes=None,
            container_command=_DEFAULT_CONTAINER_COMMAND,
            remove_volumes=remove_volumes,
            force=force,
            # start kwargs - used at the tail of phase="full".
            checkpoint_path=checkpoint_path,
            policy_name=policy_name,
            port=port,
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
            host=host,
            timeout=timeout,
            use_tensorrt=use_tensorrt,
            trt_engine_path=trt_engine_path,
            vit_dtype=vit_dtype,
            llm_dtype=llm_dtype,
            dit_dtype=dit_dtype,
            http_server=http_server,
            api_token=api_token,
            protocol=protocol,
            use_sim_policy_wrapper=use_sim_policy_wrapper,
        )
    elif action == "start":
        if checkpoint_path is None:
            return {"status": "error", "message": "Checkpoint path required to start service"}
        # HTTP server uses port 8000 by default
        if http_server and port == 5555:
            port = 8000
        return _start_service(
            checkpoint_path=checkpoint_path,
            port=port,
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
            host=host,
            container_name=container_name,
            policy_name=policy_name,
            timeout=timeout,
            use_tensorrt=use_tensorrt,
            trt_engine_path=trt_engine_path,
            vit_dtype=vit_dtype,
            llm_dtype=llm_dtype,
            dit_dtype=dit_dtype,
            http_server=http_server,
            api_token=api_token,
            protocol=protocol,
            use_sim_policy_wrapper=use_sim_policy_wrapper,
        )
    elif action == "restart":
        if checkpoint_path is None:
            return {"status": "error", "message": "Checkpoint path required for restart"}
        # Stop existing service and start new one
        _stop_service(port)
        time.sleep(2)  # Brief pause to allow port release before rebind
        return _start_service(
            checkpoint_path=checkpoint_path,
            port=port,
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
            host=host,
            container_name=container_name,
            policy_name=policy_name,
            timeout=timeout,
            use_tensorrt=use_tensorrt,
            trt_engine_path=trt_engine_path,
            vit_dtype=vit_dtype,
            llm_dtype=llm_dtype,
            dit_dtype=dit_dtype,
            http_server=http_server,
            api_token=api_token,
            protocol=protocol,
            use_sim_policy_wrapper=use_sim_policy_wrapper,
        )
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def _find_gr00t_containers() -> dict[str, Any]:
    """Find available Isaac-GR00T containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}"],
            capture_output=True,
            text=True,
            check=True,
        )

        containers = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) >= 3:
                    name, image, status = parts[0], parts[1], parts[2]
                    ports = parts[3] if len(parts) > 3 else ""

                    is_gr00t_container = "isaac-gr00t" in image.lower() or (
                        "isaac" in image.lower() and ("gr00t" in image.lower() or "jetson" in name.lower())
                    )

                    if is_gr00t_container:
                        containers.append({"name": name, "image": image, "status": status, "ports": ports})

        return {"status": "success", "containers": containers, "message": f"Found {len(containers)} GR00T containers"}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to find containers: {e}"}


def _list_running_services() -> dict[str, Any]:
    """List all running GR00T inference services by checking common ports."""
    try:
        services = []
        common_ports = [5555, 5556, 5557, 5558, 8000, 8001, 8002, 8003]

        for port in common_ports:
            if _is_service_running(port):
                protocol = "HTTP" if port >= 8000 else "ZMQ"
                services.append({"port": port, "protocol": protocol, "status": "running"})

        return {"status": "success", "services": services, "message": f"Found {len(services)} running services"}

    except Exception as e:
        return {"status": "error", "message": f"Failed to list services: {e}"}


def _is_service_running(port: int) -> bool:
    """Check if service is running on port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _check_service_status(port: int) -> dict[str, Any]:
    """Check status of service on specific port."""
    if _is_service_running(port):
        protocol = "HTTP" if port >= 8000 else "ZMQ"
        return {"status": "success", "port": port, "service_status": "running", "protocol": protocol}
    else:
        return {
            "status": "error",
            "port": port,
            "service_status": "not_running",
            "message": f"No service running on port {port}",
        }


def _stop_service(port: int) -> dict[str, Any]:
    """Stop GR00T inference service running on specific port."""
    try:
        containers_result = _find_gr00t_containers()
        if containers_result["status"] == "success":
            running_containers = [c for c in containers_result["containers"] if "Up" in c["status"]]

            for container in running_containers:
                container_name = container["name"]
                try:
                    result = subprocess.run(
                        ["docker", "exec", container_name, "pgrep", "-f", f"inference_service.py.*--port {port}"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        pids = result.stdout.strip().split("\n")
                        for pid in pids:
                            if pid:
                                subprocess.run(["docker", "exec", container_name, "kill", "-TERM", pid], check=True)

                        time.sleep(2)

                        result = subprocess.run(
                            ["docker", "exec", container_name, "pgrep", "-f", f"inference_service.py.*--port {port}"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )

                        if result.returncode == 0 and result.stdout.strip():
                            pids = result.stdout.strip().split("\n")
                            for pid in pids:
                                if pid:
                                    subprocess.run(["docker", "exec", container_name, "kill", "-KILL", pid], check=True)

                        return {
                            "status": "success",
                            "port": port,
                            "container": container_name,
                            "message": f"GR00T service on port {port} stopped in container {container_name}",
                        }

                except subprocess.CalledProcessError:
                    continue

        # Fallback: try host system
        result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)

        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    subprocess.run(["kill", "-TERM", pid], check=True)

            time.sleep(2)

            result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)

            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid:
                        subprocess.run(["kill", "-KILL", pid], check=True)

            return {"status": "success", "port": port, "message": f"Service on port {port} stopped"}
        else:
            return {"status": "success", "port": port, "message": f"No service running on port {port}"}

    except Exception as e:
        return {"status": "error", "message": f"Failed to stop service: {e}"}


def _build_inference_command(
    *,
    container_name: str,
    checkpoint_path: str,
    port: int,
    host: str,
    data_config: str,
    embodiment_tag: str,
    denoising_steps: int,
    http_server: bool,
    use_tensorrt: bool,
    trt_engine_path: str,
    vit_dtype: str,
    llm_dtype: str,
    dit_dtype: str,
    api_token: str | None,
    protocol: str,
    use_sim_policy_wrapper: bool,
) -> list[str]:
    """Build the ``docker exec`` argv for the inference service.

    Two entrypoint scripts ship with Isaac-GR00T:

    * ``/opt/Isaac-GR00T/scripts/inference_service.py`` (N1.5, N1.6) -
      standalone server with embodiment data-config + denoising-steps
      flags.
    * ``python -m gr00t.eval.run_gr00t_server`` (N1.7) - rewritten
      entrypoint that reads data-config + denoising-steps from the
      checkpoint metadata and adds an optional ``--use-sim-policy-wrapper``
      flag for sim eval (LIBERO, RoboCasa, …).

    Both share ``--server``, ``--model-path``, ``--port``, ``--host``,
    ``--embodiment-tag``, ``--api-token``, and the TensorRT flag set.
    The split keeps the ``protocol`` branch shallow - one ``if`` per
    diverging flag rather than two parallel command-builder functions.
    """
    if protocol == "n1.7":
        # The N1.7 entrypoint (``python -m gr00t.eval.run_gr00t_server``)
        # does NOT accept a ``--server`` flag — passing it makes ``tyro``
        # reject the invocation with ``Unrecognized options: --server``
        # and the inference process exits before binding the port. The
        # legacy N1.5/N1.6 ``inference_service.py`` did take ``--server``;
        # keeping the flag there preserves back-compat for older images.
        cmd = [
            "docker",
            "exec",
            "-d",
            container_name,
            "python",
            "-m",
            "gr00t.eval.run_gr00t_server",
            "--model-path",
            checkpoint_path,
            "--port",
            str(port),
            "--host",
            host,
            "--embodiment-tag",
            embodiment_tag,
        ]
        if use_sim_policy_wrapper:
            cmd.append("--use-sim-policy-wrapper")
    else:  # n1.5 / n1.6
        cmd = [
            "docker",
            "exec",
            "-d",
            container_name,
            "python",
            "/opt/Isaac-GR00T/scripts/inference_service.py",
            "--server",
            "--model-path",
            checkpoint_path,
            "--port",
            str(port),
            "--host",
            host,
            "--data-config",
            data_config,
            "--embodiment-tag",
            embodiment_tag,
            "--denoising-steps",
            str(denoising_steps),
        ]

    # Shared optional flags - apply to every protocol.
    if http_server:
        cmd.append("--http-server")

    if use_tensorrt:
        cmd.extend(
            [
                "--use-tensorrt",
                "--trt-engine-path",
                trt_engine_path,
                "--vit-dtype",
                vit_dtype,
                "--llm-dtype",
                llm_dtype,
                "--dit-dtype",
                dit_dtype,
            ]
        )

    if api_token:
        cmd.extend(["--api-token", api_token])

    return cmd


def _start_service(
    checkpoint_path: str,
    port: int,
    data_config: str,
    embodiment_tag: str,
    denoising_steps: int,
    host: str,
    container_name: str | None,
    policy_name: str | None,
    timeout: int,
    use_tensorrt: bool,
    trt_engine_path: str,
    vit_dtype: str,
    llm_dtype: str,
    dit_dtype: str,
    http_server: bool,
    api_token: str | None,
    protocol: str = "n1.5",
    use_sim_policy_wrapper: bool = False,
) -> dict[str, Any]:
    """Start GR00T inference service using Isaac-GR00T's native inference service."""
    try:
        # Find container if not specified
        if container_name is None:
            containers = _find_gr00t_containers()
            if containers["status"] == "error":
                return containers

            running_containers = [c for c in containers["containers"] if "Up" in c["status"]]
            if not running_containers:
                return {"status": "error", "message": "No running GR00T containers found"}

            container_name = running_containers[0]["name"]

        cmd = _build_inference_command(
            container_name=container_name,
            checkpoint_path=checkpoint_path,
            port=port,
            host=host,
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
            http_server=http_server,
            use_tensorrt=use_tensorrt,
            trt_engine_path=trt_engine_path,
            vit_dtype=vit_dtype,
            llm_dtype=llm_dtype,
            dit_dtype=dit_dtype,
            api_token=api_token,
            protocol=protocol,
            use_sim_policy_wrapper=use_sim_policy_wrapper,
        )

        # Start service
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Wait for service to start
        wire_protocol = "HTTP" if http_server else "ZMQ"
        start_time = time.time()
        while time.time() - start_time < timeout:
            if _is_service_running(port):
                response: dict[str, Any] = {
                    "status": "success",
                    "port": port,
                    "checkpoint_path": checkpoint_path,
                    "container_name": container_name,
                    "policy_name": policy_name,
                    "protocol": wire_protocol,
                    "server_protocol": protocol,
                    "embodiment_tag": embodiment_tag,
                    "message": f"GR00T {wire_protocol} service started on port {port} (server: {protocol})",
                }
                # Server flags that only apply to the legacy entrypoint -
                # surface them only when actually used so the response
                # accurately reflects what was passed.
                if protocol != "n1.7":
                    response["data_config"] = data_config
                    response["denoising_steps"] = denoising_steps
                else:
                    response["use_sim_policy_wrapper"] = use_sim_policy_wrapper
                if use_tensorrt:
                    response["tensorrt"] = {
                        "enabled": True,
                        "engine_path": trt_engine_path,
                        "vit_dtype": vit_dtype,
                        "llm_dtype": llm_dtype,
                        "dit_dtype": dit_dtype,
                    }
                if http_server:
                    response["endpoint"] = f"http://{host}:{port}/act"
                return response
            time.sleep(1)

        return {"status": "error", "message": f"{wire_protocol} service failed to start within {timeout} seconds"}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to start service: {e.stderr or e}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}


# Container lifecycle helpers (#148-F3 wider)
#
# Each helper is idempotent and returns a structured status dict. They wrap
# the manual four-step Isaac-GR00T setup (clone → docker build → hf
# download → docker run) so an LLM driving this AgentTool can fully
# orchestrate a GR00T eval from one prompt. Splitting them into per-action
# entry points (rather than burying everything inside ``start``) keeps
# each step independently re-runnable and makes the failure surface
# obvious - if "build_image" succeeds but "download_checkpoint" fails,
# the user / agent knows exactly which step to retry.


def _image_exists(image_name: str) -> bool:
    """Return True iff the local docker daemon already has ``image_name``.

    Uses ``docker image inspect`` rather than ``docker images`` because the
    former returns a non-zero exit code on miss (cleaner branch logic) and
    works for tag-less digest pins too. Any docker invocation failure
    (daemon down, command missing) returns False so the caller falls
    through to a regular build, which then surfaces the real error.
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False


def _container_state(name: str) -> str:
    """Return the current state of container ``name`` or ``"absent"``.

    Possible values: ``"running"``, ``"exited"``, ``"created"``, ``"paused"``,
    ``"restarting"``, ``"removing"``, ``"dead"``, ``"absent"``. Anything
    other than ``"running"`` and ``"absent"`` is unusual and the caller
    must decide whether to remove + recreate (``force=True``) or fail.
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return "absent"
        return result.stdout.strip() or "absent"
    except (FileNotFoundError, OSError):
        return "absent"


def _build_image(
    *,
    repo_url: str,
    repo_tag: str,
    image_name: str,
    force: bool,
) -> dict[str, Any]:
    """Clone Isaac-GR00T at ``repo_tag`` and run ``bash docker/build.sh``.

    Idempotent: when ``image_name`` is already in the local docker daemon
    AND ``force=False``, returns success without touching the filesystem
    or the docker daemon. Pass ``force=True`` to clean-rebuild.

    Defence in depth: ``repo_url``/``repo_tag`` are resolved+validated at the
    dispatch boundary (the agent cannot supply them), but this private entry
    point is reachable by operators/tests, so it re-asserts the same URL
    allowlist + tag-shape guard before any ``git``/``bash`` subprocess. The
    clone destination is fixed (``_isaac_gr00t_dir()``), never caller-supplied.
    """
    if not _is_allowed_repo_url(repo_url):
        return {
            "status": "error",
            "message": (
                f"refusing to clone {repo_url!r}: not in the repo-URL allowlist {list(_repo_url_allowlist())}."
            ),
        }
    if not _is_allowed_repo_tag(repo_tag):
        return {
            "status": "error",
            "message": f"refusing to clone tag {repo_tag!r}: not a valid git ref.",
        }
    if not force and _image_exists(image_name):
        return {
            "status": "success",
            "image_name": image_name,
            "skipped": True,
            "message": f"Docker image {image_name!r} already exists; skipping build (use force=True to rebuild)",
        }

    dest = _isaac_gr00t_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Clone or update the repo at the requested tag. ``git fetch + checkout``
        # is faster than re-cloning when the branch has just moved.
        if (dest / ".git").is_dir():
            subprocess.run(
                ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", repo_tag],
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(dest), "checkout", repo_tag],
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"],
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    repo_tag,
                    "--recurse-submodules",
                    repo_url,
                    str(dest),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

        # Build the image. ``docker/build.sh`` is the canonical entrypoint
        # documented in the Isaac-GR00T README. Use bash explicitly so the
        # script's shebang isn't required.
        build_script = dest / "docker" / "build.sh"
        if not build_script.is_file():
            return {
                "status": "error",
                "message": (
                    f"docker/build.sh not found at {build_script} - the cloned repo may "
                    f"not match the expected layout for tag {repo_tag!r}."
                ),
            }
        subprocess.run(
            ["bash", str(build_script)],
            cwd=str(dest),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "DOCKER_BUILDKIT": os.environ.get("DOCKER_BUILDKIT", "1")},
        )

        return {
            "status": "success",
            "image_name": image_name,
            "source_dir": str(dest),
            "repo_tag": repo_tag,
            "skipped": False,
            "message": f"Built docker image {image_name!r} from {repo_url}@{repo_tag}",
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Failed to build image: {e.stderr or e}",
            "stderr": e.stderr or "",
        }


def _download_checkpoint(
    *,
    hf_repo: str,
    hf_subfolder: str | None,
    hf_local_dir: str | None,
    hf_token: str | None,
    force: bool,
) -> dict[str, Any]:
    """Download a HuggingFace checkpoint via ``huggingface_hub.snapshot_download``.

    Idempotent: when ``local_dir`` already exists and is non-empty AND
    ``force=False``, returns success without touching the network. Pass
    ``force=True`` to refresh.

    HF token resolution order:
        1. Explicit ``hf_token`` kwarg.
        2. ``HF_TOKEN`` env var (canonical, what ``huggingface_hub`` reads).
        3. ``HUGGING_FACE_HUB_TOKEN`` env var (legacy alias).
        4. None - downloads continue for ungated repos; gated ones surface
           a clear ``snapshot_download`` error.
    """
    # Defence in depth: hf_local_dir is written to directly on the host here
    # (no docker mediation). The agent dispatch validates it, but guard the
    # internal/operator/test entry point too so a future caller that reaches
    # _download_checkpoint without going through the tool inherits the check.
    _hf_dir_reason = _check_hf_local_dir_safety(hf_local_dir)
    if _hf_dir_reason is not None:
        return {"status": "error", "message": _hf_dir_reason}

    local_dir = Path(hf_local_dir).expanduser() if hf_local_dir else _checkpoints_dir() / hf_repo.replace("/", "__")

    if not force and local_dir.is_dir() and any(local_dir.iterdir()):
        return {
            "status": "success",
            "hf_repo": hf_repo,
            "hf_subfolder": hf_subfolder,
            "local_dir": str(local_dir),
            "skipped": True,
            "message": f"Checkpoint already at {local_dir}; skipping download (use force=True to refresh)",
        }

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        hub = require_optional(
            "huggingface_hub",
            pip_install="huggingface_hub",
            purpose="downloading GR00T checkpoints",
        )
    except ImportError as e:
        return {"status": "error", "message": str(e)}

    local_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [f"{hf_subfolder}/*"] if hf_subfolder else None

    try:
        hub.snapshot_download(  # type: ignore[attr-defined]
            repo_id=hf_repo,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            token=token,
        )
    except Exception as e:  # noqa: BLE001 - HF errors are opaque + varied
        return {"status": "error", "message": f"Failed to download {hf_repo!r}: {e}"}

    return {
        "status": "success",
        "hf_repo": hf_repo,
        "hf_subfolder": hf_subfolder,
        "local_dir": str(local_dir),
        "skipped": False,
        "message": f"Downloaded {hf_repo}{('/' + hf_subfolder) if hf_subfolder else ''} → {local_dir}",
    }


def _start_container(
    *,
    image_name: str,
    container_name: str | None,
    port: int,
    volumes: dict[str, str] | None,
    hf_token: str | None,
    container_command: str,
    hf_local_dir: str | None,
    force: bool,
) -> dict[str, Any]:
    """``docker run -d`` the GR00T container so subsequent ``start`` actions can
    ``docker exec`` into it.

    Idempotent: when a container with ``container_name`` is already
    running, returns success without touching docker. When it exists but
    is stopped, ``force=True`` removes + recreates it (otherwise returns
    an error that names the recovery flag).

    Defence in depth: although the agent-facing tool no longer lets a
    caller supply ``image_name``, ``volumes``, or ``container_command``,
    this private entry point is still reachable by operators and tests. We
    validate the image against the allowlist and reject any bind-mount of a
    protected host path (root fs, system dirs, credential dirs, docker
    socket) before building the ``docker run`` argv. This closes the
    host-mount / container-escape surface even for the internal path.
    """
    if not _is_allowed_image(image_name):
        return {
            "status": "error",
            "message": (
                f"image {image_name!r} is not in the allowlist {list(_image_allowlist())}. "
                "Set STRANDS_GR00T_IMAGE / STRANDS_GR00T_IMAGE_ALLOW to permit it."
            ),
        }
    _vol_reason = _check_volume_safety(volumes)
    if _vol_reason is not None:
        return {"status": "error", "message": _vol_reason}
    name = container_name or "gr00t"
    state = _container_state(name)
    if state == "running" and not force:
        return {
            "status": "success",
            "container_name": name,
            "image_name": image_name,
            "state": state,
            "skipped": True,
            "message": f"Container {name!r} already running; skipping (use force=True to recreate)",
        }
    if state not in ("absent", "running") and not force:
        return {
            "status": "error",
            "message": (f"Container {name!r} exists in state {state!r}; pass force=True to remove + recreate"),
        }
    if state != "absent":
        # force=True OR running-but-force-set: remove first.
        _remove_container(name=name, remove_volumes=False)

    # Build the docker-run argv.
    cmd: list[str] = [
        "docker",
        "run",
        "-d",
        "--gpus",
        "all",
        "--ipc=host",
        "--name",
        name,
        "-p",
        f"{port}:{port}",
    ]

    # Default volume layout: mount the checkpoint dir into /data/checkpoints
    # and the host's HF cache so `huggingface_hub` reuses already-downloaded
    # snapshots. Override with explicit ``volumes={...}`` to customise.
    effective_volumes = dict(volumes) if volumes is not None else {}
    if not volumes:
        if hf_local_dir:
            effective_volumes[str(Path(hf_local_dir).expanduser())] = "/data/checkpoints"
        else:
            effective_volumes[str(_checkpoints_dir())] = "/data/checkpoints"
        hf_cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        effective_volumes[hf_cache] = "/root/.cache/huggingface"

    # Defence in depth: validate agent-supplied paths in effective_volumes.
    # The initial _check_volume_safety(volumes) above only sees the caller-
    # supplied dict (None from the agent path). The hf_local_dir parameter
    # is agent-supplied and flows into effective_volumes -- validate it
    # before building the docker argv so a prompt-injected path like '/etc'
    # is caught by the same prefix-match guard. We check only agent-supplied
    # entries (hf_local_dir) rather than the full dict, because auto-derived
    # paths (HF_HOME / ~/.cache/huggingface) are operator-controlled and may
    # legitimately reside under /home.
    _hf_dir_reason = _check_hf_local_dir_safety(hf_local_dir)
    if _hf_dir_reason is not None:
        return {"status": "error", "message": _hf_dir_reason}

    for host_path, container_path in effective_volumes.items():
        cmd.extend(["-v", f"{host_path}:{container_path}"])

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        cmd.extend(["-e", f"HF_TOKEN={token}"])

    cmd.append(image_name)
    # Split on whitespace to support both string + list-style commands while
    # keeping the simple "tail -f /dev/null" default working.
    if container_command:
        cmd.extend(container_command.split())

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"docker run failed: {e.stderr or e}"}

    return {
        "status": "success",
        "container_name": name,
        "image_name": image_name,
        "port": port,
        "volumes": effective_volumes,
        "skipped": False,
        "message": f"Started container {name!r} from {image_name!r}",
    }


def _remove_container(*, name: str, remove_volumes: bool) -> dict[str, Any]:
    """``docker rm -f`` the container; optionally also remove its volumes.

    Tolerant of missing containers - returns success with ``skipped=True``
    when the name is unknown so ``lifecycle="teardown"`` is safe to call
    multiple times.
    """
    state = _container_state(name)
    if state == "absent":
        return {
            "status": "success",
            "container_name": name,
            "skipped": True,
            "message": f"Container {name!r} not present; nothing to remove",
        }

    cmd = ["docker", "rm", "-f"]
    if remove_volumes:
        cmd.append("-v")
    cmd.append(name)

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"docker rm failed: {e.stderr or e}"}

    return {
        "status": "success",
        "container_name": name,
        "skipped": False,
        "remove_volumes": remove_volumes,
        "message": f"Removed container {name!r}" + (" (with volumes)" if remove_volumes else ""),
    }


def _lifecycle(
    *,
    phase: str,
    # build_image kwargs
    repo_url: str,
    repo_tag: str,
    image_name: str,
    # download_checkpoint kwargs
    hf_repo: str | None,
    hf_subfolder: str | None,
    hf_local_dir: str | None,
    hf_token: str | None,
    # start_container kwargs
    container_name: str | None,
    volumes: dict[str, str] | None,
    container_command: str,
    # teardown
    remove_volumes: bool,
    # shared
    force: bool,
    # start kwargs (for phase="full" tail)
    checkpoint_path: str | None,
    policy_name: str | None,
    port: int,
    data_config: str,
    embodiment_tag: str,
    denoising_steps: int,
    host: str,
    timeout: int,
    use_tensorrt: bool,
    trt_engine_path: str,
    vit_dtype: str,
    llm_dtype: str,
    dit_dtype: str,
    http_server: bool,
    api_token: str | None,
    protocol: str,
    use_sim_policy_wrapper: bool,
) -> dict[str, Any]:
    """Orchestrate the four-step setup or tear down a previously-started container.

    ``phase="full"``: ``build_image`` → ``download_checkpoint`` →
    ``start_container`` → ``start`` → wait-for-port. Each sub-step is
    idempotent so re-runs after a crash resume from the failed step.

    ``phase="teardown"``: ``_remove_container`` (with optional volume
    removal). The image and downloaded checkpoint are preserved -
    teardown is intentionally cheap to re-run.
    """
    if phase not in ("full", "teardown"):
        return {"status": "error", "message": f"Unknown lifecycle phase {phase!r}. Valid: ['full', 'teardown']"}

    steps: list[dict[str, Any]] = []

    if phase == "teardown":
        rm_result = _remove_container(name=container_name or "gr00t", remove_volumes=remove_volumes)
        steps.append({"step": "remove_container", "result": rm_result})
        return {
            "status": rm_result["status"],
            "phase": phase,
            "steps": steps,
            "message": rm_result["message"],
        }

    # phase == "full"
    if hf_repo is None:
        return {
            "status": "error",
            "message": "lifecycle='full' requires 'hf_repo' so the checkpoint can be downloaded",
        }

    build_result = _build_image(
        repo_url=repo_url,
        repo_tag=repo_tag,
        image_name=image_name,
        force=force,
    )
    steps.append({"step": "build_image", "result": build_result})
    if build_result["status"] != "success":
        return {"status": "error", "phase": phase, "steps": steps, "message": "lifecycle aborted: build_image failed"}

    download_result = _download_checkpoint(
        hf_repo=hf_repo,
        hf_subfolder=hf_subfolder,
        hf_local_dir=hf_local_dir,
        hf_token=hf_token,
        force=force,
    )
    steps.append({"step": "download_checkpoint", "result": download_result})
    if download_result["status"] != "success":
        return {
            "status": "error",
            "phase": phase,
            "steps": steps,
            "message": "lifecycle aborted: download_checkpoint failed",
        }

    # The container needs to mount the just-downloaded checkpoint. If the
    # caller didn't pre-resolve ``hf_local_dir``, propagate the path the
    # download step actually used so the container can see the files.
    resolved_local_dir = download_result.get("local_dir") or hf_local_dir

    container_result = _start_container(
        image_name=image_name,
        container_name=container_name,
        port=port,
        volumes=volumes,
        hf_token=hf_token,
        container_command=container_command,
        hf_local_dir=resolved_local_dir,
        force=force,
    )
    steps.append({"step": "start_container", "result": container_result})
    if container_result["status"] != "success":
        return {
            "status": "error",
            "phase": phase,
            "steps": steps,
            "message": "lifecycle aborted: start_container failed",
        }

    # The inference service expects to see the checkpoint mounted under
    # /data/checkpoints (the default volume mapping). Translate the host
    # path the user / download step gave us into the in-container path.
    mounted_checkpoint = checkpoint_path
    if mounted_checkpoint is None and hf_subfolder:
        mounted_checkpoint = f"/data/checkpoints/{hf_subfolder}"
    if mounted_checkpoint is None:
        return {
            "status": "error",
            "phase": phase,
            "steps": steps,
            "message": (
                "lifecycle='full' needs either 'checkpoint_path' (in-container path) or "
                "'hf_subfolder' (auto-resolved to /data/checkpoints/<subfolder>) so the inference "
                "service knows where to load the model from"
            ),
        }

    start_result = _start_service(
        checkpoint_path=mounted_checkpoint,
        port=port,
        data_config=data_config,
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
        host=host,
        container_name=container_result["container_name"],
        policy_name=policy_name,
        timeout=timeout,
        use_tensorrt=use_tensorrt,
        trt_engine_path=trt_engine_path,
        vit_dtype=vit_dtype,
        llm_dtype=llm_dtype,
        dit_dtype=dit_dtype,
        http_server=http_server,
        api_token=api_token,
        protocol=protocol,
        use_sim_policy_wrapper=use_sim_policy_wrapper,
    )
    steps.append({"step": "start", "result": start_result})

    return {
        "status": start_result["status"],
        "phase": phase,
        "steps": steps,
        "message": start_result.get("message", "lifecycle complete"),
    }


if __name__ == "__main__":
    print("🐳 GR00T Inference Service Manager (Isaac-GR00T Native)")
    print("Supports ZMQ, HTTP, and TensorRT inference modes")
    print()
    print("Examples:")
    print("  # Start ZMQ server (default)")
    print("  gr00t_inference(action='start', checkpoint_path='/data/checkpoints/model', port=5555)")
    print()
    print("  # Start HTTP server")
    print("  gr00t_inference(action='start', checkpoint_path='/data/checkpoints/model', port=8000, http_server=True)")
    print()
    print("  # Start with TensorRT acceleration")
    print("  gr00t_inference(action='start', checkpoint_path='/data/checkpoints/model', port=5555, use_tensorrt=True)")
    print()
    print("  # Start HTTP + TensorRT")
    print(
        "  gr00t_inference(action='start', checkpoint_path='/data/checkpoints/model',"
        " port=8000, http_server=True, use_tensorrt=True)"
    )
