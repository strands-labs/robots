"""Pydantic-free config for the VERA policy provider.

VERA (Video-to-Embodied Robot Action, MIT/CSAIL) is a two-stage video-to-action
policy: an embodiment-agnostic **video planner** (DFoT / WAN) dreams future
frames, and an embodiment-specific **Jacobian IDM** translates the dream into
robot actions. The two stages run inside a single websocket policy server
(``vera.server.start_vera_server``); this provider is a typed client + a managed
server subprocess, mirroring the ``cosmos3`` provider's service pattern.

The config mirrors VERA's server flags 1:1 and is env-overridable so a rollout
can be driven entirely from the environment (CI / fleet) without code changes.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Literal, get_args

from strands_robots.utils import (
    dial_host_error,
    positive_finite_number_error,
    positive_whole_number_error,
    tcp_port_error,
)

Embodiment = Literal["pusht", "mimicgen", "allegro", "droid"]
# "mimicgen" is the working, faithful embodiment end-to-end (eef-delta -> IK
# onto a real arm). "pusht" is experimental: its server runs, but VERA's IDM
# "du" action path for pusht is not wired end-to-end upstream (documented in
# VERA's configurations/dataset/pusht.yaml), so it validates the
# provider -> server -> action plumbing rather than producing a solving
# rollout. "allegro"/"droid" are code-present but checkpoint-absent (Wave 2).

# The embodiments this provider knows, derived from the type alias rather than
# re-listed, so an embodiment added to :data:`Embodiment` participates in the
# refusal below on arrival instead of being silently absorbed by a fallback.
_EMBODIMENTS: tuple[str, ...] = get_args(Embodiment)


# Per-embodiment default ports (policy, viz) - match the VERA examples
# (PushT uses 8820/8821; everything else uses 8800/8801).
# Per-embodiment per-view render width the VERA WAN/DFoT planner expects. The
# server does NOT advertise this (image_resolution is None); it is the client's
# job to send each view at this width (matching VERA's own RemotePolicy
# render_size). pusht: single 252-wide view (PushTImageEnv default); mimicgen:
# 128/view (run_mimicgen_eval default); droid/allegro follow upstream eval.
_DEFAULT_RENDER_WIDTH: dict[str, int] = {
    "pusht": 252,
    "mimicgen": 128,
    "droid": 128,
    "allegro": 128,
}

# Seconds the readiness wait allows the server to open its websocket. A WAN
# model load is slow, so the budget is generous; it is a default rather than a
# bound, and ``VERA_SERVER_READY_TIMEOUT`` overrides it.
_DEFAULT_SERVER_READY_TIMEOUT = 600.0

_DEFAULT_PORTS: dict[str, tuple[int, int]] = {
    "pusht": (8820, 8821),
    "mimicgen": (8800, 8801),
    "allegro": (8802, 8803),
    "droid": (8804, 8805),
}


def _env(*names: str) -> str | None:
    """Return the first set (non-empty) environment variable among ``names``."""
    for n in names:
        v = os.environ.get(n)
        if v is not None and v.strip() != "":
            return v
    return None


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _embodiment_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not an embodiment this provider knows.

    The embodiment is not one knob among the others: it is the field the other
    per-embodiment defaults are *looked up by*. It selects both ports, the
    per-view render width, the checkpoint-root variable that is probed, the
    container name and the ``--embodiment`` flag the server itself is launched
    with, so a spelling no table has an entry for is not a single wrong value -
    it is a whole configuration assembled from whatever each of those six
    readers does with an unknown key.

    Refusing it here rather than downstream is what the package already does on
    the other side of the container boundary: ``docker/entrypoint.sh`` ends its
    per-embodiment ``case`` with ``ERROR: unknown embodiment`` and ``exit 2``,
    listing the same four names. That refusal cannot stand in for this one,
    because it is only reached in ``server_mode="docker"`` after an image has
    been started, and it never runs at all for the subprocess runner or for a
    server that is merely dialed (``auto_launch_server=False``).

    Args:
        value: The caller-supplied embodiment.
        param: The field name it came from, used in the message.
        context: Message prefix identifying the surface that received it.

    Returns:
        An error message, or ``None`` when the value is a known embodiment.
    """
    if value in _EMBODIMENTS:
        return None
    return (
        f"{context}: {param} must be one of {', '.join(map(repr, _EMBODIMENTS))}, got {value!r}. "
        "The embodiment selects the planner/IDM pair, both default ports and the render width, "
        "so an unknown one cannot be resolved to a configuration."
    )


def _viewer_port_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` cannot address the MJPEG live-viewer port.

    :func:`~strands_robots.utils.tcp_port_error` owns the range and the scalar
    policy; this wrapper decides only the floor, because ``vis_port`` documents
    ``0`` as "disable the live viewer" rather than as a port to bind - the
    runner omits the ``--vis-port`` flag entirely for a falsy value. A genuine
    ``int`` zero is therefore accepted here and every other value is deferred,
    so the two ports cannot drift apart on what counts as an addressable one.

    ``bool`` is deliberately not spelled in the zero test: ``False == 0``, so a
    bare ``value == 0`` would read ``False`` as the disable spelling. The type
    identity is checked first, and a boolean falls through to
    :func:`~strands_robots.utils.tcp_port_error`, which refuses it for the same
    reason it refuses one for ``server_port``.

    Args:
        value: The caller-supplied value, after the default has been applied.
        param: The field name it came from, used in the message.
        context: Message prefix identifying the surface that received it.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value == 0:
        return None
    return tcp_port_error(value, param, context)


def _env_float(name: str) -> float | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


@dataclasses.dataclass
class VeraConfig:
    """Runtime configuration for :class:`VeraPolicy`.

    Every field maps to a VERA server flag / env var. Construction applies
    environment overrides last so callers can pin values in code and still let
    the environment win for deploy-time knobs (ports, checkpoint roots).

    Args:
        embodiment: VERA embodiment - selects the WAN/DFoT planner + Jacobian
            IDM pair and the client-side action adapter. Must be one of
            ``pusht``, ``mimicgen``, ``allegro``, ``droid`` (the members of
            :data:`Embodiment`); any other spelling is refused, because this
            field is the key every other per-embodiment default is looked up by
            and an unknown one would otherwise resolve to another embodiment's
            ports and render width.
        host: Policy-server hostname or IP literal, and the host half of
            :attr:`server_uri`. Must be a bare host a URI can carry - no ``/``,
            ``?``, ``#``, ``@``, ``:`` (outside a bracketed IPv6 literal such as
            ``[::1]``), whitespace or control character - because those end the
            host inside ``ws://<host>:<port>`` and the port is what they take:
            ``"127.0.0.1/foo"`` dials port 80, discarding the port the shared
            TCP-port domain just accepted. ``""`` is refused with ``"0.0.0.0"``
            named as the spelling that reaches a server bound on every
            interface. Whether the host resolves is left to the readiness probe,
            which is the surface that can observe it.
        server_port: Policy-server websocket port. ``None`` applies
            ``VERA_SERVER_PORT`` else the per-embodiment default; any other
            value must be an ``int`` in ``[1, 65535]``, because the client dials
            it and the server binds it. That includes a value the environment
            supplied: ``VERA_SERVER_PORT=0`` is refused rather than read as
            "unset", so it cannot resolve to the default under a success.
        vis_port: MJPEG live-viewer port. ``None`` applies ``VERA_VIS_PORT``
            else the per-embodiment default; ``0`` disables the viewer (the
            runner omits ``--vis-port``); any other value must be an ``int`` in
            ``[1, 65535]``. ``0`` means the same through the environment as it
            does through the keyword, which is why the override is read for its
            presence and not for its truth.
        render_width: Per-view width, in pixels, each camera frame is resized
            to before it is sent to the planner. ``None`` applies
            ``VERA_RENDER_WIDTH`` else the per-embodiment default; any other
            value must be a positive whole number of pixels - the shared media
            domain (:func:`~strands_robots.utils.positive_whole_number_error`)
            that the recorders' ``width``/``height`` and
            :class:`~strands_robots.rendering.HybridCompositor` already use.
        algo_config: WAN planner ``algo_config.yaml`` path. Point at the omni
            config to swap the planner without retraining the IDM.
        dynamics_run_id: Jacobian/IDM checkpoint id (wandb run id); falls back
            to the embodiment's in-tree default when unset.
        text_prompt: Optional text conditioning for the video planner.
        ckpt_root: Root of the downloaded VERA checkpoints
            (``hf download sizhe-lester-li/VERA --local-dir …``). Exported to
            ``VERA_CKPT_ROOT`` for the server subprocess.
        sample_steps: WAN denoise steps (deploy default is 10; ``None`` uses the
            planner yaml's value). A supplied count must be a positive whole
            number - the shared domain ``render_width`` takes - and is converted
            to ``int``, because it reaches the server only as the text of
            ``--sample-steps`` / ``VERA_SAMPLE_STEPS`` and ``str(10.0)`` is not a
            token that flag can parse.
        tracker_backend: IDM point tracker backend override.
        motion_plan_scale: IDM motion-plan scale override (live-tunable).
            ``None`` - the default, and what an unset ``VERA_MOTION_PLAN_SCALE``
            resolves to - leaves the server's own scale alone. Any other value
            must be a positive finite number, because it multiplies the motion
            plan the IDM turns into actions.
        teacache: Enable the near-lossless DiT teacache speedup (default True).
        teacache_thresh: teacache rel_l1 threshold (>0.15 hits a quality cliff,
            which is guidance rather than a bound). Must be a positive finite
            number - the shared domain ``motion_plan_scale`` takes - checked
            whatever ``teacache`` is set to, since that flag can be turned on
            after construction. Use ``teacache=False`` to switch the cache off;
            ``0`` is not that opt-out. Carried to the server by both launch
            modes: as ``--teacache-thresh`` on the subprocess argv, and as
            ``-e VERA_TEACACHE_THRESH`` for the container, which the entrypoint
            turns back into the same flag. Only the ``teacache`` off-switch used
            to be forwarded to the container, so a tuned threshold applied under
            ``server_mode="subprocess"`` and was silently dropped under
            ``server_mode="docker"``.
        auto_launch_server: Launch + manage the server subprocess on first use.
        server_ready_timeout: Seconds the readiness wait allows the server
            websocket to come up before raising (WAN model load can be slow).
            ``None`` applies ``VERA_SERVER_READY_TIMEOUT`` else 600; any other
            value must be a positive finite number of seconds, the shared
            continuous-span domain
            (:func:`~strands_robots.utils.positive_finite_number_error`) a
            ``duration`` in seconds already takes. The environment spelling is
            the one both readiness timeouts name as the remedy, so it is read
            here rather than nowhere.
        python_executable: Interpreter used to launch the server subprocess
            (defaults to the current interpreter / ``VERA_PYTHON``).
    """

    embodiment: Embodiment = "pusht"
    host: str = "127.0.0.1"
    server_port: int | None = None
    vis_port: int | None = None
    render_width: int | None = None  # per-view pixel width sent to the server (per-embodiment default)
    algo_config: Path | None = None
    dynamics_run_id: str | None = None
    text_prompt: str | None = None
    ckpt_root: Path | None = None
    wan_ckpt_root: Path | None = None  # frozen Wan2.1-T2V-1.3B base (mimicgen/omni); env VERA_WAN_CKPT_ROOT
    sample_steps: int | None = None
    tracker_backend: str | None = None
    motion_plan_scale: float | None = None
    teacache: bool = True
    teacache_thresh: float = 0.10
    auto_launch_server: bool = True
    server_ready_timeout: float | None = None
    python_executable: str | None = None
    # --- server launch mode -------------------------------------------------
    server_mode: str = "subprocess"  # "subprocess" | "docker"
    docker_image: str = "strands-vera-server:latest"
    docker_container_name: str | None = None  # default: vera-server-<embodiment>
    docker_gpus: str = "all"  # --gpus value (e.g. "all" or "device=0")
    docker_extra_args: list[str] | None = None  # extra `docker run` args (list, no shell)

    def __post_init__(self) -> None:
        # Checked first, because every per-embodiment default below is looked up
        # BY this field. Both lookups used to carry their own fallback -
        # ``_DEFAULT_PORTS.get(self.embodiment, (8800, 8801))`` and
        # ``_DEFAULT_RENDER_WIDTH.get(self.embodiment, 128)`` - and those two
        # literals are byte-for-byte mimicgen's entries, so every unrecognised
        # spelling resolved to mimicgen's ports and mimicgen's width. That is
        # not a degraded configuration, it is an indistinguishable one:
        # ``VeraServerRunner.start`` reuses a server that is already listening
        # ("ours or someone else's"), so ``embodiment="PushT"`` dialed 8800,
        # found a running mimicgen server and ran the whole rollout against the
        # wrong embodiment's planner/IDM pair under a success. A typo could not
        # be told from a deliberate ``embodiment="mimicgen"``.
        #
        # With the vocabulary held here the tables are the single statement of
        # what each embodiment defaults to, so they are indexed directly: a
        # second copy of a default is what made "not a known embodiment" and
        # "mimicgen" the same request.
        if (err := _embodiment_error(self.embodiment, "embodiment", type(self).__name__)) is not None:
            raise ValueError(err)
        # The other half of ``server_uri``. Checked here, in the same funnel the
        # port half passes through, because the two are one expression: a URI
        # cut apart by an unchecked host is not a bad address, it is a
        # *different* address, and the port is the component it takes. Nothing
        # downstream refuses it - the runner's probe reports a bind-only host as
        # ready, the client raises ``InvalidURI`` past the ``OSError`` channel
        # that carries its actionable hint, and a non-string surfaces as a
        # ``getaddrinfo`` ``TypeError`` out of ``start()``.
        if (err := dial_host_error(self.host, "host", type(self).__name__)) is not None:
            raise ValueError(err)
        # Apply per-embodiment port defaults when not explicitly set.
        default_policy, default_vis = _DEFAULT_PORTS[self.embodiment]
        # Both env overrides are read with ``is not None``, never with ``or``.
        # The two spellings are not interchangeable for a port: ``0`` is falsy,
        # so the ``or`` this line used to carry discarded the override and
        # applied the per-embodiment default in its place - the same discard
        # that ``render_width`` below was converted away from. That made one
        # value mean two things depending on how it was spelled:
        # ``VeraConfig(server_port=0)`` was refused by the shared domain (the
        # client has no way to learn which ephemeral port the kernel handed the
        # server, so it cannot dial one), while ``VERA_SERVER_PORT=0`` reported
        # success on the default. Reading the override for its presence rather
        # than its truth sends both spellings to the one check below, so the
        # caller who asked for 0 gets the refusal rather than port 8820.
        if self.server_port is None:
            env_port = _env_int("VERA_SERVER_PORT")
            self.server_port = env_port if env_port is not None else default_policy
        if self.vis_port is None:
            env_vis = _env_int("VERA_VIS_PORT")
            self.vis_port = env_vis if env_vis is not None else default_vis

        # Both ports are validated here, on the effective value, because this is
        # the one funnel every caller passes through - the ``VeraPolicy``
        # keywords, a pre-built config handed to it, and the ``VERA_*_PORT``
        # environment overrides above - and because a port reaches three
        # consumers under three different coercions: the provider dials
        # ``int(server_port or 0)``, :attr:`server_uri` interpolates the field
        # verbatim, and the runner's argv carries ``str(server_port)``. An
        # unusable value is therefore not merely refused late; it is applied as
        # three different ports (``2.7`` dials ``:2``, reports
        # ``ws://host:2.7`` and launches ``--port 2.7``, so the client cannot
        # reach the server it just started). Refusing before any client or
        # runner is built leaves nothing half-configured behind.
        if (err := tcp_port_error(self.server_port, "server_port", type(self).__name__)) is not None:
            raise ValueError(err)
        if (err := _viewer_port_error(self.vis_port, "vis_port", type(self).__name__)) is not None:
            raise ValueError(err)

        # ``render_width`` is a pixel count, so it takes the shared media domain
        # rather than a local rule: it is the same quantity as the recorders'
        # ``width``/``height`` and ``HybridCompositor.default_width``, and
        # ``positive_whole_number_error`` names pixels as one of the two families
        # it exists for. Applied here, on the effective value, for the reason the
        # ports above are - ``render_width`` is read only inside
        # ``_extract_frame``, which runs per frame *after* ``_ensure_started``
        # has launched the WAN server subprocess and completed the handshake, so
        # an unusable width surfaced there costs a model load (up to
        # ``server_ready_timeout``) before reporting a value that was wrong at
        # construction.
        #
        # The env override is read with ``is not None``, matching ``vis_port``
        # above rather than the ``or`` this line used to carry. The two spellings
        # are not interchangeable here: ``VERA_RENDER_WIDTH=0`` is falsy, so the
        # override was discarded and the per-embodiment default silently applied
        # in its place - a width of 0 cannot be honored, and the caller who asked
        # for it is owed the refusal below, not 128 under a success.
        if self.render_width is None:
            env_width = _env_int("VERA_RENDER_WIDTH")
            self.render_width = env_width if env_width is not None else _DEFAULT_RENDER_WIDTH[self.embodiment]
        if (err := positive_whole_number_error(self.render_width, "render_width", type(self).__name__)) is not None:
            raise ValueError(err)
        # Normalized to a plain ``int`` here because the domain accepts any real
        # scalar with an integral value - a ``128.0`` or a ``np.int64`` passes -
        # and the consumer requires a true ``int``: ``_resize_frame`` compares it
        # against ``frame.shape`` and hands it to ``Image.resize``, and it is
        # sent to the server as ``view_widths``. This is the normalization the
        # shared domain documents as the caller's obligation, and it is why the
        # ``int()`` at each read site is no longer needed.
        self.render_width = int(self.render_width)

        # Environment overrides (deploy/CI win over code defaults).
        if self.algo_config is None:
            ac = _env("VERA_ALGO_CONFIG")
            self.algo_config = Path(ac) if ac else None
        if self.dynamics_run_id is None:
            self.dynamics_run_id = _env("VERA_DYNAMICS_RUN_ID")
        if self.text_prompt is None:
            self.text_prompt = _env("VERA_TEXT_PROMPT")
        if self.ckpt_root is None:
            cr = _env(
                "VERA_CKPT_ROOT",
                f"VERA_{self.embodiment.upper()}_CKPT_ROOT",
                "VERA_MIMICGEN_CKPT_DIR" if self.embodiment == "mimicgen" else "",
            )
            self.ckpt_root = Path(cr) if cr else None
        if self.wan_ckpt_root is None:
            wr = _env("VERA_WAN_CKPT_ROOT")
            self.wan_ckpt_root = Path(wr) if wr else None
        if self.sample_steps is None:
            self.sample_steps = _env_int("VERA_SAMPLE_STEPS")
        if self.tracker_backend is None:
            self.tracker_backend = _env("VERA_TRACKER_BACKEND")
        if self.motion_plan_scale is None:
            self.motion_plan_scale = _env_float("VERA_MOTION_PLAN_SCALE")
        # Checked here, on the effective value, for the reason the two ports and
        # ``render_width`` above are: this is the one funnel every caller passes
        # through, and the only place the ``VERA_MOTION_PLAN_SCALE`` override just
        # applied can still be refused. ``_env_float`` returns whatever ``float()``
        # accepts, so ``nan``, ``inf``, ``1e999`` and a negative all reach the field
        # from the environment; the field was checked nowhere, so a ``str`` or a
        # ``list`` reached it from a keyword too.
        #
        # It is the third scale in this package and takes the domain the other two
        # already take (``translation_scale``, ``ik_smoothing``): a multiplier on a
        # motion plan has no usable non-positive or non-finite value. ``None`` stays
        # valid because it is the documented opt-out - it gates the ``configure``
        # call away entirely, so "leave the server's scale alone" and "scale the
        # plan to nothing" stay different requests.
        #
        # Leaving it to be refused downstream is not an option, because nothing
        # downstream refuses it: ``_ensure_started`` sends the value with
        # ``self._client.configure(...)`` inside a best-effort ``except Exception``
        # that logs at INFO and marks the policy started regardless. A value
        # ``float()`` cannot convert is therefore neither applied nor reported -
        # the rollout proceeds at whatever scale the server already had while the
        # config says otherwise.
        if self.motion_plan_scale is not None:
            if (
                err := positive_finite_number_error(self.motion_plan_scale, "motion_plan_scale", type(self).__name__)
            ) is not None:
                raise ValueError(err)
            # Normalized for the reason ``render_width`` is: the shared domain admits
            # any real scalar, so an ``int`` 1 or a ``np.float64`` passes it, while
            # the field is declared ``float``.
            self.motion_plan_scale = float(self.motion_plan_scale)
        # ``sample_steps`` and ``teacache_thresh`` are the video planner's two
        # sampler knobs, and they were the two numeric fields this funnel did not
        # look at. Five of the seven are held to a shared domain on the effective
        # value - both ports, ``render_width``, ``motion_plan_scale`` and the
        # readiness budget below - and these two were held to nothing, so every
        # spelling of them was accepted: ``nan``, ``inf``, a negative, a zero, a
        # ``bool``, a ``str``, and ``None`` on a field declared ``float``.
        #
        # Neither field is read anywhere else. Their only consumer is the launch
        # command, which carries them as TEXT: ``str(cfg.sample_steps)`` and
        # ``str(cfg.teacache_thresh)`` in ``VeraServerRunner._build_command``, and
        # ``f"VERA_SAMPLE_STEPS={cfg.sample_steps}"`` in the docker ``-e`` overlay.
        # Nothing between here and the server inspects the value, so the server is
        # left to report it, and it has two ways to - neither naming the field:
        #
        # * A token the flag's own type cannot parse (``'2.7'``, ``'nan'``,
        #   ``'True'``, ``'ten'`` for an ``int`` flag) makes the server exit before
        #   it opens its port, and ``_wait_until_ready`` reports "VERA server
        #   exited early (code N) ... common causes are missing checkpoints (set
        #   VERA_CKPT_ROOT / ckpt_root) or CUDA OOM" - two causes that are not the
        #   cause.
        # * A token it can parse (``'0'`` or ``'-5'`` denoise steps, a ``nan`` or
        #   ``inf`` threshold) starts a server configured by a value nobody asked
        #   for, and the rollout runs on it under a reported success.
        #
        # Which of the two happens is not a property of the value being usable, it
        # is a property of how ``str()`` happens to spell it. ``start()`` already
        # holds this position two statements above the launch:
        # ``_require_vera_installed`` exists because, in its own words, without it
        # "a missing install surfaces only as an opaque 'server exited early (code
        # 1)' RuntimeError several seconds later". A value this constructor can
        # judge belongs in the same place.
        #
        # The two spellings of each knob disagreed, too. ``_env_int`` and
        # ``_env_float`` return ``None`` for anything ``int()``/``float()`` refuses,
        # so ``VERA_SAMPLE_STEPS=ten`` is absorbed and the planner yaml decides -
        # deliberate, and pinned. The keyword spelling of the same knob was checked
        # nowhere, so one knob was guarded from the environment and unguarded from
        # the API.
        #
        # ``sample_steps`` is a count of denoise steps, so it takes the same shared
        # domain ``render_width`` does, and is normalized for the same reason - and
        # here the conversion is load-bearing rather than tidy. That domain admits
        # an integral float, and a computed count is one: ``sample_steps=20 / 2`` is
        # ``10.0``, ``str(10.0)`` is ``'10.0'``, and ``--sample-steps`` cannot parse
        # it. Converting after the domain accepts the value is what puts ``10`` on
        # the command line rather than a token that ends the server.
        #
        # ``teacache_thresh`` is a continuous rel_l1 threshold, so it takes the
        # continuous domain ``motion_plan_scale`` does. It is checked
        # unconditionally even though the command carries it only when ``teacache``
        # is on, because this is a plain dataclass: ``teacache`` can be turned on
        # after construction, and a check scoped to its value here would not be
        # there when the field is read. ``0`` is not the opt-out either -
        # ``teacache=False`` is, and it emits ``--no-teacache`` in place of the
        # threshold. The documented quality cliff above ``0.15`` is guidance and
        # stays a legitimate request; only the values no threshold can be are
        # refused.
        if self.sample_steps is not None:
            if (err := positive_whole_number_error(self.sample_steps, "sample_steps", type(self).__name__)) is not None:
                raise ValueError(err)
            self.sample_steps = int(self.sample_steps)
        if (
            err := positive_finite_number_error(self.teacache_thresh, "teacache_thresh", type(self).__name__)
        ) is not None:
            raise ValueError(err)
        self.teacache_thresh = float(self.teacache_thresh)
        # The readiness budget is resolved and checked here for the reason the
        # ports, ``render_width`` and ``motion_plan_scale`` above are: this is the
        # one funnel every caller passes through, and it is the only place the
        # value can still be refused. Both ``VeraServerRunner._wait_until_ready``
        # implementations - the subprocess one and the docker one - consume it as
        # ``time.monotonic() + cfg.server_ready_timeout``, so the field is read
        # only once the server has already been launched.
        #
        # ``VERA_SERVER_READY_TIMEOUT`` is read here because BOTH of those
        # timeouts name it in the message they raise ("raise server_ready_timeout
        # / VERA_SERVER_READY_TIMEOUT if needed"), and nothing read it: those two
        # strings were the environment variable's only appearances in the tree.
        # An operator who took the advice exported it, re-ran, and timed out
        # after exactly the same 600 seconds with exactly the same message
        # pointing at exactly the same variable - the one failure mode a
        # readiness timeout exists to make actionable. It is applied for its
        # presence and not its truth, matching the ports above, so a spelling the
        # environment carries reaches the same check a keyword does.
        #
        # A span of seconds has no usable non-positive or non-finite value, and
        # each way this field could hold one was reachable and silent:
        #
        # * ``inf`` made ``deadline`` infinite, so ``while time.monotonic() <
        #   deadline`` never ended. The wait that documents "or raise on timeout"
        #   instead polled forever, and because the raise is what calls
        #   ``self.stop()``, the server subprocess (or container) it had just
        #   launched was never torn down either.
        # * ``nan`` is below nothing, so ``time.monotonic() < nan`` was False on
        #   the first test: the loop body never ran, the port was never probed
        #   once, and a server that was coming up fine was torn down and reported
        #   as "did not become ready within nans".
        # * ``0`` and a negative did the same thing in plainer words ("within 0s",
        #   "within -30s"), and ``True`` - an ``int`` subclass - silently meant a
        #   one-second budget.
        # * a ``str`` (``"600"``, the shape an environment value has) raised
        #   ``TypeError: unsupported operand type(s) for +: 'float' and 'str'``
        #   out of the wait, past the ``TimeoutError``/``RuntimeError`` channel
        #   the runner documents, with the server already spawned and again no
        #   ``stop()``.
        #
        # ``0`` is not an opt-out here, for the reason it is not one for
        # ``motion_plan_scale``: ``_ensure_started`` already probes the port
        # BEFORE launching anything and reuses a server that is listening, so
        # "do not wait" is expressible without a zero budget, and a zero budget
        # only ever launches a server and instantly tears it down.
        if self.server_ready_timeout is None:
            env_timeout = _env_float("VERA_SERVER_READY_TIMEOUT")
            self.server_ready_timeout = env_timeout if env_timeout is not None else _DEFAULT_SERVER_READY_TIMEOUT
        if (
            err := positive_finite_number_error(self.server_ready_timeout, "server_ready_timeout", type(self).__name__)
        ) is not None:
            raise ValueError(err)
        # Normalized for the reason ``render_width`` and ``motion_plan_scale``
        # are: the shared domain admits any real scalar, while the consumers add
        # it to a ``float`` clock and format it with ``:.0f``.
        self.server_ready_timeout = float(self.server_ready_timeout)
        if self.python_executable is None:
            self.python_executable = _env("VERA_PYTHON")
        _sm = _env("VERA_SERVER_MODE")
        if _sm:
            self.server_mode = _sm
        _di = _env("VERA_DOCKER_IMAGE")
        if _di:
            self.docker_image = _di
        _dg = _env("VERA_DOCKER_GPUS")
        if _dg:
            self.docker_gpus = _dg
        if self.docker_container_name is None:
            self.docker_container_name = _env("VERA_DOCKER_CONTAINER") or f"vera-server-{self.embodiment}"

        # Coerce string paths to Path (defensive - callers may pass str).
        if self.algo_config is not None and not isinstance(self.algo_config, Path):
            self.algo_config = Path(self.algo_config)
        if self.ckpt_root is not None and not isinstance(self.ckpt_root, Path):
            self.ckpt_root = Path(self.ckpt_root)
        if self.wan_ckpt_root is not None and not isinstance(self.wan_ckpt_root, Path):
            self.wan_ckpt_root = Path(self.wan_ckpt_root)

    @property
    def server_uri(self) -> str:
        """Websocket URI the client connects to (``ws://<host>:<server_port>``)."""
        return f"ws://{self.host}:{self.server_port}"

    def server_env(self) -> dict[str, str]:
        """Environment overlay for the server subprocess (checkpoints, tracker)."""
        env: dict[str, str] = {}
        if self.ckpt_root is not None:
            env["VERA_CKPT_ROOT"] = str(self.ckpt_root)
        if self.wan_ckpt_root is not None:
            env["VERA_WAN_CKPT_ROOT"] = str(self.wan_ckpt_root)
        if self.tracker_backend is not None:
            env["VERA_TRACKER_BACKEND"] = self.tracker_backend
        if self.dynamics_run_id is not None:
            env["VERA_DYNAMICS_RUN_ID"] = str(self.dynamics_run_id)
        return env
