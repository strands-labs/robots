"""Shared utilities for strands-robots."""

import importlib
import logging
import math
import numbers
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache of lazy-loaded modules
_lazy_modules: dict[str, object] = {}


def require_optional(
    module_name: str,
    *,
    pip_install: str | None = None,
    extra: str | None = None,
    purpose: str = "",
) -> object:
    """Import an optional dependency, raising a clear error if missing.

    Once imported, the module is cached so subsequent calls are free.

    Args:
        module_name: Dotted module name to import (e.g. ``"zmq"``).
        pip_install: Explicit pip package name if it differs from *module_name*.
        extra: ``pyproject.toml`` extras group (e.g. ``"groot-service"``).
        purpose: Human-readable description shown in the error message.

    Returns:
        The imported module object.

    Raises:
        ImportError: With a helpful install instruction.
    """
    if module_name in _lazy_modules:
        return _lazy_modules[module_name]

    try:
        module = importlib.import_module(module_name)
        _lazy_modules[module_name] = module
        return module
    except ImportError:
        install_hint = pip_install or module_name
        parts = [f"'{module_name}' is required"]
        if purpose:
            parts[0] += f" for {purpose}"
        parts.append("Install with:")
        if extra:
            parts.append(f"  pip install 'strands-robots[{extra}]'")
        parts.append(f"  pip install {install_hint}")
        raise ImportError("\n".join(parts)) from None


def require_optionals(
    module_names: list[str] | tuple[str, ...],
    *,
    extra: str | None = None,
    purpose: str = "",
) -> None:
    """Require several optional dependencies, reporting ALL missing ones at once.

    Unlike calling :func:`require_optional` in a loop -- which raises on the
    FIRST missing module and hides the rest -- this probes every name and, if
    any are absent, raises a single ``ImportError`` naming every missing module.
    That lets a caller in a partially-provisioned environment fix all of them in
    one install instead of discovering them one reinstall at a time (each retry
    of a heavy load path is expensive).

    Present modules are imported and cached (same as :func:`require_optional`),
    so a follow-up ``require_optional`` for any of them is free.

    Args:
        module_names: Dotted module names to require (e.g. ``("transformers",
            "peft", "scipy")``).
        extra: ``pyproject.toml`` extras group naming where the deps ship
            (e.g. ``"molmoact2"``); shown in the install hint.
        purpose: Human-readable description shown in the error message.

    Raises:
        ImportError: If one or more modules are missing, listing every missing
            module and an actionable install instruction.
    """
    missing: list[str] = []
    for name in module_names:
        if name in _lazy_modules:
            continue
        try:
            _lazy_modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)

    if not missing:
        return

    joined = ", ".join(f"'{m}'" for m in missing)
    label = "is required" if len(missing) == 1 else "are required"
    parts = [f"{joined} {label}"]
    if purpose:
        parts[0] += f" for {purpose}"
    parts.append("Install with:")
    if extra:
        parts.append(f"  pip install 'strands-robots[{extra}]'")
    parts.append(f"  pip install {' '.join(missing)}")
    raise ImportError("\n".join(parts)) from None


def lerobot_version() -> str:
    """Return the installed lerobot version, or ``"unknown"`` if undeterminable.

    Best-effort and never raises: it exists for error messages, where naming the
    installed version is what distinguishes "lerobot is missing" from "lerobot
    is present but something it needs is not". Lives here rather than beside one
    of its callers because those callers sit in different modules
    (:mod:`strands_robots.dataset_recorder` and
    :mod:`strands_robots.streaming_dataset`) and must not report the version
    differently.

    ``except ImportError`` is deliberately the whole handler. ``version`` signals
    an unresolvable distribution with ``PackageNotFoundError``, which subclasses
    ``ModuleNotFoundError`` and so already *is* an ``ImportError``. Naming it in
    the handler as well would bind it as a local that the ``import`` above may
    never reach, and evaluating the handler would then raise
    ``UnboundLocalError`` out of a function documented never to raise - on
    precisely the failure the second name looked like it was covering.
    """
    try:
        from importlib.metadata import version

        return version("lerobot")
    except ImportError:
        return "unknown"


#
# Path resolution - single source of truth for all strands-robots paths
#

#: Default base directory for all user data.
DEFAULT_BASE_DIR = Path.home() / ".strands_robots"


def get_base_dir() -> Path:
    """Get the base directory for strands-robots user data.

    Resolution (in priority order):

    1. ``STRANDS_BASE_DIR`` env var - explicit override. Use this when
       you want to relocate *all* strands-robots user data (assets,
       user registry, caches) to a non-default location.
    2. ``~/.strands_robots/`` - default.

    Note:
        ``STRANDS_ASSETS_DIR`` **only** controls the assets subdirectory
        (see :func:`get_assets_dir`). It does *not* move the base dir,
        so user-level metadata like ``user_robots.json`` always lands in
        a predictable location rather than wherever the assets happen
        to be pointed.

    Returns:
        Path to the base directory (created if needed).
    """
    custom = os.getenv("STRANDS_BASE_DIR")
    d = Path(custom) if custom else DEFAULT_BASE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_assets_dir() -> Path:
    """Get the assets directory (robot model files, meshes, URDFs).

    Resolution:
        1. ``STRANDS_ASSETS_DIR`` env var - used as-is
        2. ``~/.strands_robots/assets/`` - default

    Returns:
        Path to the assets directory (created if needed).
    """
    custom = os.getenv("STRANDS_ASSETS_DIR")
    if custom:
        d = Path(custom)
    else:
        d = DEFAULT_BASE_DIR / "assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_asset_path(relative_or_absolute: str | Path | None, default_name: str = "") -> Path:
    """Resolve an asset path against the assets directory.

    Args:
        relative_or_absolute: Path to resolve.
            - ``None`` → ``<assets_dir>/<default_name>/``
            - Absolute (or ``~/...``) → expanded as-is
            - Relative → ``<assets_dir>/<relative>/``
        default_name: Fallback subdirectory name when path is None.

    Returns:
        Resolved absolute Path.
    """
    assets = get_assets_dir()
    if relative_or_absolute is None:
        return assets / default_name
    expanded = Path(relative_or_absolute).expanduser()
    if expanded.is_absolute():
        return expanded
    return assets / expanded


#
# Path safety - prevent traversal via untrusted components
#


def safe_join(base: Path, untrusted: str, *, resolve_symlinks: bool = False) -> Path:
    """Join *base* with an untrusted relative path, rejecting traversal.

    Used to protect against ``../`` escapes in registry-sourced or
    user-supplied path components before they reach the filesystem. Containment
    is always verified lexically; set *resolve_symlinks* to additionally reject
    symlinked components that escape *base* after resolution.

    Args:
        base: Trusted base directory.
        untrusted: Relative path component (may contain ``/`` but must not
            escape *base*).
        resolve_symlinks: When ``True``, containment is re-verified after full
            symlink resolution so a symlinked component that points outside
            *base* (e.g. ``base/link -> /etc`` followed by ``link/passwd``) is
            rejected. Enable this when *base* is an untrusted or externally
            sourced tree - e.g. a freshly cloned repository - whose symlinks may
            escape. Leave ``False`` (the default) for the managed asset cache,
            whose robot directories are intentionally symlinked to installed
            ``robot_descriptions`` packages that legitimately live outside the
            cache; resolving those would wrongly reject them.

    Returns:
        Normalised absolute Path under *base*.

    Raises:
        ValueError: If the resulting path would escape *base* (lexically, or via
            a symlink when *resolve_symlinks* is set).

    Example::

        safe_join(Path("/assets"), "robot/model.xml")   # OK
        safe_join(Path("/assets"), "../etc/passwd")     # ValueError
    """
    joined = Path(os.path.normpath(base / untrusted))
    base_norm = Path(os.path.normpath(base))
    if not (joined == base_norm or str(joined).startswith(str(base_norm) + os.sep)):
        raise ValueError(f"Path traversal blocked: {untrusted!r} escapes {base}")
    if resolve_symlinks:
        # Lexical normalisation cannot see through symlinks: a component such as
        # ``link/passwd`` where ``base/link`` targets ``/etc`` stays lexically
        # under *base* yet resolves outside it. ``resolve(strict=False)``
        # resolves the existing prefix and appends the remainder lexically for
        # not-yet-created files; resolving *base* too keeps a symlinked base
        # prefix (e.g. /tmp on macOS) consistent on both sides.
        base_resolved = base_norm.resolve()
        joined_resolved = joined.resolve()
        if not (joined_resolved == base_resolved or str(joined_resolved).startswith(str(base_resolved) + os.sep)):
            raise ValueError(f"Path traversal blocked: {untrusted!r} escapes {base} via symlink")
    return joined


def get_search_paths() -> list[Path]:
    """Get ordered list of asset search paths.

    Used by both :mod:`strands_robots.assets.manager` and
    :mod:`strands_robots.assets.download` - centralised here to avoid
    a circular dependency between those two modules.

    Order (local assets take priority over defaults):
        1. User asset dir (``STRANDS_ASSETS_DIR`` or ``~/.strands_robots/assets/``)
        2. ``CWD/assets`` (project-local, deduplicated if it resolves to the same dir)
    """
    paths: list[Path] = []
    user_cache = get_assets_dir()
    paths.append(user_cache)
    cwd_assets = Path.cwd() / "assets"
    if cwd_assets not in paths:
        paths.append(cwd_assets)
    return paths


def process_rss_mb() -> float | None:
    """Current resident set size (RSS) of this process, in megabytes.

    Used to surface ``policy_resident_rss_mb`` telemetry so a caller can see
    whether a heavy model (e.g. a multi-GB VLA checkpoint) is actually resident
    after a load - and, across a multi-episode loop, that it stays resident
    rather than oscillating as it would if the policy were rebuilt per episode.

    Prefers :mod:`psutil` (true *current* RSS, the meaningful "is it resident
    now" number). Falls back to :func:`resource.getrusage`, which reports peak
    RSS for the process (``ru_maxrss``) - an over-estimate of current usage, but
    still a useful floor when psutil is absent. ``ru_maxrss`` is in kilobytes on
    Linux and bytes on macOS; both are normalised to MB.

    Returns:
        Resident memory in MB as a float, or ``None`` when neither source is
        available (e.g. a platform without ``resource``), so callers can omit
        the field rather than report a misleading zero.
    """
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except (ImportError, OSError):
        # psutil missing or the /proc read failed; fall back to stdlib resource.
        pass
    try:
        import resource
        import sys

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss units differ by platform: bytes on macOS, kilobytes on Linux.
        divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        return float(maxrss) / divisor
    except (ImportError, ValueError, OSError):
        return None


def positive_finite_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable positive finite number.

    Shared domain for every CONTINUOUS knob that names a rate or a span of
    time - a control-loop frequency in Hz, a rollout or teleop ``duration`` in
    seconds. Unlike :func:`positive_whole_number_error` a fractional value is
    perfectly usable here (``2.5`` seconds, ``62.5`` Hz), so only the sign and
    the finiteness are constrained. It lives here rather than beside one of its
    callers because those callers sit in different layers
    (:mod:`strands_robots.teleop_mixin` must not depend on
    :mod:`strands_robots.simulation`), and the accepted domain must not diverge
    between them.

    Only a positive finite value can be honored. Such a knob is always a
    divisor (the loop period is ``1 / hz``) or a horizon (``duration *
    frequency`` steps), so ``0`` makes the period undefined or the horizon
    empty, a negative value inverts it, ``nan`` poisons every comparison it
    reaches (``nan > 0`` and ``nan <= 0`` are both ``False``), and ``inf``
    collapses the period to ``0`` - an unthrottled loop, not a fast one.
    Accepts any real scalar (so a NumPy ``np.float32`` rate read from a config
    array passes) and rejects ``bool`` explicitly - an ``int`` subclass whose
    ``True`` would act as a silent ``1``.

    Args:
        value: The caller-supplied value.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            normally the public method name.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        # ``isfinite`` before the sign test: ``nan`` is never ``<= 0``, so
        # ordering these the other way lets it through.
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        return f"{context}: {param} must be > 0, got {value!r}."
    return None


def finite_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable finite number of either sign.

    Shared domain for a SIGNED physical quantity a command carries verbatim to a
    robot - a linear or angular velocity component, an offset, a coordinate read
    as a scalar. Both signs are legitimate (reverse is a negative linear
    velocity, clockwise a negative yaw rate), so :func:`positive_finite_number_error`
    cannot express this domain; only finiteness and numeric-ness are constrained.
    It lives here rather than beside one of its callers because those callers sit
    in different layers (:mod:`strands_robots.mesh` must not depend on
    :mod:`strands_robots.simulation`), and the accepted domain must not diverge
    between them.

    Only a finite value can be honored. ``nan``/``inf`` serialize into a wire
    message as a valid IEEE-754 float64, so the transport accepts them and the
    receiving controller integrates them into its state estimate - a silently
    poisoned pose rather than a rejected command. A non-real value (a numeric
    string, ``None``, a list) otherwise raises a bare ``TypeError`` from the
    ``float()`` coercion, escaping the structured ``{"status": "error"}``
    tool-result contract. Accepts any real scalar (so a NumPy ``np.float32``
    velocity read from a policy action passes) and rejects ``bool`` explicitly -
    an ``int`` subclass whose ``True`` would act as a silent ``1``.

    Args:
        value: The caller-supplied value.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            normally the public method name.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        return f"{context}: {param} must be a finite number, got {value!r}."
    return None


def positive_whole_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable positive whole number.

    Shared domain for every media knob that counts frames or pixels - the
    recorders' ``fps``, ``width``, ``height`` and in-memory frame cap, the
    ``run_policy(video=...)`` dict fields, and the
    :func:`~strands_robots.rendering.encode_clip` playback rate. It lives here
    rather than beside one of its callers because those callers sit in different
    layers (:mod:`strands_robots.rendering` must not depend on
    :mod:`strands_robots.simulation`), and the accepted domain must not diverge
    between them. Only a positive whole number can be honored: ``0`` makes the capture loop's ``1 / fps``
    period undefined, a negative rate is rejected by the ffmpeg writer, and a
    zero/negative frame cap drops every frame. Accepts any real scalar with an
    integral value (so a NumPy ``np.int64`` height or a ``30.0`` computed from a
    config float passes) and rejects ``bool`` explicitly - an ``int`` subclass
    whose ``True`` would act as a silent 1.

    Args:
        value: The caller-supplied value.
        param: The parameter (or dict key) it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            ``"video"`` for the :class:`VideoConfig` dict, the method name for a
            keyword parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    message = f"{context}: {param} must be a positive whole number, got {value!r}."
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return message
    numeric = float(value)
    # ``isfinite`` first: ``int(nan)`` raises, and short-circuiting keeps it
    # out of the integrality check below.
    if not math.isfinite(numeric) or numeric != int(numeric) or numeric < 1:
        return message
    return None


def positive_count_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable positive integer count.

    Shared domain for two families of discrete quantity:

    * The knobs that count iterations of a control or rollout loop - the
      simulation's ``n_episodes`` / ``max_steps`` / ``control_substeps`` /
      ``action_horizon`` and the hardware control loop's ``action_horizon``.
    * A camera's pixel dimensions - the ``width`` / ``height`` of
      ``add_camera`` and of the render family (``render``, ``get_frame``,
      ``get_camera_params``) on every simulation backend. A backend may add
      an upper bound of its own on top of this floor (MuJoCo caps at the
      offscreen framebuffer size; the ray-traced backends have no such
      buffer), but the floor itself must not differ between them: the same
      camera configuration cannot be refused on one backend and accepted on
      another.

    It lives here rather than beside one of its callers because those callers
    sit in different layers (:mod:`strands_robots.hardware_robot` must not
    depend on :mod:`strands_robots.simulation`), and the accepted domain must
    not diverge between them: the same count cannot be refused for a digital
    twin and accepted for the arm it mirrors.

    Distinct from :func:`positive_whole_number_error`, which accepts any real
    scalar with an integral value so a ``30.0`` read from a config or an
    ``np.int64`` probed from a camera can be honored. The values guarded here
    are consumed directly as ``range()`` bounds, slice indices, or an array /
    framebuffer dimension, where an integral float raises ``TypeError``
    ("``'float' object cannot be interpreted as an integer``") rather than being
    coerced, so only a true ``int`` can be honored.

    ``bool`` is rejected explicitly. It is an ``int`` subclass, so a bare
    ``value < 1`` test lets ``True`` through as a silent count of 1 while
    rejecting ``False`` - a value the caller never meant either way.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message. Callers that
            accept a ``{robot_name: count}`` mapping pass a subscripted label
            (``"action_horizon['alice']"``) so the message names the entry the
            caller got wrong rather than the whole mapping.
        context: Message prefix identifying the surface that received it - the
            public method name, or the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return f"{context}: {param} must be a positive integer, got {value!r}."
    return None


def tcp_port_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` cannot address a TCP port.

    Shared domain for every caller-supplied port number: the agent tools that
    reach a service over TCP (``use_rosbridge``'s WebSocket,
    ``gr00t_inference``'s inference service) and the mesh bridges that construct
    one. A port is an index into the 16-bit TCP port space, so only an ``int``
    in ``[1, 65535]`` names one: ``0`` asks the kernel for an ephemeral port
    rather than naming a port, and a value outside the range has nothing to bind
    or connect to.

    It lives here rather than beside one of its callers for the same reason
    :func:`positive_count_error` does: those callers sit in different layers
    (:mod:`strands_robots.tools` and :mod:`strands_robots.mesh` must not depend
    on each other) and the accepted domain must not diverge between them - the
    same port cannot be refused by one transport onto a service and accepted by
    the next.

    ``bool`` is rejected explicitly. It is an ``int`` subclass, so a bare
    ``1 <= value <= 65535`` test lets ``True`` through as a silent port 1 - a
    privileged port the caller never named.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            requested action for an agent tool, or the class name for a
            constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        return f"{context}: invalid {param}: {value!r} (expected 1-65535)"
    return None


def non_negative_count_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable non-negative integer count.

    Shared domain for a discrete count whose ``0`` is a first-class value rather
    than a degenerate one - the number of control steps a loop executes while an
    inference request is in flight
    (:attr:`~strands_robots.policies.base.Policy.rtc_observed_delay_steps`).
    That count is exactly ``0`` in the dominant case: a synchronous eval loop
    pauses the world during inference, so no step elapses. Refusing ``0`` here
    would therefore reject the common configuration, which is why this is a
    separate domain rather than a caller of :func:`positive_count_error`.

    In every other respect it mirrors :func:`positive_count_error`: the value is
    consumed as an offset into an action chunk, so only a true ``int`` can be
    honored (an integral float raises ``TypeError`` at the slice rather than
    being coerced), and ``bool`` is rejected explicitly because as an ``int``
    subclass a bare ``value < 0`` test lets ``True`` through as a silent count
    of one.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            public method name, or the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return f"{context}: {param} must be a non-negative integer, got {value!r}."
    return None


def name_list_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable list of distinct key names.

    Shared domain for the parameters that carry an ordered list of KEY NAMES
    into a policy: the LeRobot ``image_keys`` (model VISUAL feature keys to
    declare on the config) and the VERA ``image_keys`` (observation camera keys
    to width-concat into one frame). The two name different vocabularies, but
    the shape contract is identical - several distinct non-blank names, in the
    order the caller wants them - and both consumers reach the same failure when
    it is not met, so the rule lives here rather than beside either of them.

    The mistake this exists for is a single name passed as a bare string.
    ``str`` is iterable, so ``list("wrist")`` yields ``['w', 'r', 'i', 's', 't']``
    - five names the caller never wrote, one per character. Nothing downstream
    can tell that apart from a deliberate five-entry list, so it is accepted and
    the consequence surfaces far from the call: a model built declaring
    per-character features, or a ``KeyError: 'w'`` raised mid-rollout when the
    frame for a one-letter camera is looked up.

    A ``Mapping`` is refused for the same reason in the other direction: it is
    iterable over its keys, so its values would be silently discarded.

    A repeated name is refused because it cannot be honored as written - the
    LeRobot side builds a feature dict, where a duplicate collapses and declares
    fewer features than asked for, and the VERA side concatenates one panel per
    entry, where a duplicate doubles the width of the frame the model sees.

    Only a :class:`~collections.abc.Sequence` is accepted, which excludes
    one-shot iterators. That matters on the LeRobot path, where the value is
    read twice - once by the pre-flight check and again at load - so a generator
    exhausted by the first read would present as empty to the second.

    Callers gate this check on a truthy value, because in both consumers a falsy
    ``image_keys`` (``None``, or an empty list) already means "not supplied" and
    the list is derived instead. So an empty sequence is not rejected here, and
    ``None`` is the caller's to skip rather than this function's to accept - a
    surface where an absent value IS an error keeps that verdict its own.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            public method name, or the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, str | bytes):
        shown = value.decode(errors="replace") if isinstance(value, bytes) else value
        return (
            f"{context}: {param} must be a list of names, not a single string, got {value!r}. "
            f"A string is iterable per character, so this would be read as "
            f"{[c for c in shown][:6]}{' ...' if len(shown) > 6 else ''} "
            f"({len(shown)} name(s)). Wrap it in a list: [{shown!r}]."
        )
    if isinstance(value, Mapping):
        return (
            f"{context}: {param} must be a list of names, not a mapping, got {value!r}. "
            f"A mapping is iterable over its keys, so its values would be discarded - "
            f"pass the names as a list: {list(value)!r}."
        )
    if not isinstance(value, Sequence):
        return (
            f"{context}: {param} must be a list of names, got {type(value).__name__} "
            f"({value!r}). Pass a list or tuple; a one-shot iterator cannot be used "
            f"because the value is read more than once."
        )
    for i, entry in enumerate(value):
        if not isinstance(entry, str):
            return f"{context}: {param}[{i}] must be a name (str), got {type(entry).__name__} ({entry!r})."
        if not entry.strip():
            return f"{context}: {param}[{i}] must be a non-blank name, got {entry!r}."
    seen: set[str] = set()
    repeated: set[str] = set()
    for entry in value:
        if entry in seen:
            repeated.add(entry)
        seen.add(entry)
    if repeated:
        return (
            f"{context}: {param} must not repeat a name, got {list(value)!r} "
            f"({sorted(repeated)!r} appears more than once)."
        )
    return None


def finite_vector_error(method: str, param_name: str, vec: Any) -> str | None:
    """Return an error message if any element of ``vec`` is not a finite number.

    Guards the numeric vectors a scene-construction call bakes into the
    compiled MJCF (``add_object`` color/size, ``add_camera`` position/target,
    etc.) against the two classes the numeric-input campaign targets:

    * A non-numeric or non-iterable element (e.g. ``["a", "b", "c"]`` or a
      nested list) otherwise raises a bare ``TypeError``/``ValueError`` deep
      inside MuJoCo's ``add_geom`` or a ``size <= 0`` comparison - escaping the
      structured ``{"status": "error"}`` tool-result contract.
    * A ``nan``/``inf`` component is baked verbatim into the geom/camera and
      either poisons the physics state on the next ``mj_forward`` or aborts the
      spec recompile with a cryptic "spec recompile refused", reporting a
      success/garbage result instead of an actionable error.

    A numpy real scalar per element is accepted (``np.float64`` and friends are
    registered as ``numbers.Real``), matching the "accept NumPy scalar
    components" behaviour of the other sim setters; a ``bool`` is refused
    because ``float(True)`` would silently write ``1.0`` where a coordinate,
    extent or colour channel belongs. Length is NOT checked here (size is shape-dependent, so
    its count is checked against the shape afterwards); use
    :func:`pose_vector_error` for a fixed length, or the rgba coercion in
    :mod:`strands_robots.simulation.mujoco.physics` for a colour, whose count
    the geom's rgba row defines. Returns ``None`` when every element is a finite
    real number.
    """
    try:
        iter(vec)
    except TypeError:
        return f"{method}: '{param_name}' must be a list/tuple of numbers, got {vec!r}"
    for _elem in vec:
        # ``numbers.Real`` accepts a numpy scalar (``np.float32`` / ``np.int64``
        # are registered) and rejects a string, ``None`` or a nested list.
        # ``bool`` is an ``int`` subclass, so it would otherwise pass as a
        # silent ``1.0`` - a ``True`` coordinate placing a body 1 m out. The
        # agent-tool router already refuses a bool component, so refusing it
        # here keeps the direct API and the tool surface in step.
        if isinstance(_elem, bool) or not isinstance(_elem, numbers.Real):
            return f"{method}: '{param_name}' elements must be numbers, got {vec!r}"
        if not math.isfinite(float(_elem)):
            return f"{method}: '{param_name}' must contain finite numbers (no nan/inf), got {vec!r}"
    return None


def pose_vector_error(method: str, param_name: str, vec: Any, expected_len: int) -> str | None:
    """Return an error message if ``vec`` is not ``expected_len`` finite numbers.

    Fixed-length wrapper over :func:`finite_vector_error` for the pose
    vectors written straight into ``data.qpos`` (``move_object`` /
    ``add_object`` position+orientation, ``add_camera`` position+target). A
    wrong-length vector otherwise raises a bare ``ValueError`` inside the numpy
    assignment - escaping the structured ``{"status": "error"}`` tool-result
    contract - and a ``nan``/``inf`` component is propagated through the whole
    physics state by ``mj_forward``, reporting ``success`` while silently
    poisoning the simulation. A numpy real scalar per element is accepted.

    It lives here rather than beside one of its callers because those callers
    sit in different layers: the scene-construction facade builds a pose before
    the model is compiled, while the motion primitives take one against a live
    model, and their accepted domain must not diverge - a pose either backend
    entry point refuses must be refused by the other. Returns ``None`` when
    ``vec`` is acceptable.
    """
    try:
        length = len(vec)
    except TypeError:
        return f"{method}: '{param_name}' must be a list/tuple of {expected_len} numbers, got {vec!r}"
    if length != expected_len:
        return f"{method}: '{param_name}' must be a {expected_len}-element vector, got {length} ({vec!r})"
    return finite_vector_error(method, param_name, vec)


def coerce_pose_vector(
    method: str, param_name: str, vec: Any, expected_len: int
) -> tuple[list[float] | None, str | None]:
    """Validate an optional pose vector and normalize it to plain floats.

    Membership, not truthiness: a pose parameter is "supplied" when it is not
    ``None``. Testing the vector itself (``if position:``, ``position or
    <default>``) is wrong twice over. A NumPy array - the natural product of any
    pose arithmetic, and what every docstring here advertises as accepted -
    raises a bare ``ValueError: truth value of an array ... is ambiguous``
    through the structured tool-result contract, and an empty vector reads as
    "omitted", so the default is substituted (or the write skipped) while the
    call reports success.

    Normalizing to a ``list[float]`` keeps the accepted NumPy input from
    outliving this boundary: the pose is stored on :class:`SimObject` /
    :class:`SimRobot` (both annotated ``list[float]``), echoed in the status
    text, and written into the spec, so a raw ``np.float64`` element would leak
    ``np.float64(0.05)`` into agent-visible output.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        vec: The caller's value, or ``None`` when the parameter was omitted.
        expected_len: Component count the target buffer defines (3 for a
            position, 4 for a wxyz quaternion).

    Returns:
        ``(None, None)`` when ``vec`` is ``None`` (omitted - the caller applies
        its own default), ``(floats, None)`` for an acceptable vector, or
        ``(None, error_message)`` for a wrong length, a non-numeric element or a
        ``nan``/``inf`` component.
    """
    if vec is None:
        return None, None
    if (err := pose_vector_error(method, param_name, vec, expected_len)) is not None:
        return None, err
    return [float(v) for v in vec], None


def camera_fov_error(method: str, param_name: str, value: Any) -> str | None:
    """Return an error message if ``value`` is not a usable camera field of view.

    A camera's vertical field of view must be a finite angle in the open
    interval ``(0, 180)`` degrees. This lives here, beside
    :func:`pose_vector_error`, for the same reason that one does: the MuJoCo and
    Newton backends both expose ``add_camera(fov=...)`` and their accepted
    domain must not diverge - an fov either backend refuses must be refused by
    the other, and a second copy of the interval would drift from the first.

    Outside that interval a camera is not merely mis-posed but unusable, and
    each backend fails differently and late rather than at config time:

    * MuJoCo bakes the value into the MJCF ``fovy`` attribute, so the spec
      recompile aborts inside ``inject_camera_into_scene`` with a cryptic "spec
      recompile refused".
    * Newton stores it and derives the pinhole intrinsics
      ``0.5 * height / tan(radians(fov) / 2)`` at render time, which is ``nan``
      for a ``nan`` fov and raises ``ZeroDivisionError`` for ``0``.

    A NumPy real scalar is accepted (``np.float32(58.0)`` read out of a config
    array is a legitimate fov); a ``bool`` is refused because ``float(True)``
    would silently mean a 1-degree lens. Returns ``None`` when ``value`` is a
    usable field of view.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        return f"{method}: '{param_name}' must be a finite number in degrees, got {value!r}."
    if not (0.0 < float(value) < 180.0):
        return f"{method}: '{param_name}' must be in the open interval (0, 180) degrees, got {value}."
    return None


def entity_name_error(method: str, param_name: str, name: Any) -> str | None:
    """Return an error message if ``name`` cannot address the entity it names.

    The creation-site counterpart to the total lookups in
    :mod:`strands_robots.simulation.models`. A lookup asks whether a name
    addresses something, so a name that cannot be a registry key is honestly
    "absent"; a creation site *claims* a name, so the same value has to be
    refused instead - an entity registered under a name that nothing can
    resolve is unreachable through the very API that created it.

    It lives here, beside :func:`camera_fov_error`, for the same reason that one
    does: ``add_object`` / ``add_camera`` / ``add_robot`` exist on more than one
    backend and their accepted domain must not diverge - a name one backend
    refuses must be refused by the others, and a second copy of the rule would
    drift from the first.

    Three values are refused, each because it produces an entity that some part
    of this API cannot address (measured on MuJoCo 3.11.0, one ``create_world``
    then ``add_object``):

    * **Not a ``str``.** ``add_object(7, ...)`` registered the key ``7`` and
      only then raised ``TypeError`` out of MuJoCo's ``add_body``, leaving the
      world holding an entry for a body that does not exist; a name that is not
      hashable at all (``["x"]``) raised out of the duplicate-name test.
      ``add_robot(7, ...)`` did compile, under the int registry key ``7`` -
      which the tool surface, where every name arrives as a JSON string, can
      never address. A falsy non-``str`` was worse than either: ``add_robot(0)``
      silently derived the label ``"arm"`` from the model and reported success
      under a name the caller never asked for.
    * **The empty string.** ``""`` is MuJoCo's own sentinel for an unnamed
      entity, so ``mj_name2id(model, BODY, "")`` returns ``-1``:
      ``add_object("")`` succeeded and ``get_body_state(body_name="")`` then
      reported ``Body '' not found``. For a camera it also collides with a
      routing token - ``render(camera_name="")`` selects the free camera - so a
      camera created as ``""`` can never be rendered from.
    * **A ``str`` containing a NUL.** The compiled model compares names only up
      to the first NUL, so ``add_object("a\\x00b")`` registered ``'a\\x00b'``
      while the body compiled as ``'a'``, and ``mj_name2id(..., "a\\x00b")``
      returned the id of ``'a'``. Two names, one entity, each layer believing a
      different one. Through ``add_robot`` the NUL took the namespace separator
      with it: the bodies compiled as ``'abase'`` / ``'alink'`` rather than
      under the ``'a\\x00b/'`` prefix ``list_bodies`` looks for.

    Nothing else is refused. In particular this is NOT the MJCF-interpolation
    allowlist (``^[a-zA-Z0-9_-]+$``): a namespaced body label (``so101/gripper``)
    and a dotted or unicode object name are all addressable, and narrowing the
    domain to an allowlist is a separate decision from refusing a name that
    demonstrably cannot address its entity.

    A ``str`` subclass is accepted - it is a string by every operation here and
    by the registry. Callers with a documented "derive the name" input (the
    MuJoCo ``add_robot(name=None)`` / ``add_robot(name="")`` short form) must
    check for that input before calling this, since ``""`` is refused here.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        name: The caller's value.

    Returns:
        ``None`` when ``name`` can address the entity it creates, otherwise the
        error message to report through the structured tool-result dict.
    """
    if not isinstance(name, str):
        return (
            f"{method}: '{param_name}' must be a non-empty string, got {name!r} "
            f"({type(name).__name__}); an entity is addressed by name and every "
            "agent-tool call carries that name as a string."
        )
    if not name:
        return (
            f"{method}: '{param_name}' must be a non-empty string, got ''; an empty "
            "name is the backend's own sentinel for an unnamed entity, so the entity "
            "created under it could not be addressed afterwards."
        )
    if "\x00" in name:
        return (
            f"{method}: '{param_name}' must not contain a NUL character, got {name!r}; "
            "the compiled model reads a name only up to the first NUL, so the registry "
            "and the model would disagree about the entity's name."
        )
    return None


def validation_split_fraction(val_episodes: int, total_episodes: int) -> float:
    """``dataset.eval_split`` that holds out exactly ``val_episodes`` episodes.

    lerobot builds its held-out validation set by taking
    ``ceil(n_episodes * eval_split)`` episodes from each task's tail, so every
    fraction in ``((N - 1) / total, N / total]`` holds out exactly ``N``. This
    returns the MIDPOINT of that interval rather than its ``N / total`` upper
    bound, because the bound is not float-safe: ``25 * (7 / 25)`` evaluates to
    ``7.000000000000001``, whose ceiling is 8 - one episode more than the caller
    asked to reserve. The midpoint leaves half an episode of slack on either
    side, so the requested count survives the round trip at every dataset size.

    Args:
        val_episodes: Number of episodes to hold out. Must be positive and
            smaller than ``total_episodes``; callers validate that themselves so
            they can name their own parameter in the error.
        total_episodes: Episode count of the dataset being split.

    Returns:
        The fraction to pass as lerobot's ``--dataset.eval_split``.
    """
    return (val_episodes - 0.5) / total_episodes


def validation_split_error(val_episodes: int, total_tasks: Any, context: str) -> str | None:
    """Error text when a global episode COUNT cannot be honored as a split.

    lerobot expresses a validation split as one ``eval_split`` FRACTION and
    applies ``ceil(n_episodes * eval_split)`` to each task independently, so a
    single fraction reproduces a global episode count only on a single-task
    dataset. With ``T > 1`` tasks the ceiling is taken ``T`` times and the total
    held out is generally not the requested ``N`` - reserving 1 episode of a
    two-task dataset would hold out 2. Rather than quietly reserve a different
    number of episodes than asked, callers refuse and point at the fraction,
    which addresses the per-task behaviour directly.

    A ``total_tasks`` of 0 or ``None`` means the dataset does not record a task
    count (lerobot's own field defaults to 0), which is treated as single-task.

    Args:
        val_episodes: The requested held-out episode count, for the message.
        total_tasks: ``total_tasks`` from the dataset's ``meta/info.json``.
        context: Caller label the message is prefixed with.

    Returns:
        The error text, or None when the count can be honored exactly.
    """
    if not isinstance(total_tasks, int) or isinstance(total_tasks, bool) or total_tasks <= 1:
        return None
    return (
        f"{context}: val_episodes={val_episodes} cannot be reserved exactly on a "
        f"dataset with {total_tasks} tasks. A validation split is a per-task "
        "fraction in lerobot (it holds out ceil(episodes_in_task * eval_split) "
        "from every task), so a single global count is not expressible: the "
        "ceiling would be applied once per task. Pass the fraction directly, "
        "e.g. extra_flags={'dataset.eval_split': 0.1, 'eval_steps': 1000}, and "
        "the split will hold out a tenth of each task."
    )
