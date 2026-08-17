"""Shared utilities for strands-robots."""

import importlib
import logging
import math
import numbers
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# Cache of lazy-loaded modules
_lazy_modules: dict[str, object] = {}


def require_optional(
    module_name: str,
    *,
    pip_install: str | None = None,
    extra: str | None = None,
    purpose: str = "",
    system_install: str | None = None,
) -> object:
    """Import an optional dependency, raising a clear error if missing.

    Once imported, the module is cached so subsequent calls are free.

    Args:
        module_name: Dotted module name to import (e.g. ``"zmq"``).
        pip_install: Explicit pip package name if it differs from *module_name*.
        extra: ``pyproject.toml`` extras group (e.g. ``"groot-service"``).
        purpose: Human-readable description shown in the error message.
        system_install: Remedy for a module that arrives with a system package
            rather than from an index - the ROS 2 client libraries are the case
            in this package. Replaces the ``pip install`` block entirely, and
            *pip_install* / *extra* are then not consulted, because a pip
            command for such a module is a remedy the caller can follow to no
            effect: it either installs something that leaves the module exactly
            as missing, or fails outright.

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
        parts = [f"'{module_name}' is required"]
        if purpose:
            parts[0] += f" for {purpose}"
        if system_install is not None:
            # No pip line at all: naming one here would hand the caller an
            # instruction that reports success without supplying the module.
            parts.append(system_install)
        else:
            install_hint = pip_install or module_name
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


def is_boolean(value: Any) -> bool:
    """Return True when ``value`` is a python or a numpy boolean.

    The single boolean predicate behind every numeric domain in this module and
    the runtime writers that reuse them. Two properties make a boolean worth its
    own check rather than letting the numeric coercion decide:

    * ``bool`` is an ``int`` subclass, so ``float(True)`` is ``1.0`` and a
      boolean survives every ``float()`` / ``numbers.Real`` gate as a silent
      ``1.0`` - one radian, one metre, one Newton, depending on where it landed.
    * ``numpy.bool_`` is *not* a ``bool`` subclass, so ``isinstance(value, bool)``
      alone misses the boolean a policy or a comparison produces
      (``gripper > 0.5``). It is also not registered as ``numbers.Real``, which
      is why the vector domains here reject it through their ``numbers.Real``
      check; a writer that coerces with a bare ``float()`` has no such backstop
      and needs this predicate.

    The ``.item()`` unwrap covers a numpy boolean scalar and a 0-d boolean array
    while leaving every numeric scalar - and every multi-element array, which
    has no single item - reported as non-boolean.
    """
    if isinstance(value, bool):
        return True
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return isinstance(item(), bool)
        except (TypeError, ValueError):  # a multi-element array has no single item
            return False
    return False


def sequence_length(value: Any) -> int | None:
    """Return the length of ``value``, or ``None`` when it does not carry one.

    Every validator that accepts a vector first asks "how many components is
    this?", and the obvious spelling - ``hasattr(value, "__len__")`` followed by
    ``len(value)`` - is unsafe for the value class this library actually
    receives. A 0-d numpy array (``np.array(0.5)``, or the result of a reduction
    such as ``np.mean(...)``) and a 0-d torch tensor both *declare* ``__len__``
    and then raise from it, so the ``hasattr`` probe passes and the ``len()``
    call escapes with a bare ``len() of unsized object`` that names neither the
    parameter nor the method - past agent-tool dispatch, which is documented
    never to raise.

    ``TypeError`` covers the two spellings this probe was written for - CPython
    raises it for a value carrying no ``__len__`` at all (a plain ``float``, a
    ``numpy.float64``) and for the 0-d array whose ``__len__`` exists and
    refuses - but it is not the superset it was once documented as, and the gap
    needs no hostile value to reach. ``len()`` converts whatever ``__len__``
    returns into an index, and that conversion has refusals of its own: a
    negative length raises ``ValueError`` and a length past ``sys.maxsize``
    raises ``OverflowError``, both from CPython rather than from the value. A
    ``__len__`` that computes its answer - ``self._end - self._start`` on a
    proxy whose window is inverted - is ordinary Python returning an ordinary
    ``int`` and reaches neither branch a ``TypeError`` probe has.

    So the exception's type is not the question being asked. Every one of these
    answers the caller's question the same way - this value does not carry a
    readable component count - so all of them report as ``None`` and one branch
    covers them. A probe on the path whose whole purpose is to answer an
    unusable input with a message cannot assume good behaviour of the value it
    was handed, which is the property #1873, #1874, #1875 and #1878 established
    on four neighbouring guards. Only ``len(value)`` runs inside the ``try``, so
    a broad clause here masks no logic of this library's own; ``Exception``
    rather than ``BaseException`` keeps ``KeyboardInterrupt`` and ``SystemExit``
    propagating.

    Args:
        value: Any caller-supplied value a validator needs the length of.

    Returns:
        The component count, or ``None`` when the value has no readable length.
    """
    try:
        return len(value)
    except Exception:
        return None


def _refusal_repr(value: Any) -> str:
    """``repr(value)`` for a refusal message, or a description when it cannot be built.

    Every scalar guard below renders the value it refuses through this, and none
    of them renders one any other way. Rendering a rejected value must not be
    able to raise: it runs only on the path whose entire purpose is to answer an
    unusable input with a structured message instead of an exception, and every
    caller of every one of those guards documents that ``{status, content}``
    result as the only channel a bad value is reported on. A guard that raises
    while building a refusal fails on exactly the path that exists so it does
    not.

    Two cases reach it, and neither is hypothetical. ``repr`` of an ``int`` wider
    than :func:`sys.get_int_max_str_digits` (4300 digits by default) raises
    ``ValueError``, and ``device_connect``'s ``@rpc()`` surfaces forward a remote
    caller's number unchanged while Python integers are arbitrary-precision. And
    a third-party type may raise anything at all from its own ``__repr__`` -
    :class:`numbers.Real` is a registration rather than an inheritance, so a
    scalar that satisfies a guard's type test owes it nothing else - which is why
    the guarantee here is unconditional rather than a list of the exceptions
    known today.

    ``int.bit_length`` needs no decimal conversion, so the value most likely to
    arrive unprintable is still described by its magnitude rather than reported
    as an opaque type name.

    Args:
        value: The rejected value.

    Returns:
        Its ``repr``, or a bracketed description when that cannot be produced.
    """
    try:
        return repr(value)
    except Exception:
        return _describe_unrenderable(value)


def _refusal_str(value: Any) -> str:
    """``str(value)`` for a refusal message, or a description when it cannot be built.

    The :func:`_refusal_repr` counterpart for the two messages that report a
    value plainly rather than quoted, where ``repr`` is not interchangeable:
    NumPy 2 reprs a scalar with its type, so rendering an ``np.float32`` fov
    through ``repr`` would silently turn ``got 200.0`` into
    ``got np.float32(200.0)`` in text an agent reads.

    ``str`` is exactly as able to raise as ``repr`` and for the same two reasons:
    it falls back to ``__repr__`` when a type defines no ``__str__``, and
    ``str`` of an ``int`` wider than :func:`sys.get_int_max_str_digits` performs
    the same decimal conversion. So a plainly-rendered value needs the same
    guarantee rather than a weaker one.

    Args:
        value: The rejected value.

    Returns:
        Its ``str``, or a bracketed description when that cannot be produced.
    """
    try:
        return str(value)
    except Exception:
        return _describe_unrenderable(value)


def _describe_unrenderable(value: Any) -> str:
    """Describe a value whose own rendering raised.

    Shared by :func:`_refusal_repr` and :func:`_refusal_str` so the two render
    forms cannot describe the same unrenderable value differently.

    ``int.bit_length`` needs no decimal conversion, so the value most likely to
    arrive unrenderable - an ``int`` past the interpreter's digit limit - is
    still described by its magnitude rather than reported as an opaque type
    name. The sign is not recoverable from a bit length, which is acceptable
    because every guard's own text states the domain the value was refused
    against.

    Args:
        value: The value whose rendering raised.

    Returns:
        A bracketed description, which is built from the type and (for an
        integer) a bit count, so it cannot itself raise.
    """
    if isinstance(value, int):
        return f"<int of {value.bit_length()} bits>"
    return f"<unrepresentable {type(value).__name__}>"


def _describe_failed_read(exc: Exception) -> str:
    """``RuntimeError: no keys for you`` for a read a refusal could not complete.

    The one spelling of a failed read in this module's refusal text, shared by
    the guard that reports a read whose *verdict* it became
    (:func:`_read_name_list`) and the read a message makes only to *quote*
    something (:func:`_read_to_quote`). Named rather than repeated for the reason
    :func:`_describe_unrenderable` is shared between the two render forms: a
    refusal degraded on one path must not describe the same failure differently
    from another.

    ``exc`` goes through :func:`_refusal_str` rather than being interpolated: a
    value hostile enough to raise from its own read is not one whose exception is
    assumed to have a working ``__str__``, which would be the #1873 escape
    reintroduced inside a message built to avoid it.

    Args:
        exc: The exception the read raised.

    Returns:
        Its type name and text, which cannot itself raise.
    """
    return f"{type(exc).__name__}: {_refusal_str(exc)}"


def _read_to_quote(value: Any) -> tuple[list[Any] | None, str | None]:
    """The elements of ``value`` for a refusal to quote, or why they could not be read.

    The read a refusal makes for its own *text*, which is a different job from
    :func:`_read_name_list`'s and needs the opposite answer. There the verdict
    depends on the read, so a read that fails becomes the verdict. Here the
    verdict is settled *before* the read happens - a mapping is refused whatever
    its keys say, and a string whatever its characters are - so a read that fails
    must not replace it: the caller passed a mapping, and that is the mistake
    worth reporting even when its keys cannot be quoted. The failure is therefore
    described for the message to carry rather than returned as a message of its
    own (#1903).

    Args:
        value: The caller-supplied value a refusal wants to quote.

    Returns:
        ``(elements, None)`` when the read finished, or ``(None, description)``
        when it did not. Exactly one side is ever populated, so a caller holding
        a ``None`` description holds the elements.
    """
    try:
        return list(value), None
    except Exception as exc:
        return None, _describe_failed_read(exc)


def _refusal_container_repr(value: Any) -> str:
    """``repr(value)`` for a refusal that reports a whole container, elementwise if it must.

    The container counterpart to :func:`_refusal_repr`, and the reason the two
    cannot be one function. ``repr`` of a list recurses into its elements, so a
    container is unrenderable whenever any *one* of its elements is, and
    :func:`_refusal_repr`'s whole-value fallback would answer that with
    ``<unrepresentable list>`` - erasing every element that rendered perfectly
    well, and the element count with them. That count is frequently the entire
    reason for the refusal (``must be a 3-element vector, got 4``), so a
    container needs a *rendering* rather than a fallback.

    ``repr`` is tried on the whole container first, so a container that can
    render itself is reported in exactly the text it was before this existed -
    including a ``tuple``, a ``dict`` and a NumPy array, whose own reprs
    (``(1.0, 2.0)``, ``{'a': 1}``, ``array([1., 2.])``) no elementwise form
    reproduces. Only when that raises is the container described component by
    component, substituting just the components that cannot print::

        [1.0, <int of 16610 bits>, 3.0]

    The offending component is located by its **position**, not by an inserted
    index, so the fallback keeps the shape of the ``repr`` it stands in for.
    Every guard that names an index does so in its own text (``{param}[{i}]``),
    which is unchanged; an index inserted here too would state it twice, in two
    forms that could disagree.

    Nothing is elided. Truncating would erase elements that rendered fine, which
    is the exact failure of the whole-value fallback this exists to avoid, and
    ``repr`` of a long container - the text this stands in for - is not
    truncated either.

    A :class:`~collections.abc.Mapping` is rendered as a mapping rather than as
    the list of its keys, because :func:`name_list_error` refuses one *for*
    discarding its values: a message showing only the keys would perform the
    discarding it is complaining about.

    The rendering is one level deep. These containers carry scalars by contract -
    a nested list is itself a refusal reason, reported as ``elements must be
    numbers`` - so an inner container's contents are never what the message is
    about, and recursing would additionally have to track what it had already
    visited to terminate on a self-referential value, which the interpreter's own
    ``repr`` does for the fast path above and a hand-written walk would not.

    Args:
        value: The rejected container.

    Returns:
        Its ``repr``; an elementwise rendering when that raises; or a bracketed
        description when ``value`` cannot be iterated either, which is
        :func:`_refusal_repr`'s answer for a value that is not a container at
        all - every one of these guards accepts ``Any``, so a scalar reaches
        them too.
    """
    try:
        return repr(value)
    except Exception:
        pass
    if isinstance(value, Mapping):
        try:
            items = list(value.items())
        except Exception:
            return _describe_unrenderable(value)
        return "{" + ", ".join(f"{_refusal_repr(key)}: {_refusal_repr(val)}" for key, val in items) + "}"
    try:
        elements = list(value)
    except Exception:
        return _describe_unrenderable(value)
    return "[" + ", ".join(_refusal_repr(element) for element in elements) + "]"


def _beyond_float_range(value: Any) -> bool:
    """Whether ``float(value)`` overflows, i.e. no float64 stands for ``value``.

    Not a domain in its own right - a helper that names, in one place, the
    condition the three numeric guards below need a *reason* for rather than a
    crash. ``float()`` raises ``OverflowError`` for a real whose magnitude
    exceeds :data:`sys.float_info.max`, and Python integers are
    arbitrary-precision while ``device_connect``'s ``@rpc()`` surfaces forward a
    remote caller's number unchanged, so one is a request away.

    The condition is *not* limited to ``int``: ``Fraction(10**400, 3)`` is a
    registered :class:`numbers.Real` that overflows identically, which is why
    the guards ask this question of the conversion rather than of the type.

    Returns:
        ``True`` when the conversion overflows. ``False`` when it succeeds *or*
        fails any other way - a :class:`numbers.Real` registration guarantees no
        working ``__float__``, and a value no number can be read from at all is
        not a magnitude complaint and must not be reported as one.
    """
    try:
        float(value)
    except OverflowError:
        return True
    except Exception:
        return False
    return False


def positive_finite_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable positive finite number.

    Shared domain for every CONTINUOUS knob a caller supplies as a positive
    real: a rate or a span of time (a control-loop frequency in Hz, a rollout
    or teleop ``duration`` in seconds), and a dimensionless multiplier (the
    terrain curriculum ``difficulty`` that scales a heightfield's peak
    elevation). Unlike :func:`positive_whole_number_error` a fractional value is
    perfectly usable here (``2.5`` seconds, ``62.5`` Hz, a ``0.5`` scale), so
    only the sign and the finiteness are constrained. It lives here rather than beside one of its
    callers because those callers sit in different layers
    (:mod:`strands_robots.teleop_mixin` must not depend on
    :mod:`strands_robots.simulation`), and the accepted domain must not diverge
    between them.

    Only a positive finite value can be honored. Such a knob is always a
    divisor (the loop period is ``1 / hz``), a horizon (``duration *
    frequency`` steps) or a multiplier (``elevation * difficulty``), so ``0``
    makes the period undefined, the horizon empty or the scaled quantity
    degenerate, a negative value inverts it, ``nan`` poisons every comparison it
    reaches (``nan > 0`` and ``nan <= 0`` are both ``False``), and ``inf``
    collapses the period to ``0`` - an unthrottled loop, not a fast one.
    Accepts any real scalar (so a NumPy ``np.float32`` rate read from a config
    array passes) and rejects ``bool`` explicitly - an ``int`` subclass whose
    ``True`` would act as a silent ``1``.

    **A value past the float64 range is refused with a reason of its own.**
    ``10**400`` is positive and finite, so ``must be > 0`` would be a false
    statement about it, and it used to raise ``OverflowError`` out of the
    ``float()`` this guard converts with - failing on the path that exists so it
    does not. It is not a new boundary: this guard already accepts up to
    ``sys.float_info.max`` (``1e300`` and ``10**308`` both pass) and stopped
    dead one step past it, so the range is the edge the domain always had and
    all that changes is that the guard now names it. Nothing raises out of here
    for any real scalar; see :func:`_beyond_float_range`.

    Args:
        value: The caller-supplied value.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            normally the public method name.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return f"{context}: {param} must be > 0, got {_refusal_repr(value)}."
    if _beyond_float_range(value):
        # A real past the float64 range is positive-or-negative and finite, so
        # neither of this guard's own reasons is true of it - hence its own text.
        # Refusing stays right: the value is a divisor (``1 / hz``) or a
        # multiplier evaluated in float64, and no float64 stands for it.
        return f"{context}: {param} must be within the range of a 64-bit float, got {_refusal_repr(value)}."
    try:
        # ``isfinite`` before the sign test: ``nan`` is never ``<= 0``, so
        # ordering these the other way lets it through.
        unusable = not math.isfinite(float(value)) or float(value) <= 0
    except Exception:
        # A ``numbers.Real`` registration owes this guard no working
        # ``__float__``, and a value no number can be read from is refused for
        # the same reason a non-real one is - the message it already had.
        unusable = True
    if unusable:
        return f"{context}: {param} must be > 0, got {_refusal_repr(value)}."
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

    **A value past the float64 range is refused with a reason of its own.**
    The paragraph above is the argument for it: an accepted value goes onto the
    wire as an IEEE-754 float64, and ``10**400`` has no float64 form to put
    there. Its own reason is needed because ``10**400`` *is* a finite number, so
    ``must be a finite number`` would be false of the one value class most in
    need of an honest refusal - and before that reason existed the guard raised
    ``OverflowError`` here instead of answering.

    Args:
        value: The caller-supplied value.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            normally the public method name.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return f"{context}: {param} must be a finite number, got {_refusal_repr(value)}."
    if _beyond_float_range(value):
        # ``10**400`` *is* a finite number, so this guard's own reason would be
        # a false statement about it. Refusing stays right: the docstring above
        # is explicit that an accepted value is serialized onto the wire as an
        # IEEE-754 float64, and this one has no float64 form to serialize.
        return f"{context}: {param} must be within the range of a 64-bit float, got {_refusal_repr(value)}."
    try:
        unusable = not math.isfinite(float(value))
    except Exception:
        unusable = True
    if unusable:
        return f"{context}: {param} must be a finite number, got {_refusal_repr(value)}."
    return None


def positive_whole_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable positive whole number.

    Shared domain for two families of positive discrete quantity:

    * The media knobs that count frames or pixels - the recorders' ``fps``,
      ``width``, ``height`` and in-memory frame cap, the
      ``run_policy(video=...)`` dict fields, and the
      :func:`~strands_robots.rendering.encode_clip` playback rate.
    * The physics steps one applied action is held for - the ``n_substeps`` of
      every backend's
      :meth:`~strands_robots.simulation.base.SimEngine.send_action`.

    It lives here rather than beside one of its callers because those callers sit
    in different layers (:mod:`strands_robots.rendering` must not depend on
    :mod:`strands_robots.simulation`), and the accepted domain must not diverge
    between them. Only a positive whole number can be honored: ``0`` makes the
    capture loop's ``1 / fps`` period undefined, a negative rate is rejected by
    the ffmpeg writer, a zero/negative frame cap drops every frame, and a
    ``send_action`` advancing no physics leaves a target written that the world
    never integrates - which is why ``0`` is refused there rather than honored
    as "write but do not advance", with
    :meth:`~strands_robots.simulation.base.SimEngine.step` named as the surface
    that advances a count of its own. Accepts any real scalar with an integral
    value (so a NumPy ``np.int64`` height or a ``30.0`` computed from a config
    float passes) and rejects ``bool`` explicitly - an ``int`` subclass whose
    ``True`` would act as a silent 1.

    **Magnitude is part of this domain, unlike its ``non_negative`` sibling.**
    :func:`non_negative_whole_number_error` accepts ``10**400`` as a matter of
    documented policy, on the grounds that a per-call ceiling belongs to the
    consumer - and for its one caller that holds, because MuJoCo's
    ``_MAX_STEPS_PER_CALL`` is exactly such a ceiling and refuses an outsized
    step count with a reason of its own. Exactly one consumer of *this* domain
    owns such a ceiling: :func:`coerce_zmq_timeout_ms` composes this guard and
    then applies :data:`MAX_ZMQ_TIMEOUT_MS`, because a ZMQ send/receive timeout
    is stored as a C ``int`` and this domain accepts ``2**31``.
    Its callers are ``fps``, ``width``, ``height``, ``max_frames``, the mesh
    robots' ``drive(count=...)``, ``send_action(n_substeps=)`` and the ZMQ
    clients' ``timeout_ms``. Two of those
    repeat work that nothing bounds: ``drive`` repeats an actuation command, so
    an unbounded count is an unbounded actuation loop against a physical robot
    rather than a slow call, and ``n_substeps`` is a physics-step count that
    ``_MAX_STEPS_PER_CALL`` does *not* reach - that ceiling is applied by
    ``step`` alone, so a count ``step`` refuses is still accepted through
    ``send_action`` on every backend. So the two guards are the same scalar
    policy with the floor moved *and* one deliberate difference, which is
    recorded here rather than left for a reader to find by measuring.

    That last point is a boundary, not a claim to have closed it: refusing
    ``10**400`` here is the float64 edge the domain already had, and choosing a
    *resource* ceiling for a substep count is the same per-backend decision
    tracked for ``step`` in #1871.

    Refusing is therefore the verdict, and it needs a reason of its own:
    ``10**400`` *is* a positive whole number. As above, the float64 range is not
    a new boundary - ``1e300`` is accepted by this guard today - it is the edge
    the domain already had, at which the guard used to raise ``OverflowError``
    rather than answer.

    Args:
        value: The caller-supplied value.
        param: The parameter (or dict key) it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            ``"video"`` for the :class:`VideoConfig` dict, the method name for a
            keyword parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """

    def message() -> str:
        # Rendered on demand, not up front. The text used to be built on this
        # function's first line, before the value had been classified at all, so
        # ``repr`` raised on an outsized ``int`` ahead of every verdict - the
        # guard failing while preparing a refusal it had not decided to return,
        # and doing it on the accept path too.
        return f"{context}: {param} must be a positive whole number, got {_refusal_repr(value)}."

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return message()
    if _beyond_float_range(value):
        # ``10**400`` is a positive whole number, so ``message()`` would state
        # something false about it. It is refused rather than accepted, and
        # deliberately unlike its ``non_negative`` sibling - see the docstring.
        return f"{context}: {param} must be within the range of a 64-bit float, got {_refusal_repr(value)}."
    try:
        numeric = float(value)
    except Exception:
        # No number can be read from it at all; refused, never raised.
        return message()
    # ``isfinite`` first: ``int(nan)`` raises, and short-circuiting keeps it
    # out of the integrality check below.
    if not math.isfinite(numeric) or numeric != int(numeric) or numeric < 1:
        return message()
    return None


def non_negative_whole_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable non-negative whole number.

    Shared domain for two families of discrete quantity whose ``0`` is a real
    setting rather than a degenerate one: the number of physics steps a caller
    asks a simulation to advance - the ``n_steps`` of every backend's
    :meth:`~strands_robots.simulation.base.SimEngine.step` - and the two
    whole-number teleop knobs :mod:`~strands_robots.tools.lerobot_teleoperate`
    puts on the lerobot CLI, where ``dataset_reset_time_s=0`` is "no operator
    pause between recorded episodes" and ``replay_episode=0`` is the first
    episode.

    Not the only physics-step count in the tree, and the difference is the
    floor rather than the scalar policy: the ``n_substeps`` of
    :meth:`~strands_robots.simulation.base.SimEngine.send_action` is guarded by
    :func:`positive_whole_number_error`, because that surface writes an actuator
    target before it advances and a ``0`` there leaves the target written and
    never integrated. ``step`` owns the honored zero; ``send_action`` refuses
    its own and names ``step``. A reader arriving here from ``send_action``
    would otherwise take this floor for that surface's.

    It stands to :func:`positive_whole_number_error` exactly as
    :func:`non_negative_count_error` stands to :func:`positive_count_error`: the
    same scalar policy with the floor moved to ``0``, because ``0`` is a
    first-class value here rather than a degenerate one. The MuJoCo backend has
    documented ``step(0)`` as "an accepted no-op" since its numeric inputs were
    hardened, and an agent loop that computes ``n_steps`` from a remaining
    duration reaches ``0`` on the last iteration of a rollout that has run out
    of time. Refusing it would reject that loop's final call, which is why this
    is a separate domain rather than a caller of the positive one.

    Accepts any real scalar with an integral value, so a step count that came
    from arithmetic - ``int(duration / dt)`` promoted to ``np.int64`` by a NumPy
    dt, or a ``30.0`` read from a config - is honored. The caller coerces the
    accepted value with ``int()`` before using it as a ``range()`` bound; that
    coercion is safe *because* this guard has already performed it and compared
    the result back, which is the whole reason the two steps are ordered this
    way. ``bool`` is rejected explicitly: it is an ``int`` subclass, so a bare
    ``value < 0`` test lets ``True`` through as a silent count of one physics
    step, and ``numpy.bool_`` is not registered as ``numbers.Real`` so it is
    refused by the scalar check.

    Every real scalar gets a verdict; nothing raises out of here. That is the
    contract, not a detail, because the callers document their structured
    ``{status, content}`` result as the only channel a bad count is reported
    through, and one of them takes its count from a remote process. It is why
    the integrality test is an ``int()`` in a ``try`` rather than a ``float()``
    round-trip: ``float`` raises ``OverflowError`` for an ``int`` wider than a
    float, which turned a refusal into a crash for the very values most in need
    of one.

    **Magnitude is not part of this domain.** ``10**400`` is a non-negative
    whole number and is accepted as one. Whether a count is too large to advance
    in a single call is a per-call resource policy - MuJoCo's
    ``_MAX_STEPS_PER_CALL``, which refuses it with a reason of its own - and a
    domain guard that quietly stopped at the float range would be exactly the
    kind of boundary-by-implementation-accident this family exists to remove.

    **This is the one place the two guards differ**, and the ceiling above is
    the whole reason: :func:`positive_whole_number_error` refuses an outsized
    value because none of its consumers owns such a ceiling and one of them
    repeats a command to a robot. So "the same policy with the floor moved" is
    true of every other input and not of this one. The asymmetry is deliberate;
    if a ceiling ever appears on that side, this is the paragraph to revisit.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            public method name, or the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """

    def message() -> str:
        # Rendered on demand, not up front: an accepted count must neither pay
        # for nor be refused by a message it never receives. A count wider than
        # ``sys.get_int_max_str_digits()`` is accepted here, and building the
        # text eagerly made ``repr`` raise on it - the guard failing on the
        # accept path, doing work only the refuse path needs.
        return f"{context}: {param} must be a non-negative whole number, got {_refusal_repr(value)}."

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return message()
    try:
        integral = int(value)
    except (OverflowError, TypeError, ValueError):
        # The one conversion answers every value no count can be read from:
        # ``int(nan)`` raises ``ValueError`` and ``int(+-inf)``
        # ``OverflowError``, so no separate ``isfinite`` check is needed - and
        # none is possible, because ``isfinite`` needs a ``float()`` whose own
        # ``OverflowError`` on an ``int`` wider than a float is the escape this
        # form exists to remove. ``TypeError`` covers a ``numbers.Real``
        # registered without an ``__int__``: refused, never raised.
        return message()

    # ``int`` rather than ``math.trunc``, which looks like the conversion the
    # ABC guarantees and is not usable here: NumPy scalars implement
    # ``__int__`` but not ``__trunc__``, so ``math.trunc(np.int64(3))`` raises
    # ``TypeError`` and the NumPy rows this domain must honor would be refused.
    #
    # The sign is read off the coerced ``int`` rather than the original: an
    # ``int``-to-``int`` comparison is total, where ``value < 0`` would defer to
    # a ``__lt__`` a ``numbers.Real`` registration does not actually guarantee.
    # ``integral != value`` is what establishes integrality - exactly, and with
    # no float round-trip, so a large-but-usable count is neither rounded nor
    # refused - and it falls back to Python's default inequality rather than
    # raising when a type supplies no ``__eq__``.
    if integral < 0 or integral != value:
        return message()
    return None


def step_aborted_msg(completed: int, requested: int, *, context: str = "step") -> str:
    """Refusal text when a batched step loop loses its world mid-run.

    Pairs with :func:`non_negative_whole_number_error`, which settles the step
    *count*; this settles what a backend says when a count it accepted becomes
    unfinishable partway through. Every backend's ``step`` releases
    ``SimEngine._STEPS_PER_BATCH`` steps' worth of lock so concurrent actions can
    interleave, and releasing the lock is what lets a concurrent teardown - the
    ``cleanup`` world handoff of GH #116 - land between two batches.

    The count is in the text because some of it happened: a bare "no world"
    would read as the call having done nothing, which is the same
    report-disagrees-with-the-world defect the step-count domain removed. It is
    one shared helper rather than three literals so the three backends refuse
    identically in wording and not merely in verdict, as the pose, colour and
    size domains do.

    Args:
        completed: Steps actually advanced before the world went away.
        requested: The count the caller asked for.
        context: Calling method name, for the message prefix.

    Returns:
        Human-readable refusal text.
    """
    return (
        f"{context}: world was destroyed mid-run after {completed} of {requested} steps; aborting. "
        "The steps already advanced are not rolled back."
    )


def positive_count_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable positive integer count.

    Shared domain for three families of discrete quantity:

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
    * A bound on a slice of a token sequence - the ``tokenizer_max_length`` a
      VLA provider hands a HuggingFace tokenizer as ``max_length`` alongside
      ``truncation=True``. The tokenizer takes it as a slice bound over the
      encoded instruction, so a count below one silently produces an EMPTY
      prompt rather than an error.

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
        return f"{context}: {param} must be a positive integer, got {_refusal_repr(value)}."
    return None


def tcp_port_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` cannot address a TCP port.

    Shared domain for every caller-supplied port number: the agent tools that
    reach a service over TCP (``use_rosbridge``'s WebSocket,
    ``gr00t_inference``'s inference service), the mesh bridges that construct
    one, the policy providers that dial one (``groot``, ``moveit2``,
    ``cosmos3``, ``lerobot_async``, ``vera``), the Device Connect drivers
    that address a device daemon
    (:class:`~strands_robots.device_connect.reachy_mini_driver.ReachyMiniDriver`'s
    ``api_port``), and the simulation backends that bind one
    (:meth:`~strands_robots.simulation.newton.simulation.NewtonSimEngine.open_viewer`'s
    ``port`` for the viser dashboard). A port is an index into the
    16-bit TCP port
    space, so only an ``int`` in ``[1, 65535]`` names one: ``0`` asks the kernel
    for an ephemeral port rather than naming a port, and a value outside the
    range has nothing to bind or connect to.

    A lazily-connecting transport makes the range load-bearing at the boundary
    rather than at the socket: ZMQ's ``connect`` accepts ``tcp://host:99999``
    and a WebSocket/gRPC target is only resolved on first use, so a port outside
    the range is not refused by the transport - it fails much later as an
    unreachable service, implicating the server rather than the port.

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
        return f"{context}: invalid {param}: {_refusal_repr(value)} (expected 1-65535)"
    return None


def non_negative_count_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable non-negative integer count.

    Shared domain for two families of discrete quantity whose ``0`` is a
    first-class value rather than a degenerate one:

    * The number of control steps a loop executes while an inference request is
      in flight
      (:attr:`~strands_robots.policies.base.Policy.rtc_observed_delay_steps`).
      That count is exactly ``0`` in the dominant case: a synchronous eval loop
      pauses the world during inference, so no step elapses.
    * A reproducibility seed
      (:attr:`~strands_robots.training.base.TrainSpec.seed`), where ``0`` is
      simply a seed. Its appliers disagree about everything outside this domain:
      ``torch.manual_seed`` reduces a negative seed modulo ``2**64`` (so ``-1``
      silently becomes ``2**64 - 1`` and collides with a seed a caller could
      have named), while NumPy's legacy seeder refuses a negative or a float.
    * The episode counts of the dataset-integrity gate
      (:func:`strands_robots.verify_dataset.verify_dataset`'s ``expected`` and
      ``min_frames``, and the sim facade's
      :meth:`~strands_robots.simulation.base.SimEngine.verify_dataset_episodes`).
      ``expected=0`` asks that a dataset be empty and ``min_frames=0`` skips the
      per-episode length check, so ``0`` selects a mode there rather than
      degenerating one. A value outside the domain cannot: the length check runs
      only for a threshold above zero, so a negative or non-finite one disables
      the check instead of failing it.

    Refusing ``0`` would reject the common configuration for both, which is why
    this is a separate domain rather than a caller of
    :func:`positive_count_error`.

    In every other respect it mirrors :func:`positive_count_error`: only a true
    ``int`` can be honored - an integral float raises ``TypeError`` at an action
    chunk's slice, and ``numpy.random.seed(3.0)`` raises the same class of cast
    error rather than being coerced - and ``bool`` is rejected explicitly because
    as an ``int`` subclass a bare ``value < 0`` test lets ``True`` through as a
    silent count, or a silent seed, of one.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            public method name, or the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return f"{context}: {param} must be a non-negative integer, got {_refusal_repr(value)}."
    return None


#: Highest DDS domain id whose RTPS discovery ports fit the 16-bit port space.
#:
#: RTPS derives every discovery port from the domain id (RTPS 2.2 sec. 9.6.1.1):
#: ``PB + DG * domain_id + d0`` for the SPDP multicast port and
#: ``PB + DG * domain_id + d1 + PG * participant_id`` for the unicast one. With
#: the standard values (``PB=7400``, ``DG=250``, ``d0=0``, ``d1=10``,
#: ``PG=2``) domain 232 lands on ports 65400/65410 and domain 233 lands on
#: 65650 - past the end of the port space, so there is nothing to bind. The
#: bound is the protocol's, not a policy choice.
MAX_DDS_DOMAIN_ID = 232


def dds_domain_id_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` cannot name a DDS domain.

    Shared domain for every caller-supplied ROS 2 / DDS domain id: the hardware
    ``Robot``'s ``ros2_domain``, a simulation backend's ``ros2_domain``, and the
    ``domain_id`` the rclpy telemetry bridge
    (:class:`~strands_robots.ros_telemetry.RosTelemetryBridge`) and the pure-RTPS
    bridge (:class:`~strands_robots.hardware_rtps_bridge.HardwareRtpsBridge`)
    publish on. A domain id indexes the RTPS port map, so only an ``int`` in
    ``[0, MAX_DDS_DOMAIN_ID]`` names one - see :data:`MAX_DDS_DOMAIN_ID` for the
    arithmetic that fixes the ceiling.

    The rclpy bridge makes the range load-bearing at the boundary rather than at
    the participant: it pins the domain by writing ``ROS_DOMAIN_ID`` into the
    process environment, and that write happens before ``rclpy`` is even
    imported. So a value outside the range is not refused by the transport - it
    is published to the whole process, where it outlives the call that set it
    and steers every later participant (and every subprocess that inherits the
    environment) at a domain nothing can be reached on.

    It lives here rather than beside one of its callers for the same reason
    :func:`tcp_port_error` does: those callers sit in different layers
    (:mod:`strands_robots.hardware_robot`, :mod:`strands_robots.simulation` and
    the two bridge modules) and the accepted domain must not diverge between
    them - the same domain id cannot be refused by the rclpy bridge and accepted
    by the RTPS one, when the two exist to advertise the same topics.

    ``bool`` is rejected explicitly. It is an ``int`` subclass, so a bare
    ``0 <= value <= 232`` test lets ``True`` through as a silent domain 1 and
    ``False`` through as domain 0 - a domain the caller never named.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it,
            usually the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_DDS_DOMAIN_ID:
        return f"{context}: invalid {param}: {_refusal_repr(value)} (expected 0-{MAX_DDS_DOMAIN_ID})"
    return None


MAX_ZMQ_TIMEOUT_MS = 2**31 - 1


def coerce_zmq_timeout_ms(method: str, param_name: str, value: Any) -> tuple[int | None, str | None]:
    """Read ``value`` as a ZMQ send/receive timeout in milliseconds.

    Shared domain for the ``timeout_ms`` of both ZMQ REQ inference clients -
    :class:`~strands_robots.policies.groot.client.Gr00tInferenceClient` and
    :class:`~strands_robots.policies.moveit2.client.MoveIt2InferenceClient` -
    which each hand it to ``setsockopt(RCVTIMEO)`` and ``setsockopt(SNDTIMEO)``
    on the socket they dial their sidecar with. It is the third remote-inference
    transport, and it is spelled differently from the WebSocket and gRPC pair's
    ``connect_timeout`` / ``request_timeout`` (#1984), which is why it was
    missed when those were settled: same concern, different parameter name.

    Returns the value **coerced to** ``int``, which is load-bearing rather than
    tidiness. ``setsockopt`` takes a C ``int`` and refuses every other spelling
    of the same budget, so ``15000.0`` read out of a JSON config and
    ``np.int64(15000)`` read out of a config array both raise
    ``TypeError: expected int`` from inside ``pyzmq`` - naming no parameter -
    even though each names a perfectly usable budget. The sibling transports
    accept those spellings, so coercing here is what keeps one budget from
    being usable on two transports and unusable on the third.

    Only a positive whole number can be honored, and the floor is where the
    damage is. A ``0`` timeout is ZMQ's "return immediately" spelling, so every
    request raises ``zmq.Again`` regardless of whether a sidecar is listening -
    and both clients' ``ping()`` catch every exception and return ``False``, so
    a running, reachable server is reported as unreachable with the reason at
    ``logger.debug`` only. ``False`` is the same value arriving by a different
    route, and ``True`` is a silent 1 ms budget whose verdict depends on how
    long the peer takes to answer, so it fails on one client and passes on the
    other against the same sidecar.

    **The ceiling is the transport's, not a policy choice.** ``RCVTIMEO`` is
    stored as a C ``int``, so :data:`MAX_ZMQ_TIMEOUT_MS` - ``2**31 - 1`` ms,
    close to 24.9 days - is the largest budget ZMQ will accept; one millisecond
    more raises ``OverflowError: value too large to convert to int``. This is
    why :func:`positive_whole_number_error` cannot be applied on its own: that
    domain accepts ``2**31`` as a positive whole number inside the float64
    range, and the transport then refuses it with a message naming nothing.

    ``-1`` is refused, and it is the one value that deserves the reason stated.
    ZMQ documents it as "block forever", so unlike the ``inf`` of the sibling
    transports it *is* honored - which is what makes it dangerous rather than
    merely useless. It reinstates on the request path exactly the hang that
    ``LINGER = 0``, set two lines below in the same ``_init_socket``, exists to
    prevent on teardown: a request to an unreachable sidecar never returns, and
    ``ping()`` - whose whole contract is to answer ``True`` or ``False`` about
    connectivity - can then never answer at all. Neither client documents ``-1``
    as a spelling, and an unbounded wait on a robot control path wants its own
    parameter and its own decision rather than arriving as a negative
    millisecond count. A premise test pins ZMQ's treatment of it, so this
    reopens loudly if that changes.

    Args:
        method: Message prefix identifying the surface that received the value,
            usually the class name for a constructor parameter.
        param_name: The parameter name it came from, used in the message.
        value: The caller-supplied value.

    Returns:
        ``(timeout_ms, None)`` with ``timeout_ms`` an ``int`` when the value is
        usable, or ``(None, reason)`` when it is not.
    """
    if (reason := positive_whole_number_error(value, param_name, method)) is not None:
        return None, reason
    # Read once, and compare the number this function produced rather than the
    # caller's value a second time. The first spelling of this validated with
    # the guard and then read ``value`` twice more - ``float(value)`` for the
    # range and ``int(value)`` for the result - which is the escape #1906 closed
    # for the vector guards: independent reads are not obliged to agree, so the
    # magnitude a refusal quoted need not have been the magnitude the ceiling
    # examined, and a ``numbers.Real`` whose second ``__float__`` refuses raised
    # straight out of a function whose contract is to answer with text. This
    # module carries that as a scanned invariant - no function in it may reach a
    # ``float()`` that no ``try`` protects - and the second read was exactly such
    # a conversion, so this is the invariant's verdict and not a style
    # preference.
    #
    # ``int`` is exact for every value that reaches here and needs no ``try``:
    # the guard established a ``numbers.Real`` whose ``float()`` succeeded, is
    # finite and is integral, so no rounding is possible, and the arbitrary
    # precision of ``int`` means an integral ``1e300`` converts rather than
    # overflowing and is then refused by the ceiling below.
    timeout_ms = int(value)
    if timeout_ms > MAX_ZMQ_TIMEOUT_MS:
        return None, (
            f"{method}: {param_name} must be at most {MAX_ZMQ_TIMEOUT_MS} ms "
            f"(the largest send/receive timeout ZMQ can store), got {_refusal_repr(value)}."
        )
    return timeout_ms, None


def _read_name_list(value: object, param: str, context: str) -> tuple[list[Any], str | None]:
    """Read ``value`` into a list once, or return why the read could not finish.

    The single read every verdict in :func:`name_list_error` is then computed
    from. It exists because that guard used to read the caller's value on each
    branch that needed it - the entry walk, the duplicate walk, and the
    duplicate message's own ``list(value)`` - and none of those reads was
    guarded. An acceptable value was read twice and a repeated one three times,
    so a value that answers a read at all could break the guard on any of them.

    Reading once is not only about the escape. The reads were independent, so
    nothing required them to agree, and a value whose contents differ between
    two of them was refused on the strength of a list no check had examined: a
    two-entry ``Sequence`` yielding ``["top", "wrist"]`` and then ``["top",
    "top"]`` cleared the per-entry domain checks against the first and was
    refused as a repeat against the second, in a message rendering a third. One
    read makes the verdict and the text it quotes the same list by construction.

    The read is still element by element rather than ``list(value)``, which
    matters for the same reason it does in :func:`finite_vector_error`: a
    materialising call raises before any element has been examined, so its
    verdict could not name how far the read got. Here the whole value is
    collected either way - a repeat cannot be ruled out without reading every
    entry, so there is no verdict this guard could reach by stopping early -
    but the *index* is what distinguishes a value that produced two good names
    from one that produced none, and that is worth keeping.

    Both stems are the ones :func:`finite_vector_error` already uses, with this
    guard's own unquoted ``{param}[{i}]`` index style and a remedy worded for
    names rather than numbers. A read that never began is reported without an
    index, because there is no element to name; the index is the measurement.

    The exception is rendered by :func:`_describe_failed_read`, which is also what
    a refusal that could only not *quote* something reports (#1903), so the two
    cannot describe one failure two ways.

    Args:
        value: The caller-supplied value, already known to be a
            :class:`~collections.abc.Sequence` by the check above the call.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it.

    Returns:
        The entries read and ``None``, or the entries read so far and the
        message naming why the read stopped.
    """
    try:
        elements = iter(value)  # type: ignore[call-overload]
    except Exception as exc:
        return [], (
            f"{context}: {param} could not be iterated: "
            f"{_describe_failed_read(exc)} (got {_refusal_container_repr(value)}). "
            f"Pass a list or tuple of names."
        )
    entries: list[Any] = []
    while True:
        # ``next()`` is called explicitly because a ``for`` cannot guard the call
        # it makes, so an exception raised while *producing* an entry would escape
        # this guard exactly as one from ``__iter__`` would. ``StopIteration`` is
        # the read finishing normally; an empty value is accepted as before,
        # emptiness meaning "not supplied" to every caller of this function.
        try:
            entry = next(elements)
        except StopIteration:
            return entries, None
        except Exception as exc:
            return entries, (
                f"{context}: {param}[{len(entries)}] could not be read: "
                f"{_describe_failed_read(exc)} (got {_refusal_container_repr(value)}). "
                f"Pass a list or tuple of names."
            )
        entries.append(entry)


def name_list_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable list of distinct key names.

    Shared domain for every parameter that carries an ordered list of KEY
    NAMES: the LeRobot ``image_keys`` (model VISUAL feature keys to declare on
    the config), the VERA ``image_keys`` (observation camera keys to
    width-concat into one frame), the simulation ``cameras`` subset accepted
    by ``render_all``, the two plain-MP4 recorders and every backend's
    ``start_recording``, and the ``robot_state_keys`` accepted by every
    provider's :meth:`~strands_robots.policies.base.Policy.set_robot_state_keys`
    (the ordered joint/motor names a policy emits as its action-dict keys), and
    the ``camera_keys`` / ``joint_names`` / ``action_names`` that
    :meth:`~strands_robots.dataset_recorder.DatasetRecorder.create` declares as
    the recorded dataset's column names.
    They name different vocabularies, but the shape contract is identical -
    several distinct non-blank names, in the order the caller wants them - and
    every consumer reaches the same failure when it is not met, so the rule
    lives here rather than beside any one of them.

    On the ``robot_state_keys`` path the duplicate case is the dict collapse
    above, reached twice over: the emitted action dict is keyed by these names,
    so a three-entry list with one repeat emits two commands, and the
    ``lerobot_async`` hardware-feature map declares fewer columns than the
    action aligner is handed. Note that the two providers resolving these names
    by membership rather than by position (WBC, MotionBricks) deliberately
    tolerate a repeat - it resolves to its first occurrence - so they are not
    callers of this function.

    The mistake this exists for is a single name passed as a bare string.
    ``str`` is iterable, so ``list("wrist")`` yields ``['w', 'r', 'i', 's', 't']``
    - five names the caller never wrote, one per character. Nothing downstream
    can tell that apart from a deliberate five-entry list, so it is accepted and
    the consequence surfaces far from the call: a model built declaring
    per-character features, a ``KeyError: 'w'`` raised mid-rollout when the
    frame for a one-letter camera is looked up, or a recording refused as five
    unknown cameras rather than as one mis-typed parameter.

    A ``Mapping`` is refused for the same reason in the other direction: it is
    iterable over its keys, so its values would be silently discarded.

    A repeated name is refused because it cannot be honored as written, and the
    consumers disagree on which way it fails. A duplicate collapses where the
    name keys a dict - the LeRobot feature map, or a dataset schema, which then
    declares fewer columns than asked for (two ``camera_keys`` entries naming one
    camera declare a single camera column) - and doubles where each entry drives
    its own unit of work: VERA concatenates one panel per entry, so the frame
    the model sees is twice as wide; ``render_all`` renders the same view twice;
    and a plain-MP4 recorder opens a second encoder on the one output path, so
    the same camera is rendered and appended twice per capture tick while the
    artifact ledger reports two files where one exists.

    Only a :class:`~collections.abc.Sequence` is accepted, which excludes
    one-shot iterators. That matters on the LeRobot path, where the value is
    read twice - once by the pre-flight check and again at load - so a generator
    exhausted by the first read would present as empty to the second. That is a
    statement about the *consumers*: this function itself reads the value once,
    through :func:`_read_name_list`, and a read that cannot finish is answered
    with a message rather than raising out of the guard.

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
        # One read builds the whole consequence clause. ``bytes.decode`` is
        # overridable and a ``str`` subclass owns its own ``__iter__`` and
        # ``__len__``, so producing the characters and counting them are both
        # reads of the caller's value - and taking the count from the same read
        # that produced the characters is #1897's property applied to a message,
        # rather than a second read the first is not obliged to agree with.
        shown: Any = value
        characters: list[Any] | None = None
        unreadable: str | None = None
        try:
            shown = value.decode(errors="replace") if isinstance(value, bytes) else value
        except Exception as exc:
            unreadable = _describe_failed_read(exc)
        else:
            characters, unreadable = _read_to_quote(shown)
        if characters is None:
            consequence = (
                f"A string is iterable per character, so this would be read as one name per "
                f"character; its own characters could not be read to quote them here ({unreadable})."
            )
        else:
            consequence = (
                f"A string is iterable per character, so this would be read as "
                f"{_refusal_container_repr(characters[:6])}{' ...' if len(characters) > 6 else ''} "
                f"({len(characters)} name(s))."
            )
        return (
            f"{context}: {param} must be a list of names, not a single string, "
            f"got {_refusal_container_repr(value)}. {consequence} "
            f"Wrap it in a list: [{_refusal_repr(shown)}]."
        )
    if isinstance(value, Mapping):
        # The verdict is not in doubt on this branch, so a key read that fails
        # degrades the remedy and leaves the verdict standing (#1903).
        names, unquotable = _read_to_quote(value)
        remedy = (
            f"pass the names as a list; its own keys could not be read to quote them here ({unquotable})."
            if names is None
            else f"pass the names as a list: {_refusal_container_repr(names)}."
        )
        return (
            f"{context}: {param} must be a list of names, not a mapping, "
            f"got {_refusal_container_repr(value)}. "
            f"A mapping is iterable over its keys, so its values would be discarded - {remedy}"
        )
    if not isinstance(value, Sequence):
        return (
            f"{context}: {param} must be a list of names, got {type(value).__name__} "
            f"({_refusal_container_repr(value)}). Pass a list or tuple; a one-shot iterator cannot be used "
            f"because the value is read more than once."
        )
    # One read, and every verdict below is about the list it produced. A
    # ``Sequence`` is not obliged to answer two reads the same way, so the
    # per-entry checks and the duplicate check have to see the same entries for
    # their verdicts to be about the same value - see :func:`_read_name_list`.
    entries, unread = _read_name_list(value, param, context)
    if unread is not None:
        return unread
    for i, entry in enumerate(entries):
        if not isinstance(entry, str):
            return f"{context}: {param}[{i}] must be a name (str), got {type(entry).__name__} ({_refusal_repr(entry)})."
        if not entry.strip():
            return f"{context}: {param}[{i}] must be a non-blank name, got {_refusal_repr(entry)}."
    seen: set[str] = set()
    repeated: set[str] = set()
    for entry in entries:
        if entry in seen:
            repeated.add(entry)
        seen.add(entry)
    if repeated:
        return (
            f"{context}: {param} must not repeat a name, got {_refusal_container_repr(entries)} "
            f"({_refusal_container_repr(sorted(repeated))} appears more than once)."
        )
    return None


#: Why a boolean is refused as a vector component. Worded for what a vector
#: actually carries - a coordinate, a geom extent, a friction coefficient, a
#: colour channel - rather than the radians / rad/s / newtons of the joint
#: writers, whose reason is stated separately beside them. Lives here because
#: every surface that coerces a vector component shares this one answer.
BOOLEAN_VECTOR_REASON = (
    "float(True) is 1.0, so a boolean would silently write 1.0 where a "
    "coordinate, extent or colour channel belongs, and the call would report "
    "success. Pass the component as a number."
)


def _read_finite_vector(method: str, param_name: str, vec: Any) -> tuple[list[float], str | None]:
    """Read ``vec`` into floats once, or return why it is unusable.

    The single read every verdict in :func:`finite_vector_error` is computed
    from, together with the floats that read produced.

    It exists because the three coercions built on that guard validated their
    value with it and then read the value *again*, unguarded, to build their
    floats. Two consequences, and the second is the one that reaches a stated
    contract. A value that answers the checked read and refuses the next one
    raised straight out of the coercion (#1906) - including out of
    :func:`coerce_rgba`'s wrong-length refusal, whose whole purpose is to answer
    an unusable colour with text. And the reads were independent, so nothing
    obliged them to agree: the components a refusal quoted need not have been
    the components any check examined. Returning the read makes the verdict and
    the floats the caller keeps one list by construction.

    This is :func:`_read_name_list`'s shape (#1897) on the numeric guards. The
    read moves rather than the signature: :func:`finite_vector_error` and
    :func:`pose_vector_error` have 24 call sites across four modules that want a
    verdict and nothing else, so they stay as they were and this is what they
    are now computed from.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        vec: The caller-supplied value.

    Returns:
        The floats read and ``None``, or the floats read so far and the message
        naming why the value is unusable - the same text
        :func:`finite_vector_error` has always returned.
    """
    # Collected as the read proceeds, so every refusal below can report the
    # components read before the one that stopped it, and an accepted value can
    # be returned without a second read of it.
    floats: list[float] = []
    # ``TypeError`` is what "not iterable" means in Python and the only exception
    # a well-behaved ``__iter__`` raises, but a guard whose whole purpose is to
    # answer an unusable input with a message cannot assume good behaviour of the
    # value it was handed: any other exception escaping here would be the same
    # defect as the rendering (#1873), scalar-conversion (#1874) and
    # container-conversion (#1875) escapes, on the one path that must not raise.
    # It gets its own text because the two verdicts are not the same measurement -
    # a value whose iteration raised may well have been a list/tuple of numbers,
    # and this guard never found out, so reporting it as "must be a list/tuple of
    # numbers" would state something unmeasured (#1878).
    #
    # The iterator is bound and then iterated, rather than ``iter(vec)`` being
    # called for its exception and ``vec`` iterated again: one ``__iter__`` call
    # is what the probe is asking about, and a second one is free to answer
    # differently than the one that was checked.
    #
    # This clause answers for ``__iter__`` only, and a success here says nothing
    # about the read that follows (#1889): CPython synthesises the iterator for a
    # legacy ``__getitem__`` sequence *without* calling ``__getitem__``, and
    # ``iter()`` of a generator cannot fail at all, so a value that fails part-way
    # through its own iteration arrives past this line intact. The element loop
    # below guards the call that produces each element for that reason. A
    # materialising ``list(vec)`` would cover both in one clause and is declined
    # for a stronger reason than the memory it holds: it raises before any element
    # has been examined, so its verdict could not say how far the read got - the
    # one thing that distinguishes a part-way failure from an outright refusal.
    # ``exc`` is rendered through ``_refusal_str`` rather than interpolated: a
    # value hostile enough to raise a non-``TypeError`` from ``__iter__`` is not
    # a value whose exception is assumed to have a working ``__str__``, and that
    # is the #1873 escape reintroduced inside the fix for this one.
    try:
        elements = iter(vec)
    except TypeError:
        return floats, f"{method}: '{param_name}' must be a list/tuple of numbers, got {_refusal_container_repr(vec)}"
    except Exception as exc:
        return floats, (
            f"{method}: '{param_name}' could not be iterated: "
            f"{type(exc).__name__}: {_refusal_str(exc)} (got {_refusal_container_repr(vec)}). "
            f"Pass a list or tuple of numbers."
        )
    while True:
        # ``next()`` is called explicitly because a ``for`` cannot guard the call
        # it makes: an exception raised while *producing* an element is the
        # ``__iter__`` escape one level in, on the same path that must not raise.
        # It is reported against the element's own index, the convention every
        # other guard here that names one already uses (``{param}[{i}]``), so the
        # message states both what failed and what was read before it.
        # ``StopIteration`` is the read finishing normally; an empty ``vec`` is
        # accepted exactly as before, a component count not being this guard's
        # question.
        try:
            _elem = next(elements)
        except StopIteration:
            break
        except Exception as exc:
            return floats, (
                f"{method}: '{param_name}[{len(floats)}]' could not be read: "
                f"{type(exc).__name__}: {_refusal_str(exc)} (got {_refusal_container_repr(vec)}). "
                f"Pass a list or tuple of numbers."
            )
        # ``numbers.Real`` accepts a numpy scalar (``np.float32`` / ``np.int64``
        # are registered) and rejects a string, ``None`` or a nested list.
        # ``bool`` is an ``int`` subclass, so it would otherwise pass as a
        # silent ``1.0`` - a ``True`` coordinate placing a body 1 m out. The
        # agent-tool router already refuses a bool component, so refusing it
        # here keeps the direct API and the tool surface in step.
        if is_boolean(_elem):
            return floats, (
                f"{method}: '{param_name}' elements must be numbers, not a bool "
                f"(got {_refusal_container_repr(vec)}). {BOOLEAN_VECTOR_REASON}"
            )
        if not isinstance(_elem, numbers.Real):
            return floats, f"{method}: '{param_name}' elements must be numbers, got {_refusal_container_repr(vec)}"
        # An element past the float64 range is a *magnitude* complaint and gets
        # its own reason, exactly as the scalar guards give one (#1874). The
        # order matters: ``_beyond_float_range`` answers only ``OverflowError``,
        # so a registered ``numbers.Real`` with no working ``__float__`` falls
        # through to the not-a-number text below rather than being mis-reported
        # as out of range.
        if _beyond_float_range(_elem):
            return floats, (
                f"{method}: '{param_name}' must contain numbers within the range of a 64-bit float, "
                f"got {_refusal_container_repr(vec)}"
            )
        try:
            numeric = float(_elem)
        except Exception:
            return floats, f"{method}: '{param_name}' elements must be numbers, got {_refusal_container_repr(vec)}"
        if not math.isfinite(numeric):
            return floats, (
                f"{method}: '{param_name}' must contain finite numbers (no nan/inf), got {_refusal_container_repr(vec)}"
            )
        floats.append(numeric)
    return floats, None


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
    * A value whose iteration raises something other than ``TypeError`` -
      from ``__iter__``, or from ``__next__`` once the read is under way -
      otherwise propagates that exception out of the guard, which is the same
      contract break by a different route - the caller asked for a verdict and
      got a traceback. Both are reported, and not in the same words, because the
      two are not the same measurement: a refusing ``__iter__`` yields "could not
      be iterated", since whether it held numbers is precisely what could not be
      determined, while a read that fails part-way names the element it stopped
      at, the components before it having been read and found finite.

    A numpy real scalar per element is accepted (``np.float64`` and friends are
    registered as ``numbers.Real``), matching the "accept NumPy scalar
    components" behaviour of the other sim setters; a ``bool`` is refused
    because ``float(True)`` would silently write ``1.0`` where a coordinate,
    extent or colour channel belongs. Length is NOT checked here (size is shape-dependent, so
    its count is checked against the shape afterwards); use
    :func:`pose_vector_error` for a fixed length, or :func:`coerce_rgba` for a
    colour, whose count the rgba row it is written into defines. Returns ``None``
    when every element is a finite real number.

    The read itself is :func:`_read_finite_vector`, which returns the floats it
    built alongside this verdict; this is the verdict half, for the callers that
    want only a message. Both are one read of ``vec`` (#1906).

    Deferring the count is what makes a *readable* length part of this verdict.
    A caller holding only a message counts the components by reading ``vec``
    again, and a value with no readable length cannot be read twice - the read
    behind this verdict consumes it, so the caller's own count sees nothing. That
    is a different question from the count itself ("how many components is this?"
    versus "is that answerable at all"), and it is the one
    :func:`sequence_length` owns and cannot raise answering (#1888). Refusing it
    is the rule :func:`coerce_size_vector` - this function's own sibling over the
    same read - :func:`coerce_rgba` and :func:`_read_pose_vector` already apply,
    in the same words and for the same reason. Without it a generator of finite
    numbers passed this verdict and then reached the caller's count as an empty
    vector, so ``add_object`` reported a three-component extent as ``got 0
    (size=[])`` and ``patch_scene_mjcf`` raised ``object of type 'generator' has
    no len()`` into its envelope, naming neither the field nor the method while
    the sibling fields of that same op named both.

    The probe runs *after* the component read rather than before it, which is the
    order :func:`coerce_size_vector` uses and the reason every refusal this guard
    already gave is unchanged: a value whose ``__iter__`` raises, or whose read
    fails part-way, has no readable length either, and those verdicts say what
    actually happened instead of reporting a domain check that never ran (#1875,
    #1878).
    """
    err = _read_finite_vector(method, param_name, vec)[1]
    if err is not None:
        return err
    if sequence_length(vec) is None:
        # The components were read and were finite; what the value cannot supply
        # is a length for the caller to count. Same words as the sibling
        # coercions, because it is the same verdict about the same value.
        return f"{method}: '{param_name}' must be a list/tuple of numbers, got {_refusal_container_repr(vec)}"
    return None


def _read_pose_vector(method: str, param_name: str, vec: Any, expected_len: int) -> tuple[list[float], str | None]:
    """Read ``vec`` into ``expected_len`` floats once, or say why it is unusable.

    :func:`pose_vector_error`'s read, split out for the reason
    :func:`_read_finite_vector` is (#1906): :func:`coerce_pose_vector` validated
    the pose through the guard and then read it again to build the floats.

    The length probe is unchanged and still runs before any element is produced,
    so a wrong-length vector is still refused without reading one - it is
    :func:`sequence_length`, which answers from ``__len__`` and cannot raise.
    What this returns is therefore the *element* read, the one the coercion used
    to duplicate.

    That probe answers a *reported* length, and the read below produces the
    components, so the accepted count is checked a second time against what the
    read actually yielded (#1909). The two are independent reads and nothing
    obliges them to agree; the length gate keeps its position, because refusing a
    wrong reported length without producing an element is worth keeping, and the
    re-check is the same question asked of the components that exist.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        vec: The caller-supplied value.
        expected_len: Component count the target buffer defines.

    Returns:
        The floats read and ``None``, or the floats read so far (none, when the
        length was refused) and the message :func:`pose_vector_error` returns.
    """
    # Read through the shared probe rather than a second ``len()`` here. The
    # rule "how many components is this?" has one owner (:func:`sequence_length`)
    # precisely so it cannot be answered two ways, and this call site answered it
    # a second way: an ``except TypeError`` clause, which is the gap that owner
    # closed - a ``__len__`` refusing any other way, or returning a negative or
    # an oversized ``int`` that CPython itself refuses to convert, escaped this
    # guard and the structured contract its callers document. Both verdicts below
    # are unchanged, including the text for a value carrying no readable length.
    # A str/bytes is refused on TYPE, before its length is asked for, because it
    # carries one and the answer is meaningless: it counts characters, not
    # components. ``add_camera(target="cube")`` - a plausible call, since a camera
    # aimed at a named body is what the parameter looks like it takes - was refused
    # as ``'target' must be a 3-element vector, got 4 ('cube')``, which reports the
    # string's character count as though it were a component count and points the
    # caller at fixing the length rather than the type.
    #
    # Every string is refused either way - ``"123"`` reaches the element read and is
    # refused there, on its characters - so this changes no verdict, only which
    # question the caller is sent to fix. That is the whole point: the two refusals
    # a string currently draws are picked by its LENGTH, so the same mistake reads
    # as three unrelated problems. At ``expected_len`` 3, ``target="box"`` reports
    # ``elements must be numbers``, ``target="cube"`` reports a wrong element
    # *count*, and ``target="0.1,0.2,0.3"`` reports a count of 11. One of those
    # names the actual error and the other two describe the string's characters as
    # though they were components.
    #
    # :func:`image_keys_error` already refuses str/bytes this way on the name-list
    # path next door, for the same reason (a string is iterable per character), so
    # this is one library rule applied to the surface that was missing it.
    if isinstance(vec, str | bytes):
        return [], (
            f"{method}: '{param_name}' must be a list/tuple of {expected_len} numbers, "
            f"got {type(vec).__name__} {_refusal_container_repr(vec)}. A string carries a "
            f"length, but it counts characters rather than components, so it cannot be read "
            f"as a pose - pass the {expected_len} numbers themselves."
        )
    length = sequence_length(vec)
    if length is None:
        return [], (
            f"{method}: '{param_name}' must be a list/tuple of {expected_len} numbers, "
            f"got {_refusal_container_repr(vec)}"
        )
    if length != expected_len:
        return [], (
            f"{method}: '{param_name}' must be a {expected_len}-element vector, "
            f"got {length} ({_refusal_container_repr(vec)})"
        )
    floats, err = _read_finite_vector(method, param_name, vec)
    if err is not None:
        return floats, err
    # The gate above accepted a length the value *reported*; these are the
    # components it *produced*. A ``__len__`` of 4 over a read yielding 3 passed
    # that gate and returned a 3-component vector where a wxyz quaternion was
    # promised - reaching the bare ``ValueError`` inside the ``data.qpos``
    # assignment this guard exists to prevent, through the guard rather than
    # around it (#1909). The refusal quotes the components, since they are what
    # disagrees with the count; ``length`` is named too, because "got 3" alone
    # would not say that the value's own length claimed otherwise.
    if len(floats) != expected_len:
        return floats, (
            f"{method}: '{param_name}' must be a {expected_len}-element vector, "
            f"got {len(floats)}: {floats}. Its length reported {length}, so the "
            f"components it produced are not the vector its length promised."
        )
    return floats, None


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

    The read itself is :func:`_read_pose_vector`, for the same reason
    :func:`finite_vector_error` defers to :func:`_read_finite_vector` (#1906).
    """
    return _read_pose_vector(method, param_name, vec, expected_len)[1]


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
        its own default), ``(floats, None)`` for an acceptable vector - always
        ``expected_len`` components, whether counted from the value's length or
        from the read - or ``(None, error_message)`` for a wrong length, a
        non-numeric element or a ``nan``/``inf`` component.
    """
    if vec is None:
        return None, None
    # One read, through the guard's own: the floats returned here are the ones the
    # domain checks examined, rather than the product of a second read nothing
    # required to agree with the first (#1906).
    floats, err = _read_pose_vector(method, param_name, vec, expected_len)
    if err is not None:
        return None, err
    return floats, None


#: The component counts a 4-component RGBA row can be built from. Alpha is the
#: only component with a meaningful default (opaque), so an RGB triple can be
#: completed without inventing a colour, while any other count cannot.
RGBA_ACCEPTED_LENGTHS: tuple[int, ...] = (3, 4)

#: What those components mean, quoted in the component-count refusal.
RGBA_LAYOUT = "RGB, or RGBA with alpha"


def coerce_rgba(method: str, param_name: str, color: Any) -> tuple[list[float] | None, str | None]:
    """Validate an optional colour and normalize it to 4 RGBA components.

    Single definition of the library's colour contract, shared by every backend
    so their accepted domains cannot diverge - a colour one backend refuses is
    refused by all of them. Three components are read as RGB and completed with
    an opaque alpha, the one component a colour buffer defines a default for;
    four are read as RGBA verbatim; any other count is refused, because it can
    only be applied by fabricating the missing components or discarding the
    extra ones - either way painting a colour the caller never asked for under a
    success result.

    Membership, not truthiness: a colour is "supplied" when it is not ``None``.
    Testing the vector itself (``color or <default>``) is wrong twice over. A
    NumPy array - what colour arithmetic and any palette lookup produce - raises
    a bare ``ValueError: truth value of an array ... is ambiguous`` through the
    structured tool-result contract, and an empty vector reads as *omitted*, so
    the backend default is painted while the call reports success.

    Normalizing to a ``list[float]`` of exactly 4 keeps the accepted NumPy input
    from outliving this boundary - the colour is stored on :class:`SimObject`
    (annotated ``list[float]``) and echoed in agent-visible status text, so a
    surviving ``np.float64`` would leak ``np.float64(1.0)`` into it - and makes
    the ``color[:3]`` reads the shape builders do well-defined by construction
    rather than by the caller's discipline. Both counts above are counts of the
    components this function read, so "exactly 4" holds for any value, not only
    for one whose ``__len__`` agrees with its contents (#1909).

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        color: The caller's colour, or ``None`` when the parameter was omitted.

    Returns:
        ``(None, None)`` when ``color`` is ``None`` (omitted - the caller
        applies its own documented default), ``(rgba, None)`` with exactly 4
        finite floats, or ``(None, error_message)`` for an unusable component
        count, a non-numeric or ``bool`` component, or a ``nan``/``inf`` one.
    """
    if color is None:
        return None, None
    # Asked only as "is this a sized sequence at all", the question this probe
    # owns and cannot raise answering (#1888). It is what refuses a generator,
    # whose components a read would consume before anything could count them. The
    # component count is not taken from it - see below.
    if sequence_length(color) is None:
        return None, f"{method}: '{param_name}' must be a sequence of numbers, got {_refusal_container_repr(color)}"
    # One read: the floats quoted by the component-count refusal below are the ones
    # the domain checks examined. They used to come from a second, unguarded read,
    # so a colour that answered the checked read and refused this one raised out of
    # the branch whose purpose is to answer an unusable colour with text (#1906).
    floats, err = _read_finite_vector(method, param_name, color)
    if err is not None:
        return None, err
    # The count is the read's, not the value's ``__len__``. Those were two
    # independent reads and nothing obliged them to agree: a ``__len__`` of 4 over
    # a read yielding 3 skipped the alpha completion below and returned a
    # 3-component rgba under a success result - breaking this function's stated
    # promise of exactly 4 finite floats, and with it the ``color[:3]`` reads the
    # shape builders do - while the refusal named a count from one read beside
    # components from the other (#1909). Counting the list that is returned makes
    # the promise true by construction rather than by the two reads agreeing.
    count = len(floats)
    if count not in RGBA_ACCEPTED_LENGTHS:
        expected = " or ".join(str(n) for n in RGBA_ACCEPTED_LENGTHS)
        return None, (
            f"{method}: '{param_name}' must have exactly {expected} "
            f"component(s) ({RGBA_LAYOUT}), got {count}: {floats}. Pass every "
            f"component - a partial '{param_name}' cannot be applied "
            "without inventing the missing values."
        )
    return (floats if count == 4 else [*floats, 1.0]), None


def coerce_size_vector(method: str, param_name: str, size: Any) -> tuple[list[float] | None, str | None]:
    """Validate an optional object ``size`` and normalize it to plain floats.

    The part of the library's extent contract that does not depend on the shape:
    a ``size`` is a vector of finite real numbers, or it is omitted. The MuJoCo
    backend's ``add_object`` has composed exactly this domain with its own
    shape-aware table since its numeric inputs were hardened -
    :func:`finite_vector_error` for the components, ``_validate_size`` for the
    count and the consumed extents - and this is the half a backend with no such
    table can still apply, so the same value cannot be usable on one backend and
    unusable on another.

    Membership, not truthiness: a ``size`` is "supplied" when it is not ``None``.
    Testing the vector itself (``size or <default>``) is wrong twice over. A
    NumPy array - what any extent arithmetic or a randomization draw produces -
    raises a bare ``ValueError: truth value of an array ... is ambiguous``
    through the structured tool-result contract, and an empty vector is falsy, so
    it reads as *omitted* and the backend default extent is applied while the
    call reports success.

    An empty vector is therefore refused rather than read as an omission. That is
    a statement about zero components only, not about how many a shape needs: no
    shape can be built from no extents at all, so the refusal holds whatever the
    per-shape count turns out to be.

    Normalizing to a ``list[float]`` keeps the accepted NumPy input from
    outliving this boundary - the extent is stored on :class:`SimObject`
    (annotated ``list[float]``) and echoed in agent-visible status text, so a
    surviving ``np.float64`` would leak ``np.float64(0.05)`` into it.

    What this deliberately does NOT decide is every axis whose answer depends on
    the shape: the component **count** each shape requires, whether a short
    vector may be completed from defaults, and whether a component must be
    positive (MuJoCo bounds only the components the shape actually consumes, so a
    cylinder may legitimately pass ``size[1] == 0``). Those three differ per
    backend today and need one contract decision rather than a helper default;
    #1858 tracks them.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        size: The caller's extent vector, or ``None`` when it was omitted.

    Returns:
        ``(None, None)`` when ``size`` is ``None`` (omitted - the caller applies
        its own documented default), ``(floats, None)`` for an acceptable vector,
        or ``(None, error_message)`` for a non-numeric, ``bool`` or
        ``nan``/``inf`` component, a value that is not a vector at all, or an
        empty vector.
    """
    if size is None:
        return None, None
    # Component classes first, so a value that is not a vector at all is refused
    # in the SAME words the MuJoCo backend already uses for it - one verdict
    # should not have two spellings across backends. The empty-vector refusal is
    # this helper's own, because MuJoCo reaches that case through its per-shape
    # count instead and so states a count the shape needs rather than the
    # omission the caller probably meant.
    floats, err = _read_finite_vector(method, param_name, size)
    if err is not None:
        return None, err
    if sequence_length(size) is None:
        # Reachable only for something iterable but unsized - a generator, which
        # the check above has now consumed, so there is nothing left to store.
        return None, f"{method}: '{param_name}' must be a list/tuple of numbers, got {_refusal_container_repr(size)}"
    # Empty means the read produced no component, not that ``__len__`` reported
    # zero: the two are independent reads (#1909), and it is the absence of an
    # extent to write that makes the value unusable. A value whose length reports
    # three and whose read yields nothing has no extent, and one reporting zero
    # whose read yields components has one.
    if not floats:
        return None, (
            f"{method}: '{param_name}' must have at least one component, got an empty "
            f"vector ({_refusal_container_repr(size)}). An empty '{param_name}' is a component count, not an "
            f"omission - omit '{param_name}' to take the default extent."
        )
    # The floats the component checks above examined, not a second read of the
    # caller's value (#1906).
    return floats, None


#: Camera names that a backend's render entry points resolve to the FREE camera
#: instead of looking up, by an explicit token check rather than a registry miss.
#: ``None`` and ``""`` mean "no camera was named"; ``"default"`` and ``"free"``
#: are spellings of the free view that :meth:`describe` advertises as always
#: available, so ``render(camera_name="default")`` is a documented call.
#:
#: It lives here, beside :func:`entity_name_error` and :func:`camera_fov_error`,
#: because it is read from two sides that must agree: the render entry points
#: that route it, and the ``add_camera`` guard that refuses it as a *name*. Those
#: two lived as eleven separate copies of the same tuple literal across
#: ``mujoco/rendering.py``, ``mujoco/simulation.py``, ``newton/simulation.py`` and
#: ``base.py`` - one of them written in a different order - and the MuJoCo
#: ``add_camera`` had the set in a comment but not in code, which is exactly the
#: drift that made a reserved name accepted there while Newton refused it.
FREE_CAMERA_TOKENS: Final[tuple[str | None, ...]] = (None, "", "default", "free")


def reserved_camera_name_error(method: str, param_name: str, name: Any) -> str | None:
    """Return an error message if ``name`` is a free-camera routing token.

    The creation-site counterpart to :data:`FREE_CAMERA_TOKENS`. A backend whose
    ``render`` / ``render_depth`` / ``get_frame`` select the free camera for a
    token cannot also let ``add_camera`` *claim* that token: the camera is
    registered, compiled into the scene and advertised by ``list_cameras``, and
    every render of it silently returns the free view instead. Nothing reports an
    error at any point, so the caller reads a success and a plausible frame.

    Measured on MuJoCo 3.11.0, one ``create_world`` then
    ``add_camera("free", position=[8, 8, 8], target=[0, 0, 0])``:

    * ``status="success"``, text ``Camera 'free' added at [8.0, 8.0, 8.0]``;
    * ``world.cameras`` holds ``'free'`` and ``mj_name2id(model, CAMERA, "free")``
      resolves it, so the camera really is in the compiled model;
    * ``list_cameras()`` answers ``['default', 'free']``, offering the name as
      renderable;
    * and ``render(camera_name="free")`` takes the ``cam_id = -1`` branch with
      label ``"free (default)"`` - the camera at ``[8, 8, 8]`` is unreachable
      through the very API that created it.

    ``"default"`` reached the same end by a worse route. It is refused there only
    as a *duplicate*, because ``create_world`` registers the built-in free view
    under that name, and the refusal prescribes a remedy that completes the
    defect: ``remove_camera("default")`` succeeds, the following
    ``add_camera("default", ...)`` succeeds, and the scene is left with an
    unreachable camera where the advertised free-view alias used to be.

    This guard is why the domain is stated as a name rule rather than left to the
    duplicate-name test: a duplicate is a fact about what is registered, and a
    reserved token is a fact about what can be addressed.

    Only a backend that *routes* the tokens applies it. The Isaac backend's
    ``get_frame`` looks its camera up in ``self._cameras`` directly with no token
    check, so ``"default"`` there is an ordinary addressable name - and is that
    backend's documented signature default - which is coherent and must not be
    broken by this rule.

    Returns ``None`` for every name that is not a routing token, including a
    value that is not a string at all: an unaddressable name is
    :func:`entity_name_error`'s domain, and that guard runs first at every call
    site, so this one is only ever reached with a genuine ``str``.
    """
    if not isinstance(name, str):
        return None
    if name not in FREE_CAMERA_TOKENS:
        return None
    # Through the shared renderer like every other guard here, even though this
    # one has narrowed to ``str`` and could interpolate safely: a ``str``
    # subclass owes its ``__repr__`` nothing, and the rule that no guard renders
    # a caller value directly is worth more than the exception would save.
    rendered = _refusal_repr(name)
    return (
        f"{method}: {rendered} is reserved; pick a distinct camera name. "
        f"render/get_frame resolve {param_name}={rendered} to the free camera by an "
        f"explicit token check, so a camera created under it could never be rendered from."
    )


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

    Nothing raises out of here. This guard is the one member of the scalar
    numeric family that needed no new text to make that true: a magnitude past
    the float64 range - which used to raise ``OverflowError`` out of the
    ``float()`` below - exceeds 180 degrees by three hundred orders of
    magnitude, so the interval message it already had is a true statement about
    such a value. Having a bounded interval to appeal to is what distinguishes
    it from :func:`positive_finite_number_error` and :func:`finite_number_error`,
    whose domains are unbounded above and so have no such reason available.
    """

    def not_a_number() -> str:
        return f"{method}: '{param_name}' must be a finite number in degrees, got {_refusal_repr(value)}."

    def outside_interval() -> str:
        return f"{method}: '{param_name}' must be in the open interval (0, 180) degrees, got {_refusal_str(value)}."

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return not_a_number()
    if _beyond_float_range(value):
        # This guard is the one member of the family that needs no new text. A
        # magnitude past the float64 range exceeds 180 by three hundred orders
        # of magnitude, so "outside the open interval" is already a true
        # statement about it - and no comparison is needed to establish that,
        # which is why the overflow itself is the whole test.
        return outside_interval()
    try:
        numeric = float(value)
    except Exception:
        return not_a_number()
    if not math.isfinite(numeric):
        return not_a_number()
    if not (0.0 < numeric < 180.0):
        return outside_interval()
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
            f"{method}: '{param_name}' must be a non-empty string, got {_refusal_repr(name)} "
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
            f"{method}: '{param_name}' must not contain a NUL character, got {_refusal_repr(name)}; "
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


def validation_split_error(val_episodes: int, total_tasks: Any, context: str, *, passthrough_param: str) -> str | None:
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
        passthrough_param: Name of the caller's own raw-flag passthrough
            parameter, interpolated into the remedy. Required rather than
            defaulted because the surfaces disagree: the ``lerobot_train`` tool
            spells it ``extra_flags`` while :class:`TrainSpec` (and the
            ``train_policy`` tool) spell it ``extra``, so a default would name a
            keyword one of them does not accept - the reader would apply the
            remedy verbatim and get a ``TypeError``.

    Returns:
        The error text, or None when the count can be honored exactly.
    """
    if not isinstance(total_tasks, int) or isinstance(total_tasks, bool) or total_tasks <= 1:
        return None
    return (
        f"{context}: val_episodes={val_episodes} cannot be reserved exactly on a "
        f"dataset with {_refusal_str(total_tasks)} tasks. A validation split is a per-task "
        "fraction in lerobot (it holds out ceil(episodes_in_task * eval_split) "
        "from every task), so a single global count is not expressible: the "
        "ceiling would be applied once per task. Pass the fraction directly, "
        f"e.g. {passthrough_param}={{'dataset.eval_split': 0.1, 'eval_steps': 1000}}, "
        "and the split will hold out a tenth of each task."
    )


def boolean_flag_error(value: Any, param: str, context: str) -> str | None:
    """Return an error message unless *value* is a python or numpy boolean.

    The domain for a flag whose two values select two *postures* rather than
    scaling a quantity: an IoT policy that grants a capability or withholds it
    (:func:`~strands_robots.mesh.iot.provision.provision_robot`'s
    ``allow_estop_publish``), a confirmation gate in front of a destructive
    action, or a preview mode
    (:func:`~strands_robots.mesh.iot.bootstrap.bootstrap_account`'s ``confirm``,
    ``dry_run`` and ``force_update``).

    Reading such a flag by truthiness is what this refuses. Every non-empty
    string is truthy, so ``"false"``, ``"no"``, ``"off"`` and ``"0"`` - the
    spellings an operator reaches for when opting out - select the *permissive*
    posture: a security opt-out that fails open, and a confirmation gate that
    confirms. A non-zero number and ``math.nan`` read the same way, and ``None``
    or ``[]`` silently take the other branch without ever being a declared
    spelling of it.

    There is no vocabulary to parse as a fallback either. A flag arrives already
    typed, unlike an environment variable whose only shape is a string, so the
    honest answer is to check it. Parsing would only move which spellings invert:
    ``"on"``, ``"enabled"`` and ``"y"`` are absent from every such vocabulary in
    this package, and each would then resolve to the *restrictive* posture while
    reading as an opt-in.

    It lives here rather than beside one of its callers because the flags sit in
    more than one module, and the accepted domain must not diverge: a spelling
    one provisioning entry point refuses cannot be honoured by the next. It is
    also the inverse of the numeric domains above - those reject a boolean
    through :func:`is_boolean` because ``bool`` is an ``int`` subclass that would
    pass as a silent ``1``, while this one requires the boolean they turn away.

    Distinct from
    :func:`~strands_robots.device_connect.resolve_allow_insecure`, which answers
    a related question and is not a caller of this: it resolves one setting from
    two sources, so its domain is ``bool | None`` - ``None`` is the documented
    spelling for *fall through to the environment variable* - and its refusal
    names that variable's own opt-in vocabulary. This domain has one source and
    no such sentinel, so it refuses ``None`` along with every other non-boolean.

    Args:
        value: The flag as supplied.
        param: Parameter name, for the message.
        context: Caller label the message is prefixed with.

    Returns:
        The error text, or None when *value* is a boolean and can be honoured.
    """
    if is_boolean(value):
        return None
    return (
        f"{context}: {param} must be a boolean, got {_refusal_repr(value)}. "
        "It selects a posture rather than scaling a quantity, so it is checked "
        "rather than parsed - a truthy spelling of off, such as 'false', would "
        "otherwise select the opposite posture from the one it reads as."
    )


def partial_construction_repr(obj: object) -> str:
    """Describe an object whose ``__init__`` did not finish, naming no attribute.

    ``repr`` is what a traceback, a debugger and a failing assertion render, so
    it must not be the thing that hides a failure. A class that validates its
    own arguments raises before it assigns the attributes its ``__repr__``
    reads, and the raising frame keeps that half-built instance alive - so
    rendering it reports ``[AttributeError ... raised in repr()]`` naming an
    attribute that has nothing to do with the refusal under investigation.
    Returning this instead reports the lifecycle fact that *is* relevant, and
    deliberately names no attribute so nobody is sent chasing one.

    The wording lives here rather than beside any one caller because those
    callers sit in different layers - the ROS 2 / rosbridge / RTPS transport
    bridges, the teleop input streams, the dataset recorder, the peer registry
    and the simulation engines - and the phrase a reader learns to recognise in
    a traceback must not diverge between them.

    Args:
        obj: The partially constructed object being rendered.

    Returns:
        ``"<ClassName>(partially constructed, id=0x...)"``.
    """
    return f"{type(obj).__name__}(partially constructed, id=0x{id(obj):x})"
