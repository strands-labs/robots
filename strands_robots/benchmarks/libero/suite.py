"""Bulk benchmark registration for LIBERO task suites.

LIBERO ships ~130 tasks split across five suites:

* ``libero-spatial`` - 10 tasks, same objects / different spatial goals
* ``libero-object`` - 10 tasks, different objects / same goal structure
* ``libero-goal`` - 10 tasks, same objects / different goals
* ``libero-10`` - 10 tasks, "short-horizon diverse"
* ``libero-90`` - 90 tasks, "long-horizon diverse"

Rather than have agents call :func:`register_benchmark` 130x manually,
:func:`load_libero_suite` walks the upstream package's BDDL directory and
registers every task under a predictable ``libero-<suite>-<task>`` key.
Tasks that fail to parse are logged and skipped - a single malformed BDDL
file should never block the whole suite from loading.

Layout discovery
----------------

The ``libero`` pip package historically keeps BDDL files under
``<libero_root>/libero/bddl_files/<suite_name>/*.bddl`` and scene MJCFs
alongside the benchmark code. Because the exact subpath has drifted
between releases, :func:`load_libero_suite` accepts an explicit
``bddl_dir=`` override and falls back to probing a handful of standard
locations when not given. The scene resolver behaves similarly via
``scene_dir=``.

Init states
-----------

LIBERO ships task-specific *init states* alongside its benchmark suites -
each task has 50 sampled starting configurations of the canonical "ready"
pose plus per-object free-joint placements. :func:`load_libero_suite`
lazily imports ``libero.libero.benchmark`` and pulls
``ts.get_task_init_states(task_id)`` for every registered task, then
forwards as ``init_states=`` into :meth:`LiberoAdapter.from_file`. Without
this the robot starts at ``qpos=0`` (joint-default "stretched flat"
pose), the policy issues actions calibrated for the canonical "ready"
pose against a totally different body configuration, and the success
rate collapses to 0 (#168 bug I). When ``libero`` isn't
importable (e.g. minimal CI), init_states loading no-ops and the
adapter falls back to its snapshot-and-restore branch.

Callers who already have the BDDL files on disk (e.g. vendored into their
repo) do **not** need the ``libero`` package installed - just pass
``bddl_dir=`` and the function registers from there. Init-state loading
also short-circuits in that case - the
``benchmark.get_benchmark_dict()[suite]()`` call requires the upstream
package's package-data path resolution.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.benchmarks.libero.adapter import LiberoAdapter
from strands_robots.benchmarks.libero.bddl_parser import BDDLParseError
from strands_robots.simulation.benchmark import register_benchmark
from strands_robots.utils import require_optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Canonical suite names - these are the keys agents will use. The upstream
# directory names sometimes differ (snake_case vs kebab-case); the name
# resolver accepts both.
SUITE_NAMES = frozenset(
    {
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
    }
)


def _normalise_suite_name(name: str) -> str:
    """Accept either ``libero-spatial`` or ``libero_spatial``; return the underscore form.

    Upstream LIBERO uses snake_case directory names; our benchmark registry
    uses kebab-case keys. Keep the normalisation centralised.
    """
    key = name.strip().lower().replace("-", "_")
    if not key.startswith("libero_"):
        key = f"libero_{key}"
    return key


def _candidate_bddl_dirs(libero_root: Path, suite: str) -> list[Path]:
    """Return paths to try in order. First existing one wins."""
    return [
        libero_root / "libero" / "bddl_files" / suite,
        libero_root / "libero" / "libero" / "bddl_files" / suite,
        libero_root / "bddl_files" / suite,
        libero_root / "libero" / "tasks" / suite,
        libero_root / suite,
    ]


def _resolve_libero_root() -> Path:
    """Find the filesystem root of the installed ``libero`` package.

    Lazily imports ``libero`` via :func:`require_optional` with a helpful
    install hint pointing at ``strands-robots[benchmark-libero]``.

    Handles two install layouts:

    1. Regular package — ``libero.__file__`` points at
       ``.../site-packages/libero/__init__.py``; the parent directory is
       the package root.
    2. Namespace package (PEP 420) — ``libero.__file__`` is ``None``;
       ``libero.__path__`` carries one or more directory entries (e.g.
       NVIDIA's ``setup_libero.sh`` installs LIBERO this way: a
       symlinked checkout where ``libero`` has no ``__init__.py``).
       Fall back to the first path entry's parent.
    """
    libero = require_optional(
        "libero",
        pip_install="libero",
        extra="benchmark-libero",
        purpose="LIBERO benchmark suite discovery",
    )
    libero_file = getattr(libero, "__file__", None)
    if libero_file:
        # Regular package: __file__ lives inside the package; its parent is the package root.
        return Path(libero_file).resolve().parent.parent

    # Namespace package: pull from __path__ (the canonical PEP 420 attribute).
    libero_path = getattr(libero, "__path__", None)
    if libero_path:
        # __path__ is a (namespace) iterable of directory strings; first entry is enough.
        first = next(iter(libero_path), None)
        if first:
            return Path(first).resolve().parent

    raise RuntimeError(
        "libero package has neither __file__ nor a non-empty __path__; cannot locate BDDL tasks. "
        "Reinstall with `pip install strands-robots[benchmark-libero]` or check your install layout."
    )


def _load_init_states_by_bddl(suite: str) -> dict[str, np.ndarray]:
    """Build a ``{bddl_filename: init_states_ndarray}`` map for ``suite``.

    Returns an empty dict (with a warning) when the upstream ``libero``
    package isn't importable, when the suite name isn't a recognised
    benchmark, or when any individual task fails to load its init
    states. The empty-dict fallback is intentional - a missing init-state
    file shouldn't block suite registration; the adapter will fall back
    to its snapshot-and-restore branch in
    :meth:`LiberoAdapter._apply_canonical_state`.

    Important wart: ``benchmark.get_benchmark_dict()[suite]()`` permutes
    task order per ``task_orders[task_order_index]`` for every suite
    *except* ``libero_90``. The map this function builds keys on the
    BDDL filename returned by ``ts.get_task_bddl_files()[i]``, which
    is post-permutation - so the matching to BDDL files in
    :func:`load_libero_suite` is order-independent and correct
    regardless of which permutation libero applies internally.

    State width per task varies (libero_10 alone ships tasks with
    init-state widths of 45 / 47 / 51 / 71 / 84 / 123). Each task's
    init_states ndarray has shape ``(50, 1+nq+nv)`` for that task's
    specific ``nq`` / ``nv``.

    Source-of-truth for the API:
    ``/opt/conda/lib/python3.12/site-packages/libero/libero/benchmark/__init__.py:115-165``.
    """
    try:
        from libero.libero import benchmark
    except ImportError as e:
        logger.debug(
            "LiberoAdapter: libero not importable (%s); init_states loading skipped, "
            "adapter will fall back to snapshot-and-restore",
            e,
        )
        return {}

    benchmark_dict = getattr(benchmark, "get_benchmark_dict", None)
    if benchmark_dict is None:
        logger.warning("LiberoAdapter: libero.benchmark has no get_benchmark_dict(); init_states skipped")
        return {}

    try:
        suites_map = benchmark_dict()
    except Exception as e:  # noqa: BLE001 - never fatal for suite registration
        logger.warning("LiberoAdapter: benchmark.get_benchmark_dict() raised %s; init_states skipped", e)
        return {}

    suite_factory = suites_map.get(suite)
    if suite_factory is None:
        logger.debug(
            "LiberoAdapter: %r not in benchmark.get_benchmark_dict() (%s); init_states skipped",
            suite,
            sorted(suites_map.keys()),
        )
        return {}

    try:
        ts = suite_factory()
    except Exception as e:  # noqa: BLE001 - never fatal for suite registration
        logger.warning("LiberoAdapter: instantiating benchmark %r raised %s; init_states skipped", suite, e)
        return {}

    n_tasks = int(ts.get_num_tasks())
    bddl_files: list[str]
    try:
        bddl_files = list(ts.get_task_bddl_files())
    except Exception as e:  # noqa: BLE001 - never fatal
        logger.warning("LiberoAdapter: ts.get_task_bddl_files() raised %s; init_states skipped", e)
        return {}

    out: dict[str, np.ndarray] = {}
    for task_id in range(n_tasks):
        if task_id >= len(bddl_files):
            logger.debug(
                "LiberoAdapter: task_id=%d out of range for bddl_files (len=%d); skipping",
                task_id,
                len(bddl_files),
            )
            continue
        bddl_filename = Path(bddl_files[task_id]).name  # bare filename, no dir
        try:
            states = ts.get_task_init_states(task_id)
        except Exception as e:  # noqa: BLE001 - per-task failure shouldn't block the rest
            logger.warning(
                "LiberoAdapter: get_task_init_states(%d) for suite %r raised %s; task %r skipped",
                task_id,
                suite,
                e,
                bddl_filename,
            )
            continue
        out[bddl_filename] = np.asarray(states)
    logger.debug(
        "LiberoAdapter: loaded init_states for %d/%d tasks in suite %r",
        len(out),
        n_tasks,
        suite,
    )
    return out


def load_libero_suite(
    suite_name: str,
    *,
    bddl_dir: str | Path | None = None,
    scene_dir: str | Path | None = None,
    max_steps: int | None = None,
    init_jitter: float = 0.02,
    key_prefix: str = "libero",
    load_init_states: bool = True,
) -> dict[str, LiberoAdapter]:
    """Register every task in ``suite_name`` under the benchmark registry.

    Args:
        suite_name: One of ``libero_spatial`` / ``libero_object`` /
            ``libero_goal`` / ``libero_10`` / ``libero_90``. Accepts
            ``libero-spatial`` form too.
        bddl_dir: Explicit directory containing ``*.bddl`` files. When
            omitted, tries the installed ``libero`` package layout.
        scene_dir: Root under which per-task scene MJCFs live. When
            provided, each adapter gets ``scene_path = scene_dir /
            <task>.xml`` if the file exists; otherwise scene is left as
            ``None`` and the adapter assumes the scene is already loaded.
        max_steps: Forwarded to every :class:`LiberoAdapter`.
        init_jitter: Forwarded to every :class:`LiberoAdapter`.
        key_prefix: Registry key format is ``<key_prefix>-<suite>-<task>``.
            Pass ``key_prefix=""`` for ``<suite>-<task>``.
        load_init_states: When ``True`` (default), lazily import
            ``libero.libero.benchmark`` and pull
            ``ts.get_task_init_states(task_id)`` for every registered
            task. Required for ``success_rate > 0`` against
            ``nvidia/GR00T-N1.7-LIBERO`` - without it the robot starts
            at ``qpos=0`` instead of LIBERO's canonical "ready" pose
            (#168 bug I). Set to ``False`` to disable for unit
            tests / minimal CI / when ``libero`` isn't installed (the
            loader silently no-ops in those cases anyway, but the flag
            documents intent). When ``True`` and libero loading fails,
            registration continues with init_states=None and the
            adapter falls back to snapshot-and-restore.

    Returns:
        ``{registry_name: LiberoAdapter}`` for every successfully registered
        task. Failed tasks are logged (at WARNING) and omitted.

    Raises:
        FileNotFoundError: If no BDDL directory can be located.
        ValueError: If ``suite_name`` isn't a recognised LIBERO suite.
    """
    suite = _normalise_suite_name(suite_name)
    if suite not in SUITE_NAMES:
        raise ValueError(f"Unknown LIBERO suite {suite_name!r}. Valid: {sorted(SUITE_NAMES)}")

    resolved_bddl_dir = _locate_bddl_dir(suite, bddl_dir)
    resolved_scene_dir = Path(scene_dir).expanduser().resolve() if scene_dir else None

    init_states_by_bddl: dict[str, Any] = _load_init_states_by_bddl(suite) if load_init_states else {}

    registered: dict[str, LiberoAdapter] = {}
    failures: list[tuple[str, str]] = []

    for bddl_file in sorted(resolved_bddl_dir.glob("*.bddl")):
        task_stem = bddl_file.stem
        registry_name = _format_registry_name(key_prefix, suite, task_stem)

        scene_path: str | None = None
        if resolved_scene_dir is not None:
            candidate = resolved_scene_dir / f"{task_stem}.xml"
            if candidate.exists():
                scene_path = str(candidate)

        # init_states are keyed by BDDL filename (post-permutation, see
        # _load_init_states_by_bddl docstring). May be None when the suite
        # didn't ship init states for this task or libero wasn't importable.
        task_init_states = init_states_by_bddl.get(bddl_file.name)

        try:
            adapter = LiberoAdapter.from_file(
                bddl_file,
                scene_path=scene_path,
                max_steps=max_steps,
                init_jitter=init_jitter,
                init_states=task_init_states,
            )
        except (BDDLParseError, FileNotFoundError, ValueError) as e:
            logger.warning("Skipping LIBERO task %s: %s", bddl_file.name, e)
            failures.append((str(bddl_file), str(e)))
            continue
        register_benchmark(registry_name, adapter)
        registered[registry_name] = adapter

    logger.info(
        "Registered %d LIBERO tasks from %s (skipped %d malformed, init_states=%d)",
        len(registered),
        resolved_bddl_dir,
        len(failures),
        len(init_states_by_bddl),
    )
    return registered


def _format_registry_name(prefix: str, suite: str, task: str) -> str:
    # Suite is in ``libero_spatial`` form; keys use kebab-case.
    # * With prefix: ``<prefix>-<suite-without-libero>-<task>``
    #   (``libero-spatial-pick_cube`` when prefix="libero").
    # * Without prefix: ``<suite-without-libero>-<task>``
    #   (``spatial-pick_cube``) - agents who supply their own key scheme
    #   don't want the ``libero-`` doubled in.
    suite_kebab = suite.replace("_", "-").removeprefix("libero-")
    if prefix:
        return f"{prefix}-{suite_kebab}-{task}"
    return f"{suite_kebab}-{task}"


def _locate_bddl_dir(suite: str, override: str | Path | None) -> Path:
    if override is not None:
        d = Path(override).expanduser().resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"bddl_dir does not exist or is not a directory: {d}")
        return d

    libero_root = _resolve_libero_root()
    for candidate in _candidate_bddl_dirs(libero_root, suite):
        if candidate.is_dir():
            return candidate
    tried = [str(p) for p in _candidate_bddl_dirs(libero_root, suite)]
    raise FileNotFoundError(
        f"Could not locate BDDL directory for suite {suite!r}. Tried: {tried}. Pass bddl_dir= to override."
    )


def available_suites() -> Iterable[str]:
    """Return the canonical set of LIBERO suite names - offline constant."""
    return frozenset(SUITE_NAMES)


__all__ = [
    "SUITE_NAMES",
    "available_suites",
    "load_libero_suite",
]
