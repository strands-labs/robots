"""Import + API-drift smoke tests for the LIBERO example drivers.

The ``examples/libero`` driver scripts are not part of the installed
package, so nothing else in CI imports them - an API rename in
``strands_robots`` can silently break the documented smoke path
(``python examples/libero/run.py mujoco --policy mock --n-episodes 5``)
without any test going red. These tests pin the contract:

1. The merged driver (``run.py``) imports cleanly with no backend
   installed at all - it defers every backend import until after
   subcommand parsing - and the MuJoCo agent driver imports cleanly
   against the *current* library (module-level imports are the first
   thing that breaks on drift).
2. The library symbols the drivers call still exist with the expected
   surface (``evaluate_benchmark`` kwargs, camera-recording pair,
   ``LiberoAdapter.ensure_scene``, ``gr00t_inference``).
3. The grep-stable result line ``run.py`` prints (built by
   ``_format_result_lines``, the single formatter both subcommands
   share) stays byte-compatible with ``libero_backend_matrix.py``'s
   ``_RE_RESULT`` parser - drift there produces empty ``success_rate``
   matrix cells rather than a crash, so only a test catches it.
4. The backend subcommands scope their flags: Isaac-only flags do not
   exist on the ``mujoco`` subcommand, and the shared flag base is
   identical across the two.
5. The examples do not depend on private (``_``-prefixed) LiberoAdapter
   methods.

The Isaac agent driver (``run_isaac_agent.py``) is excluded from the
import smoke: it intentionally guards its heavy imports behind
``IsaacSimulation.is_available`` at runtime, but its module docstring +
argparse setup still get covered by the repo-wide example lint tests.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

_EXAMPLES_LIBERO = Path(__file__).resolve().parent.parent / "examples" / "libero"


def _load_example(filename: str):
    """Import an example script by path under a test-unique module name."""
    path = _EXAMPLES_LIBERO / filename
    assert path.is_file(), f"expected example driver at {path}"
    module_name = f"_example_smoke_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def test_run_imports_cleanly_without_any_backend() -> None:
    """``run.py`` must not import a simulation backend at module level.

    The subcommand design promises "import the chosen backend only
    after parsing" - a module-level ``strands_robots.simulation``
    import would make ``run.py isaac`` pay for MuJoCo (and break the
    matrix's ``skip``-row detection on hosts without it).
    """
    before = set(sys.modules)
    module = _load_example("run.py")
    assert callable(module.main)
    leaked = {m for m in set(sys.modules) - before if m.startswith(("mujoco", "omni", "isaacsim"))}
    assert not leaked, f"run.py imported backend modules at import time: {sorted(leaked)}"


@pytest.mark.parametrize("filename", ["run_mujoco_agent.py", "libero_backend_matrix.py"])
def test_driver_imports_cleanly(filename: str) -> None:
    # The MuJoCo agent driver imports `Simulation` (the MuJoCo backend)
    # at module level, so its import smoke needs mujoco installed.
    if filename == "run_mujoco_agent.py":
        pytest.importorskip("mujoco")
    module = _load_example(filename)
    assert callable(module.main)


def test_run_helper_surface() -> None:
    """Sibling agent-driver docstrings cross-reference these helpers by
    name; keep the canonical merged driver exposing them."""
    module = _load_example("run.py")
    assert callable(module._date_dir)
    assert callable(module._suite_for_task)
    assert callable(module._default_checkpoint_dir)
    assert callable(module._explain_lifecycle_failure)
    assert callable(module._configure_gr00t_image)
    assert callable(module._orchestrate_groot_server)
    assert callable(module._resolve_robot_asset)
    assert module._suite_for_task("libero-spatial-pick_up_the_red_cube") == "libero_spatial"
    assert module._suite_for_task("libero-10-LIVING_ROOM_SCENE5_x") == "libero_10"
    with pytest.raises(ValueError, match="libero-<suite>-<task_stem>"):
        module._suite_for_task("not-a-libero-task")
    with pytest.raises(ValueError, match="libero-<suite>-<task_stem>"):
        module._suite_for_task("libero-spatial")


@pytest.mark.parametrize("backend", ["mujoco", "isaac"])
def test_result_line_matches_backend_matrix_parser(backend: str) -> None:
    """The line run.py prints must parse with _RE_RESULT.

    Build the sample line with the driver's own formatter
    (``_format_result_lines`` - the exact function both subcommands
    print through); if either side changes shape, this goes red before
    the matrix silently prints empty cells.
    """
    run = _load_example("run.py")
    matrix = _load_example("libero_backend_matrix.py")
    sr, wt = 0.8, 123.4
    requested = "libero-spatial-pick_up_the_red_cube"
    resolved = "libero-spatial-pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    bench_line, result_line = run._format_result_lines(
        policy="mock",
        requested_task=requested,
        resolved_task=resolved,
        success_rate=sr,
        wall_time=wt,
        video_path="rollouts/x.mp4",
        backend=backend,
    )
    assert bench_line == f"benchmark_name={requested}"
    match = matrix._RE_RESULT.search(result_line)
    assert match is not None, f"_RE_RESULT failed to parse the driver's result-line shape: {result_line!r}"
    assert float(match.group("sr")) == pytest.approx(sr)
    assert float(match.group("wt")) == pytest.approx(wt)
    # The replay contract: task= echoes the CLI-requested task,
    # resolved_task= what actually ran, backend= the subcommand.
    assert f"task={requested}  " in result_line
    assert f"resolved_task={resolved}  " in result_line
    assert result_line.endswith(f"backend={backend}")


def test_backend_matrix_rows_invoke_run_py_subcommands() -> None:
    """The mujoco / isaac matrix rows must point at run.py with the
    matching subcommand, and the file they reference must exist."""
    matrix = _load_example("libero_backend_matrix.py")
    rows = {label: (filename, subcommand) for label, filename, subcommand, _ in matrix._BACKEND_ROWS}
    assert rows.get("mujoco") == ("run.py", ["mujoco"])
    assert rows.get("isaac-1") == ("run.py", ["isaac"])
    assert (_EXAMPLES_LIBERO / "run.py").is_file()


def test_isaac_only_flags_do_not_exist_on_the_mujoco_subcommand() -> None:
    """Subcommand flag scoping: the merge's whole point.

    Isaac-only flags must be rejected by the ``mujoco`` subcommand at
    parse time (not silently dropped), accepted by ``isaac``, and the
    shared base must parse identically on both.
    """
    run = _load_example("run.py")
    parser = run._build_parser()

    shared = ["--policy", "mock", "--n-episodes", "3", "--seed", "7", "--port", "8123"]
    mj = parser.parse_args(["mujoco", *shared])
    isc = parser.parse_args(["isaac", *shared, "--robot-usd", "/x/robot.usd", "--eef-body-name", "hand"])
    assert mj.backend == "mujoco"
    assert isc.backend == "isaac"
    for attr in ("policy", "n_episodes", "seed", "port", "task", "auto_server", "image", "deterministic"):
        assert hasattr(mj, attr) and hasattr(isc, attr), f"shared flag {attr!r} missing from a subcommand"
    assert isc.robot_usd == "/x/robot.usd"
    assert isc.eef_body_name == "hand"

    for flag in ("--robot-usd=/x.usd", "--robot-urdf=/x.urdf", "--eef-body-name=hand"):
        with pytest.raises(SystemExit):
            parser.parse_args(["mujoco", flag])
    assert not hasattr(mj, "robot_usd"), "mujoco subcommand should not carry Isaac-only namespace attrs"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("mujoco", "gr00t-libero-mujoco"), ("isaac", "gr00t-libero-isaac")],
)
def test_container_default_derives_from_the_backend_subcommand(backend: str, expected: str) -> None:
    """The ``--container`` default is computed per subcommand.

    The merge replaced two hardcoded container names with one f-string
    derived from the subparser's ``dest="backend"`` - the only
    behavioral mechanism the refactor changed. If either the f-string
    or the dest drifts, both subcommands resolve to one container name,
    which is exactly what the flag's help text promises cannot happen
    ("don't clobber each other's containers when run side-by-side").
    That failure surfaces as a GR00T lifecycle collision on a GPU host
    CI does not have, so this is the pin - it also documents that the
    two pre-merge drivers' defaults were preserved deliberately.
    """
    run = _load_example("run.py")
    args = run._build_parser().parse_args([backend])
    # Parse-time sentinel: None keeps an explicit --container
    # distinguishable from the derived default.
    assert args.container is None
    run._resolve_container_name(args)
    assert args.container == expected


def test_explicit_container_survives_default_resolution() -> None:
    """An explicit ``--container`` passes through resolution unchanged.

    The resolver guards on ``is None``, not truthiness - the half a
    naive ``args.container or f"..."`` rewrite would break.
    """
    run = _load_example("run.py")
    args = run._build_parser().parse_args(["mujoco", "--container", "my-name"])
    run._resolve_container_name(args)
    assert args.container == "my-name"


@pytest.mark.parametrize("filename", ["run.py", "run_mujoco_agent.py"])
def test_drivers_do_not_call_private_scene_generation(filename: str) -> None:
    """Public-API hygiene: the drivers must use ``LiberoAdapter.ensure_scene``
    (public), never the private ``_generate_scene_from_bddl``."""
    source = (_EXAMPLES_LIBERO / filename).read_text(encoding="utf-8")
    assert "_generate_scene_from_bddl" not in source, (
        f"{filename} references the private LiberoAdapter._generate_scene_from_bddl; "
        "use the public ensure_scene() instead."
    )
    assert "ensure_scene" in source


def test_library_surface_the_drivers_depend_on() -> None:
    """Pin the exact library API the drivers call, so a rename fails here
    with a readable message instead of at example runtime."""
    pytest.importorskip("mujoco")
    from strands_robots.benchmarks.libero import LiberoAdapter, load_libero_suite
    from strands_robots.simulation import Simulation, get_benchmark

    # Import the tool function from its submodule, not via the package's
    # lazy `strands_robots.tools.__getattr__`: when another test has already
    # imported the `strands_robots.tools.gr00t_inference` *module*, Python
    # binds the module object as the package attribute, shadowing the lazy
    # function resolution - so the package-level import is order-dependent
    # under the full test suite. (The drivers themselves run in fresh
    # interpreters, where the lazy path deterministically yields the
    # function.)
    from strands_robots.tools.gr00t_inference import gr00t_inference

    assert callable(load_libero_suite)
    assert callable(get_benchmark)
    assert callable(gr00t_inference)

    # `_resolve_task` forwards adapter_kwargs (the Isaac EEF state-source
    # configuration, #1802) through load_libero_suite.
    assert "adapter_kwargs" in inspect.signature(load_libero_suite).parameters

    # LiberoAdapter public pre-warm surface used by the scene pre-warm block.
    assert callable(LiberoAdapter.ensure_scene)
    assert callable(LiberoAdapter.prewarm)

    # Simulation methods the drivers call.
    for method in (
        "create_world",
        "add_robot",
        "load_scene",
        "list_robots",
        "start_cameras_recording",
        "stop_cameras_recording",
        "evaluate_benchmark",
        "destroy",
    ):
        assert callable(getattr(Simulation, method)), f"Simulation.{method} missing"

    # evaluate_benchmark kwargs the drivers pass.
    params = inspect.signature(Simulation.evaluate_benchmark).parameters
    for kwarg in ("benchmark_name", "n_episodes", "seed", "policy_provider", "policy_config", "robot_name"):
        assert kwarg in params, f"evaluate_benchmark lost the {kwarg!r} kwarg the LIBERO drivers pass"

    # start_cameras_recording kwargs the drivers pass.
    rec_params = inspect.signature(Simulation.start_cameras_recording).parameters
    for kwarg in ("cameras", "output_dir", "name"):
        assert kwarg in rec_params, f"start_cameras_recording lost the {kwarg!r} kwarg the LIBERO drivers pass"

    # gr00t_inference lifecycle kwargs the drivers pass. The @tool
    # decorator preserves the underlying signature via functools.wraps;
    # fall back to the raw function if not.
    tool_fn = inspect.unwrap(gr00t_inference)
    tool_params = inspect.signature(tool_fn).parameters
    for kwarg in (
        "action",
        "lifecycle",
        "hf_repo",
        "hf_subfolder",
        "hf_local_dir",
        "container_name",
        "hf_token",
        "checkpoint_path",
        "embodiment_tag",
        "protocol",
        "use_sim_policy_wrapper",
        "deterministic",
        "port",
    ):
        assert kwarg in tool_params, f"gr00t_inference lost the {kwarg!r} kwarg the LIBERO drivers pass"
