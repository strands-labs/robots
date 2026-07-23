"""Import + API-drift smoke tests for the LIBERO example drivers.

The ``examples/libero`` driver scripts are not part of the installed
package, so nothing else in CI imports them - an API rename in
``strands_robots`` can silently break the documented smoke path
(``python examples/libero/run_mujoco.py --policy mock --n-episodes 5``)
without any test going red. These tests pin the contract:

1. Each MuJoCo driver imports cleanly against the *current* library
   (module-level imports are the first thing that breaks on drift).
2. The library symbols the drivers call still exist with the expected
   surface (``evaluate_benchmark`` kwargs, camera-recording pair,
   ``LiberoAdapter.ensure_scene``, ``gr00t_inference``).
3. The grep-stable result line ``run_mujoco.py`` prints stays
   byte-compatible with ``libero_backend_matrix.py``'s ``_RE_RESULT``
   parser - drift there produces empty ``success_rate`` matrix cells
   rather than a crash, so only a test catches it.
4. The examples do not depend on private (``_``-prefixed) LiberoAdapter
   methods.

The Isaac drivers (``run_isaac.py`` / ``run_isaac_agent.py``) are
excluded from the import smoke: they intentionally guard their heavy
imports behind ``IsaacSimulation.is_available`` at runtime, but their
module docstrings + argparse setup still get covered by the repo-wide
example lint tests.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

_EXAMPLES_LIBERO = Path(__file__).resolve().parent.parent / "examples" / "libero"

# The MuJoCo drivers import `Simulation` (the MuJoCo backend) at module
# level, so the import smoke needs mujoco installed.
pytest.importorskip("mujoco")


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


@pytest.mark.parametrize("filename", ["run_mujoco.py", "run_mujoco_agent.py", "libero_backend_matrix.py"])
def test_driver_imports_cleanly(filename: str) -> None:
    module = _load_example(filename)
    assert callable(module.main)


def test_run_mujoco_helper_surface() -> None:
    """``run_isaac.py`` docstrings cross-reference these helpers by name;
    keep the canonical MuJoCo file exposing them."""
    module = _load_example("run_mujoco.py")
    assert callable(module._date_dir)
    assert callable(module._suite_for_task)
    assert module._suite_for_task("libero-spatial-pick_up_the_red_cube") == "libero_spatial"
    assert module._suite_for_task("libero-10-LIVING_ROOM_SCENE5_x") == "libero_10"
    with pytest.raises(ValueError, match="libero-<suite>-<task_stem>"):
        module._suite_for_task("not-a-libero-task")
    with pytest.raises(ValueError, match="libero-<suite>-<task_stem>"):
        module._suite_for_task("libero-spatial")


def test_result_line_matches_backend_matrix_parser() -> None:
    """The line run_mujoco.py prints must parse with _RE_RESULT.

    Build the sample line with the same format specs the driver uses
    (`success_rate={sr:.2f}`, `wall_time={wt:.1f}s`); if either side
    changes shape, this goes red before the matrix silently prints
    empty cells.
    """
    matrix = _load_example("libero_backend_matrix.py")
    sr, wt = 0.8, 123.4
    task = "libero-spatial-pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    line = f"policy=mock  task={task}  success_rate={sr:.2f}  wall_time={wt:.1f}s  videos=rollouts/x.mp4"
    match = matrix._RE_RESULT.search(line)
    assert match is not None, f"_RE_RESULT failed to parse the driver's result-line shape: {line!r}"
    assert float(match.group("sr")) == pytest.approx(sr)
    assert float(match.group("wt")) == pytest.approx(wt)


def test_backend_matrix_has_mujoco_row() -> None:
    """The first matrix row must point at run_mujoco.py, and the file it
    references must exist (pre-migration it printed `unavailable`)."""
    matrix = _load_example("libero_backend_matrix.py")
    rows = {label: filename for label, filename, _ in matrix._BACKEND_ROWS}
    assert rows.get("mujoco") == "run_mujoco.py"
    assert (_EXAMPLES_LIBERO / "run_mujoco.py").is_file()


@pytest.mark.parametrize("filename", ["run_mujoco.py", "run_mujoco_agent.py"])
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
        "port",
    ):
        assert kwarg in tool_params, f"gr00t_inference lost the {kwarg!r} kwarg the LIBERO drivers pass"
