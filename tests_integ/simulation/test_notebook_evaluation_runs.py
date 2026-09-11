"""End-to-end: notebook 7's cells run, and the fields it prints are the ones it gets.

``tests/test_notebook_evaluation_spec.py`` grades the static half of
``examples/notebooks/07_evaluate_a_policy.ipynb`` - that the authored spec is pure
data and compiles, and that the two broken specs are still refused. All of that
runs GL-free and proves nothing about whether the notebook executes.

This module runs it. Three things only a real rollout can establish, each of which
would surface to a reader as a traceback rather than to CI:

* **Every code cell executes in order.** The notebook is the deliverable, and a
  reader runs it top to bottom. ``tests_integ/simulation/test_builtin_benchmark_load.py``
  already drives ``evaluate_benchmark`` on every shipped spec, so the library seam
  is covered; what is not covered is the notebook's own call shape - the keyword
  it passes, the result envelope it unpacks, the registry lookup it indexes.

* **The result carries the fields the notebook reads.** The cells index
  ``metrics["episodes_completed"]``, ``metrics["episodes"]``, and per row
  ``success`` / ``failure`` / ``steps`` / ``cumulative_reward`` / ``seed``. A
  renamed key raises ``KeyError`` in the reader's browser and in no test,
  because a notebook is not imported. ``success`` and ``failure`` are read as
  *separate* fields, not as opposites - the notebook's third outcome, "ran out of
  steps", is exactly the row where both are false, so collapsing them would make
  its own output wrong.

* **The seed claim is true.** The notebook tells the reader that a single
  attempt can be replayed from its recorded seed because per-attempt seeds derive
  from the master ``seed``. That is a reproducibility claim about the harness, and
  it is asserted here by running the same evaluation twice.

MuJoCo + asset download + real physics, so it is deliberately out of the GL-free
unit suite (collected only via ``hatch run test-integ``).
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402, F401 - imported after the MUJOCO_GL default is set

from strands_robots import Robot  # noqa: E402
from strands_robots.simulation import register_builtin_benchmarks  # noqa: E402
from strands_robots.simulation.benchmark import list_benchmarks  # noqa: E402

_NOTEBOOK = Path(__file__).resolve().parents[2] / "examples" / "notebooks" / "07_evaluate_a_policy.ipynb"

#: Fields the notebook's aggregate cell indexes by name.
_AGGREGATE_FIELDS = ("episodes_completed", "success_rate", "avg_reward", "avg_steps", "episodes")

#: Fields the notebook's per-attempt table indexes by name, per row.
_EPISODE_FIELDS = ("episode", "success", "failure", "steps", "cumulative_reward", "seed")


def _code_cells() -> list[str]:
    nb = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def _notebook_task() -> str:
    """The task name the notebook scores, read from the notebook itself."""
    for source in _code_cells():
        module = ast.parse(source, filename=str(_NOTEBOOK))
        for node in module.body:
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "TASK" for t in node.targets)
                and isinstance(node.value, ast.Constant)
            ):
                return str(node.value.value)
    raise AssertionError(f"no TASK assignment found in {_NOTEBOOK.name}")


@pytest.fixture(scope="module")
def executed_notebook() -> dict[str, Any]:
    """Run every code cell in order and return the resulting namespace.

    Executing the notebook rather than a transcription of it is the point: a copy
    would keep passing after the notebook regressed. A failure here names the
    cell index, so the traceback points at the cell a reader would see fail.
    """
    namespace: dict[str, Any] = {"__name__": "__main__"}
    for index, source in enumerate(_code_cells()):
        try:
            exec(compile(source, f"{_NOTEBOOK.name}:cell{index}", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # pragma: no cover - fires only on a regressed notebook
            raise AssertionError(
                f"{_NOTEBOOK.name} code cell {index} failed, so the notebook is broken for its "
                f"reader: {type(exc).__name__}: {exc}"
            ) from exc
    return namespace


def test_every_code_cell_runs_top_to_bottom(executed_notebook: dict[str, Any]) -> None:
    """The whole notebook executes, and leaves the objects its later cells depend on."""
    for name in ("registry", "sim", "metrics", "episodes", "benchmark", "custom_metrics"):
        assert name in executed_notebook, (
            f"{_NOTEBOOK.name} ran but defined no {name!r}; a cell was removed or renamed and the "
            "later cells that read it will fail for a reader."
        )


def test_the_aggregate_carries_every_field_the_notebook_prints(executed_notebook: dict[str, Any]) -> None:
    """A renamed metric key raises for a reader and for no other test."""
    metrics = executed_notebook["metrics"]
    missing = [field for field in _AGGREGATE_FIELDS if field not in metrics]
    assert not missing, (
        f"{_NOTEBOOK.name} indexes {missing} on the evaluation result, which no longer carries them. "
        f"Present: {sorted(metrics)}"
    )


def test_each_attempt_carries_every_field_the_table_prints(executed_notebook: dict[str, Any]) -> None:
    """The per-attempt table is the notebook's argument against reading the average."""
    episodes = executed_notebook["episodes"]
    assert episodes, f"{_NOTEBOOK.name} produced no per-attempt rows to print"
    for row in episodes:
        missing = [field for field in _EPISODE_FIELDS if field not in row]
        assert not missing, (
            f"per-attempt row {row.get('episode')} is missing {missing}, which the notebook's table "
            f"indexes. Present: {sorted(row)}"
        )


def test_success_and_failure_are_independent_fields(executed_notebook: dict[str, Any]) -> None:
    """Both false is a legal row, and it is the notebook's third outcome.

    The table branches success -> failure -> "ran out of steps". If the two ever
    became strict opposites that third branch would be unreachable and the
    notebook would be describing an outcome it can no longer show.
    """
    for row in executed_notebook["episodes"]:
        assert isinstance(row["success"], bool), f"success is {type(row['success']).__name__}, expected bool"
        assert isinstance(row["failure"], bool), f"failure is {type(row['failure']).__name__}, expected bool"
        assert not (row["success"] and row["failure"]), (
            f"attempt {row['episode']} reports both success and failure, which the notebook's "
            "three-way branch cannot render"
        )


def test_the_scored_task_loads_its_real_robot(executed_notebook: dict[str, Any]) -> None:
    """The notebook's registry lookup resolved to a robot that actually loaded.

    ``registry[TASK]["default_robot"]`` is indexed by the notebook; an asset
    rename would make the notebook's very first evaluation raise.
    """
    registry = executed_notebook["registry"]
    task = _notebook_task()
    assert task in registry, f"{_NOTEBOOK.name} scores {task!r}, absent from the registry it printed"
    assert executed_notebook["metrics"]["episodes_completed"] >= 1, (
        "the notebook completed no attempts, so its printed metrics describe nothing"
    )


def test_the_same_master_seed_reproduces_the_same_attempt_seeds() -> None:
    """The notebook's replay claim, asserted by running the evaluation twice.

    Run independently of the notebook namespace so the two runs are built the
    same way. Two attempts is enough: the claim is that the master seed
    determines the per-attempt seeds, not that any particular value appears.
    """
    register_builtin_benchmarks()
    task = _notebook_task()
    robot_name = list_benchmarks()[task]["default_robot"]

    def attempt_seeds(master: int) -> list[int]:
        sim = Robot(robot_name, mesh=False)
        try:
            result = sim.evaluate_benchmark(task, policy_provider="mock", n_episodes=2, seed=master)
            assert result.get("status") != "error", result
            metrics = next(c["json"] for c in result["content"] if "json" in c)
            return [row["seed"] for row in metrics["episodes"]]
        finally:
            destroy = getattr(sim, "destroy", None)
            if callable(destroy):
                destroy()

    first = attempt_seeds(0)
    again = attempt_seeds(0)
    other = attempt_seeds(1)

    assert first == again, (
        f"the same master seed produced different attempt seeds ({first} then {again}), so the "
        "notebook's claim that a single attempt can be replayed from its recorded seed is false"
    )
    assert first != other, (
        f"a different master seed produced the same attempt seeds ({first}), so the seed argument "
        "is not reaching the per-attempt derivation"
    )
