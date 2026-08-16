"""Guard: a published multi-episode collection recipe must reset between rollouts.

``save_episode`` cuts a dataset episode boundary; it does not re-initialize the
world. So a collection loop that only calls ``run_policy`` then ``save_episode``
starts every rollout after the first at the *previous* rollout's terminal pose.
The dataset that comes out has the right episode COUNT and the wrong initial
state distribution: episode 0 begins at the scene's reset pose and episodes
1..N-1 begin wherever the arm was left, so the recorded start states are bimodal
and only one episode ever demonstrates the start the robot will actually be in.

That failure is invisible to the check the recipe is usually paired with -
``verify_dataset_episodes(n)`` counts episodes and passes - which is why the
recipe is graded here against the state the frames carry rather than against
their episode index.

The rest of the package already documents the three-step form: the
``run_policy(n_episodes=...)`` docstring and its implementation comment both
describe the manual loop it replaces as ``run_policy(); save_episode();
reset()``, ``docs/recording.md`` resets at the top of its loop and explains
``reset()``'s own episode-boundary behaviour, and
:mod:`strands_robots.policies.persistent` shows the same three calls. This guard
holds the remaining publication sites to that form.

Each site is graded by EXECUTING the call sequence it publishes - extracted from
the recipe's own source with :mod:`ast`, not paraphrased - against a real MuJoCo
engine, and then reading the recorded start states back out of the resulting
LeRobotDataset. A recipe whose loop body omits ``reset`` produces divergent
starts and fails; nothing about the wording is asserted. The arguments the
recipes elide as ``...`` are supplied by this module: it is the sequence of
calls that is under test.

The Newton recipe is executed on the MuJoCo engine because the shape of the
defect is backend-independent - ``save_episode`` is the shared facade method on
every backend and none of them reset - and MuJoCo is the reference backend
available without a solver install.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import strands_robots
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent

# Real arguments for the calls each recipe writes as ``...``. Only the sequence
# of calls comes from the recipe; these make it runnable.
_CALL_ARGS: dict[str, dict[str, Any]] = {
    "run_policy": {
        "policy_provider": "mock",
        "instruction": "collect",
        "n_steps": 8,
        "control_frequency": 30.0,
        "fast_mode": True,
    },
    "save_episode": {},
    "reset": {},
}

_EPISODES = 2


def _published_loop_calls(source: str) -> list[str]:
    """Names of the ``sim.<method>()`` calls inside a recipe's ``for`` loop.

    The recipes elide arguments as ``...``, which is a valid expression, so the
    snippet parses even though it would not run. Returns the calls in source
    order.
    """
    tree = ast.parse(inspect.cleandoc(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            return [
                stmt.func.attr
                for stmt in ast.walk(node)
                if isinstance(stmt, ast.Call)
                and isinstance(stmt.func, ast.Attribute)
                and isinstance(stmt.func.value, ast.Name)
                and stmt.func.value.id == "sim"
            ]
    return []


def _save_episode_docstring_recipe() -> str:
    """The collection snippet embedded in ``save_episode``'s docstring."""
    doc = inspect.getdoc(MuJoCoSimEngine.save_episode) or ""
    match = re.search(r"^ +sim\.start_recording\(.*?(?=\n\S|\Z)", doc, re.S | re.M)
    assert match is not None, (
        "premise: save_episode's docstring no longer embeds a "
        "sim.start_recording(...) collection snippet, so this guard would "
        "grade nothing. Re-point it at the snippet's new location."
    )
    return match.group(0)


def _newton_doc_recipe() -> str:
    """The collection block in the Newton recording guide."""
    text = (_REPO_ROOT / "docs" / "simulation" / "newton.md").read_text(encoding="utf-8")
    blocks = [
        block
        for block in re.findall(r"```python\n(.*?)```", text, re.S)
        if "start_recording" in block and "save_episode" in block
    ]
    assert len(blocks) == 1, (
        "premise: expected exactly one collection block in "
        f"docs/simulation/newton.md, found {len(blocks)}. A guard that matches "
        "no block, or the wrong one, would pass without grading the recipe."
    )
    return blocks[0]


_RECIPES = {
    "save_episode docstring": _save_episode_docstring_recipe,
    "docs/simulation/newton.md": _newton_doc_recipe,
}


def _episode_start_states(root: Path, repo_id: str) -> list[np.ndarray]:
    """First ``observation.state`` of every episode in a recorded dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=str(root))
    states = np.stack([dataset[i]["observation.state"].numpy() for i in range(len(dataset))])
    episodes = np.array([int(dataset[i]["episode_index"]) for i in range(len(dataset))])
    return [states[np.flatnonzero(episodes == ep)[0]] for ep in sorted(set(episodes.tolist()))]


def _collect(tmp_path: Path, calls: list[str], repo_id: str) -> list[np.ndarray]:
    """Run ``calls`` once per episode against a real engine; return start states."""
    sim = strands_robots.Robot("so101", mode="sim", mesh=False)
    try:
        started = sim.start_recording(repo_id=repo_id, task="collect", fps=30, root=str(tmp_path))
        assert started["status"] == "success", started
        for _ in range(_EPISODES):
            for call in calls:
                result = getattr(sim, call)(**_CALL_ARGS[call])
                assert result["status"] == "success", (call, result)
        stopped = sim.stop_recording()
        assert stopped["status"] == "success", stopped
    finally:
        sim.cleanup()
    return _episode_start_states(tmp_path, repo_id)


pytestmark = pytest.mark.skipif(
    strands_robots.Robot is None,  # pragma: no cover - import-time sanity only
    reason="strands_robots.Robot unavailable",
)


class TestEveryPublishedCollectionLoopResets:
    """Each published recipe must produce one start state, not a bimodal set."""

    @pytest.mark.parametrize("site", sorted(_RECIPES))
    def test_the_published_sequence_starts_every_episode_from_the_same_state(self, site: str, tmp_path: Path) -> None:
        calls = _published_loop_calls(_RECIPES[site]())
        assert "run_policy" in calls and "save_episode" in calls, (
            f"premise: {site} no longer publishes a run_policy + save_episode "
            f"loop (extracted {calls}), so this guard would grade nothing."
        )
        starts = _collect(tmp_path, calls, f"local/collect_{abs(hash(site)) % 10**6}")
        assert len(starts) == _EPISODES, f"{site}: expected {_EPISODES} episodes, got {len(starts)}"
        drift = float(np.abs(np.stack(starts) - starts[0]).max())
        assert drift == pytest.approx(0.0, abs=1e-6), (
            f"{site} publishes the loop {calls}, which does not re-initialize the "
            f"world between rollouts. Episode starts differ by up to {drift:.4f} "
            f"(episode 0 begins at {np.round(starts[0], 4).tolist()}, episode 1 at "
            f"{np.round(starts[1], 4).tolist()}), so every episode after the first "
            "demonstrates a start state the robot is never reset into. Publish "
            "run_policy() -> save_episode() -> reset(), the form run_policy's own "
            "n_episodes docstring already names."
        )


class TestTheMechanismTheRecipeDependsOn:
    """Controls: why the recipe needs ``reset()``, and that the fix is not cosmetic."""

    def test_omitting_reset_carries_the_previous_rollout_pose_into_the_next_episode(self, tmp_path: Path) -> None:
        """Without ``reset()`` the recorded starts diverge - the defect, measured."""
        starts = _collect(tmp_path, ["run_policy", "save_episode"], "local/collect_noreset")
        drift = float(np.abs(np.stack(starts) - starts[0]).max())
        assert drift > 0.1, (
            "save_episode is expected to cut an episode boundary WITHOUT "
            f"re-initializing the world, but the starts agree to {drift:.4f}. If "
            "save_episode has gained a reset, this control and the recipes it "
            "justifies both need revisiting."
        )

    def test_adding_reset_makes_every_episode_start_from_the_scene_pose(self, tmp_path: Path) -> None:
        """The three-step loop is the fix, on the same engine and rollout."""
        starts = _collect(tmp_path, ["run_policy", "save_episode", "reset"], "local/collect_reset")
        assert float(np.abs(np.stack(starts) - starts[0]).max()) == pytest.approx(0.0, abs=1e-6)

    def test_the_first_class_multi_episode_api_already_resets(self, tmp_path: Path) -> None:
        """``run_policy(n_episodes=N)`` is the behaviour the manual loop must match."""
        sim = strands_robots.Robot("so101", mode="sim", mesh=False)
        try:
            assert (
                sim.start_recording(repo_id="local/collect_firstclass", task="collect", fps=30, root=str(tmp_path))[
                    "status"
                ]
                == "success"
            )
            result = sim.run_policy(robot_name="so101", n_episodes=_EPISODES, **_CALL_ARGS["run_policy"])
            assert result["status"] == "success", result
            assert sim.stop_recording()["status"] == "success"
        finally:
            sim.cleanup()
        starts = _episode_start_states(tmp_path, "local/collect_firstclass")
        assert len(starts) == _EPISODES
        assert float(np.abs(np.stack(starts) - starts[0]).max()) == pytest.approx(0.0, abs=1e-6)


class TestTheSitesThatAlreadyPublishTheResetKeepIt:
    """Regression guard: the correct statements elsewhere must not lose ``reset``."""

    @pytest.mark.parametrize(
        "relative_path",
        [
            "strands_robots/simulation/base.py",
            "strands_robots/policies/persistent.py",
            "docs/recording.md",
        ],
    )
    def test_the_manual_loop_is_named_with_its_reset(self, relative_path: str) -> None:
        text = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "save_episode" in text, f"premise: {relative_path} no longer mentions save_episode"
        assert "reset()" in text, (
            f"{relative_path} documents the manual collection loop but no longer "
            "names reset(). Dropping it reintroduces the bimodal start-state "
            "distribution this module measures."
        )
