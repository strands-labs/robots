"""An evaluation reports how close it came and how often it can be relied on.

Two additions to the metrics :meth:`PolicyRunner.evaluate` returns, both of which
answer a question the existing fields cannot:

* **``max_step_reward``** (per attempt) and **``avg_max_step_reward``**. The
  running total says how much shaped reward accrued over however many steps ran,
  so on a dense-reward task a long flailing attempt out-totals a short one that
  nearly finished - the total is partly a step count. The peak single-step reward
  says how close the attempt ever came, which is the signal that separates "never
  approached" from "approached and missed". LeRobot reports the same pair
  (``avg_sum_reward`` beside ``avg_max_reward`` in
  ``src/lerobot/scripts/lerobot_eval.py``); this package reported only the total.

* **``pass_hat_k``**. A success rate answers how often a policy works and does not
  answer whether it can be relied on repeatedly, which for anything driven in a
  loop is the deployable question: a policy at 60% clears five consecutive
  attempts about 8% of the time. Neither this package nor LeRobot reported any
  repeated-trial figure, so a reader comparing two checkpoints at the same mean
  had nothing distinguishing a consistent one from an erratic one.

The estimator is the subset form, ``C(c, k) / C(n, k)``, and **not**
``success_rate ** k``. That distinction is the point of most of this module rather
than a detail: the exponential form assumes attempts are independent, and attempts
on one policy against one scene are correlated by construction - a systematic
grasp offset fails every attempt rather than a fixed fraction of them - so it
reports a reliability the run never demonstrated. Measured on 3 successes out of
5, the two disagree by half (0.36 against 0.30 at ``k=2``), and the exponential
form is the optimistic one, which is the direction that costs a deployment
decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.simulation.policy_runner import _PASS_HAT_K_MAX, pass_hat_k

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import _BENCHMARK_REGISTRY  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_ROBOT_XML = """
<mujoco model="peak_reward_probe">
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3 3"/>
      <geom name="link1" type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="10"/>
  </actuator>
</mujoco>
"""

#: The per-step reward the probe task pays. Every step scores exactly this, so the
#: peak is knowable in advance and the assertion is on a value rather than a range.
_STEP_REWARD = 0.25

#: Steps the probe task allows. Small, because what is under test is the arithmetic
#: over the rollout rather than the rollout.
_MAX_STEPS = 4


@pytest.fixture(autouse=True)
def _clean_registry():
    """Leave the module-global benchmark registry as it was found."""
    before = dict(_BENCHMARK_REGISTRY)
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(before)


@pytest.fixture
def sim():
    engine = Simulation()
    yield engine
    engine.destroy()


@pytest.fixture
def constant_reward_task(tmp_path: Path, sim):
    """A task paying a known constant reward per step, never succeeding.

    ``constant`` is used rather than a distance or velocity term so the peak is a
    value this module chose: a shaped term would make the assertion depend on the
    probe's physics, and a failure would then be unattributable between the
    arithmetic under test and the scene.
    """
    sim.create_world()
    robot_xml = tmp_path / "probe.xml"
    robot_xml.write_text(_ROBOT_XML)
    sim.add_robot("arm1", urdf_path=str(robot_xml))

    spec_path = tmp_path / "constant_reward.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": "constant-reward-probe",
                "default_robot": "arm1",
                "supported_robots": [],
                "max_steps": _MAX_STEPS,
                "dense_reward": [{"predicate": "constant", "value": _STEP_REWARD}],
            }
        )
    )
    result = sim.register_benchmark_from_file("constant-reward-probe", str(spec_path))
    assert result.get("status") != "error", result
    return "constant-reward-probe"


def _metrics(result: dict) -> dict:
    assert result.get("status") != "error", result
    return next(c["json"] for c in result["content"] if "json" in c)


class TestPeakStepRewardIsReportedBesideTheTotal:
    """The peak is what says how close an attempt came; the total cannot."""

    def test_each_attempt_reports_the_peak_step_reward(self, sim, constant_reward_task) -> None:
        metrics = _metrics(sim.evaluate_benchmark(constant_reward_task, policy_provider="mock", n_episodes=2, seed=0))
        assert metrics["episodes"], "no attempts completed"
        for row in metrics["episodes"]:
            assert row["max_step_reward"] == pytest.approx(_STEP_REWARD), (
                f"attempt {row['episode']} reports peak {row['max_step_reward']}, expected the constant "
                f"{_STEP_REWARD} this task pays every step"
            )

    def test_the_peak_is_not_the_total(self, sim, constant_reward_task) -> None:
        """The assertion that makes the field worth having.

        With a constant reward the total is the peak times the step count, so a
        field that merely aliased the total would satisfy the test above on a
        single-step episode and fail here.
        """
        metrics = _metrics(sim.evaluate_benchmark(constant_reward_task, policy_provider="mock", n_episodes=1, seed=0))
        row = metrics["episodes"][0]
        assert row["steps"] > 1, "probe ran a single step, so this comparison proves nothing"
        assert row["cumulative_reward"] == pytest.approx(_STEP_REWARD * row["steps"]), (
            "the total is no longer the per-step reward accumulated, so the probe's premise is broken"
        )
        assert row["max_step_reward"] < row["cumulative_reward"], (
            f"peak {row['max_step_reward']} is not below total {row['cumulative_reward']} over "
            f"{row['steps']} steps, so the peak is aliasing the total"
        )

    def test_the_aggregate_averages_the_peaks(self, sim, constant_reward_task) -> None:
        metrics = _metrics(sim.evaluate_benchmark(constant_reward_task, policy_provider="mock", n_episodes=3, seed=0))
        assert metrics["avg_max_step_reward"] == pytest.approx(_STEP_REWARD)
        assert metrics["avg_max_step_reward"] != pytest.approx(metrics["avg_reward"]), (
            "the peak aggregate equals the total aggregate, so one of them is not measuring what it names"
        )


class TestPassHatKDescribesRepeatedAttempts:
    """The subset estimator, and the independence assumption it declines to make."""

    @pytest.mark.parametrize(
        ("n", "c", "k", "expected"),
        [
            # C(3,2)/C(5,2) = 3/10. The exponential form would give 0.36.
            (5, 3, 2, 0.30),
            (5, 3, 3, 0.10),
            (5, 5, 4, 1.00),  # every attempt succeeded: every subset does too
            (5, 0, 1, 0.00),  # none did
            (4, 2, 1, 0.50),  # k=1 is the success rate by construction
        ],
    )
    def test_the_subset_probability_is_exact(self, n: int, c: int, k: int, expected: float) -> None:
        assert pass_hat_k(n, c)[k] == pytest.approx(expected)

    def test_k_of_one_is_the_success_rate(self) -> None:
        """The anchor that makes the rest of the row readable."""
        for n, c in ((10, 7), (5, 1), (3, 3), (8, 0)):
            assert pass_hat_k(n, c)[1] == pytest.approx(c / n), f"k=1 disagrees with the success rate at {c}/{n}"

    def test_it_is_not_the_exponential_form(self) -> None:
        """Pinned as a difference, because the exponential form is the tempting one.

        ``success_rate ** k`` assumes independent attempts. It is also the
        optimistic direction, which is what makes substituting it expensive rather
        than merely wrong.
        """
        n, c = 5, 3
        subset = pass_hat_k(n, c)[2]
        exponential = (c / n) ** 2
        assert subset == pytest.approx(0.30)
        assert exponential == pytest.approx(0.36)
        assert subset < exponential, "the subset form is no longer the conservative one; re-derive before trusting it"

    def test_more_consecutive_attempts_is_never_more_likely(self) -> None:
        """Monotonicity, which a hand-rolled formula is easy to get wrong."""
        for n, c in ((10, 6), (8, 7), (6, 3)):
            values = [v for _, v in sorted(pass_hat_k(n, c).items())]
            assert values == sorted(values, reverse=True), f"pass^k rises with k at {c}/{n}: {values}"

    def test_k_beyond_the_attempts_run_is_absent_not_zero(self) -> None:
        """A run of 3 attempts says nothing about 5 in a row, so it must not answer.

        Reporting 0.0 there would read as measured unreliability rather than as an
        unasked question, and a reader comparing two runs of different lengths
        would be comparing a measurement against an absence.
        """
        row = pass_hat_k(3, 3)
        assert set(row) == {1, 2, 3}, f"expected k=1..3 for 3 attempts, got {sorted(row)}"
        assert row[3] == pytest.approx(1.0), "3 of 3 succeeded, so 3 in a row is certain among them"

    def test_no_completed_attempts_reports_nothing(self) -> None:
        assert pass_hat_k(0, 0) == {}, "there is no subset to draw from zero attempts"

    def test_the_reported_range_is_capped(self) -> None:
        """Bounded so a long run does not emit an unbounded row of tiny numbers."""
        assert set(pass_hat_k(50, 40)) == set(range(1, _PASS_HAT_K_MAX + 1))

    def test_the_evaluation_reports_it_with_string_keys(self, sim, constant_reward_task) -> None:
        """JSON object keys are strings, and this dict travels in a tool result.

        Asserted because an int-keyed dict survives an in-process assertion and
        changes shape the moment the result is serialised, which is where an agent
        reads it.
        """
        metrics = _metrics(sim.evaluate_benchmark(constant_reward_task, policy_provider="mock", n_episodes=2, seed=0))
        row = metrics["pass_hat_k"]
        assert row, "no pass_hat_k reported"
        assert all(isinstance(key, str) for key in row), f"pass_hat_k keys are not strings: {sorted(row)}"
        assert json.loads(json.dumps(row)) == row, "pass_hat_k does not survive a JSON round trip"
        # The probe never satisfies a success condition, so every k is 0.0.
        assert set(row.values()) == {0.0}, f"probe task cannot succeed, so every k should be 0.0: {row}"
