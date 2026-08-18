"""Regression: a raising episode-outcome criterion names what failed.

The eval loops call a caller-supplied outcome criterion after every applied
action - ``eval_policy``'s ``success_fn`` and a benchmark spec's ``is_success``
/ ``is_failure``. Every other per-step hook on those paths already states what a
raise means: a generic ``on_frame`` failure is best-effort telemetry (warn and
continue), a ``RecordingFrameError`` from it is data loss and propagates,
``spec.on_step`` returns ``status="error"`` naming the hook, and ``run``'s
``stop_when`` is fatal with a message naming the step. The outcome criterion -
the one that decides the evaluation's headline ``success_rate`` - had none, so
it arrived as the caller's own bare exception from somewhere inside a nested
loop: nothing named the criterion, the episode or the step, and a criterion that
only breaks on a later episode cost the whole evaluation without saying why.

What is pinned here is the message, not the posture. ``evaluate`` surfaces a
rollout failure as a raise by design - a raising ``get_actions`` and a lost
recording frame both reach the caller that way - so the criterion failure stays
a raise too, and the controls pin that neighbouring policy unchanged. A
``bool()``-coercible verdict (including the NumPy scalar that
``observation["x"] > 0.5`` returns, which is not an instance of ``bool``) must
keep working, a false verdict must still be a measured episode, an absent
criterion must still report ``success_measured=False``, and ``spec.on_step`` /
``on_frame`` / ``CooperativeStop`` / ``RecordingFrameError`` must each keep the
policy they already had.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import (  # noqa: E402
    BenchmarkProtocol,
    StepInfo,
    register_benchmark,
    unregister_benchmark,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.policy_runner import (  # noqa: E402
    CooperativeStop,
    RecordingFrameError,
)

ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="30"/>
  </actuator>
</mujoco>
"""


class _Spec(BenchmarkProtocol):
    """Always-running spec whose verdict hooks are supplied per test."""

    max_steps = 4

    def __init__(self, *, success: Any = None, failure: Any = None, step: Any = None) -> None:
        self._success = success or (lambda sim: False)
        self._failure = failure or (lambda sim: False)
        self._step = step

    @property
    def supported_robots(self) -> list[str]:
        return ["arm1"]

    @property
    def default_robot(self) -> str:
        return "arm1"

    def on_episode_start(self, sim: Any, rng: random.Random) -> None:
        return None

    def on_step(self, sim: Any, obs: dict[str, Any], action: dict[str, Any]) -> StepInfo:
        if self._step is not None:
            self._step(sim)
        return StepInfo(reward=0.0)

    def is_success(self, sim: Any) -> bool:
        return self._success(sim)

    def is_failure(self, sim: Any) -> bool:
        return self._failure(sim)


@pytest.fixture
def sim(tmp_path):
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    engine = Simulation(tool_name="crit", mesh=False)
    engine.create_world()
    added = engine.add_robot(name="arm1", urdf_path=str(xml_path))
    assert added["status"] == "success", added
    try:
        yield engine
    finally:
        engine.cleanup(policy_stop_timeout=0.5)


def _json(result: dict) -> dict:
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in result: {result}")


def _text(result: dict) -> str:
    return "".join(b.get("text", "") for b in result["content"] if isinstance(b, dict))


def _eval(engine, **kwargs) -> dict:
    return engine.eval_policy(robot_name="arm1", policy_provider="mock", max_steps=4, control_frequency=20.0, **kwargs)


def _bench(engine, spec: _Spec, **kwargs) -> dict:
    name = "crit_bench"
    register_benchmark(name, spec)
    try:
        return engine.evaluate_benchmark(
            benchmark_name=name,
            robot_name="arm1",
            policy_provider="mock",
            control_frequency=20.0,
            **kwargs,
        )
    finally:
        try:
            unregister_benchmark(name)
        except Exception:
            # Best-effort teardown; never mask the real assertion failure.
            pass


def _raise(exc: BaseException):
    def _fn(_subject):
        raise exc

    return _fn


class TestARaisingOutcomeCriterionNamesWhatFailed:
    def test_a_raising_success_fn_names_the_criterion_the_episode_and_the_step(self, sim):
        """Pre-fix this arrived as a bare ``KeyError`` from inside the loop,
        naming neither the criterion nor where it broke."""
        with pytest.raises(RuntimeError, match=r"success_fn raised at episode 0, step \d+"):
            _eval(sim, n_episodes=3, success_fn=_raise(KeyError("cube")))

    def test_the_original_exception_is_chained_not_replaced(self, sim):
        """The caller still needs their own traceback to fix the criterion."""
        with pytest.raises(RuntimeError) as exc:
            _eval(sim, n_episodes=3, success_fn=_raise(KeyError("cube")))
        assert isinstance(exc.value.__cause__, KeyError), exc.value.__cause__
        assert "KeyError" in str(exc.value), str(exc.value)

    def test_the_message_says_why_the_evaluation_stopped(self, sim):
        """A success_rate over episodes whose outcome was never determined is
        not a measurement - the message has to say so, because the alternative
        (carry on and average) is the silent failure this prevents."""
        with pytest.raises(RuntimeError) as exc:
            _eval(sim, n_episodes=3, success_fn=_raise(KeyError("cube")))
        assert "success_rate" in str(exc.value), str(exc.value)

    def test_a_raising_spec_is_success_names_the_criterion(self, sim):
        """Same contract on the benchmark route, whose ``on_step`` sibling
        already named itself on failure."""
        with pytest.raises(RuntimeError, match=r"\.is_success raised at episode 0, step \d+"):
            _bench(sim, _Spec(success=_raise(ValueError("no body 'mug'"))), n_episodes=2)

    def test_a_raising_spec_is_failure_names_the_criterion(self, sim):
        """``is_failure`` is evaluated on the same per-step chain and carries
        the same contract."""
        with pytest.raises(RuntimeError, match=r"\.is_failure raised at episode 0, step \d+"):
            _bench(sim, _Spec(failure=_raise(ValueError("no body 'mug'"))), n_episodes=2)


class TestTheWorkingCriterionPathsAreUnchanged:
    """Controls: these hold both before and after the fix. They fail if the
    coercion rejects a legitimate verdict or the terminal handler swallows a
    policy that another hook family already owns."""

    def test_a_true_verdict_still_reports_success(self, sim):
        result = _eval(sim, n_episodes=2, success_fn=lambda obs: True)
        assert result["status"] == "success", result
        payload = _json(result)
        assert payload["success_rate"] == 1.0, payload
        assert payload["success_measured"] is True, payload

    def test_a_false_verdict_is_still_a_measured_episode(self, sim):
        result = _eval(sim, n_episodes=2, success_fn=lambda obs: False)
        payload = _json(result)
        assert result["status"] == "success", result
        assert payload["success_rate"] == 0.0, payload
        assert payload["success_measured"] is True, payload

    def test_a_numpy_scalar_verdict_is_still_accepted(self, sim):
        """``observation["x"] > 0.5`` returns ``numpy.bool_``, which is NOT an
        instance of ``bool`` - so the verdict must be coerced, never
        type-checked."""
        np = pytest.importorskip("numpy")
        assert not isinstance(np.bool_(True), bool)  # premise
        result = _eval(sim, n_episodes=2, success_fn=lambda obs: np.bool_(True))
        assert result["status"] == "success", result
        assert _json(result)["success_rate"] == 1.0, _json(result)

    def test_an_absent_criterion_is_still_reported_as_unmeasured(self, sim):
        result = _eval(sim, n_episodes=2)
        payload = _json(result)
        assert result["status"] == "success", result
        assert payload["success_measured"] is False, payload

    def test_a_raising_on_step_still_reports_its_own_message(self, sim):
        """The adjacent guard keeps its own wording and its own envelope."""
        result = _bench(sim, _Spec(step=_raise(RuntimeError("sensor gone"))), n_episodes=1)
        assert result["status"] == "error", result
        assert "on_step failed" in _text(result), _text(result)

    def test_a_lost_recording_frame_still_propagates(self, sim):
        """``RecordingFrameError`` from ``on_frame`` is data loss, not
        telemetry, and reaches the caller as a raise. Wrapping the eval loop in
        a terminal handler to convert the criterion failure into an envelope
        would have swallowed this one too."""

        def lose_a_frame(step, obs, action):
            raise RecordingFrameError("dataset add_frame failed after 0 frame(s) written")

        with pytest.raises(RecordingFrameError, match="add_frame failed"):
            _eval(sim, n_episodes=2, success_fn=lambda obs: False, on_frame=lose_a_frame)

    def test_a_raising_on_frame_hook_stays_best_effort(self, sim):
        """``on_frame`` is telemetry: a failure is logged and the eval still
        completes. A terminal handler that made it fatal would break this."""

        def boom(step, obs, action):
            raise RuntimeError("telemetry sink down")

        result = _eval(sim, n_episodes=2, success_fn=lambda obs: False, on_frame=boom)
        assert result["status"] == "success", result
        assert _json(result)["episodes_completed"] == 2, _json(result)
        assert _json(result)["success_measured"] is True, _json(result)

    def test_a_cooperative_stop_still_ends_the_eval_cleanly(self, sim):
        """``CooperativeStop`` is a BaseException and its handler precedes the
        terminal one, so a graceful stop still reports the completed episodes."""
        calls = {"n": 0}

        def stop_after_two(step, obs, action):
            calls["n"] += 1
            if calls["n"] > 2:
                raise CooperativeStop()

        result = _eval(sim, n_episodes=3, success_fn=lambda obs: False, on_frame=stop_after_two)
        assert result["status"] == "success", result
        assert _json(result)["stopped_early"] is True, _json(result)
