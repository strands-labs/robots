# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A declarative embodiment must report a state_keys mismatch, with a remedy.

``PackStateProcessorStep`` (the step a declarative ``embodiment=`` installs on
the preprocessor) can bind fewer state_keys than it declares in two ways, and
the generic ``robot_state_keys`` path already reports both the same way: it
quotes the keys the observation actually carries and appends the
registry-checked remedy from :func:`state_key_remedy`, which names an embodiment
only when the registry confirms that embodiment binds THIS observation.

The declarative path did neither:

* An **all-missing** binding returned the observation untouched and said
  nothing at all - no warning, no error, no remedy. The caller learned only
  that something downstream wanted ``observation.state``, after the multi-minute
  weight download, without the one fact that resolves it.
* A **partly-missing** binding warned, but asserted a single cause (a
  mimic/tendon gripper) as fact and offered no remedy and no view of what the
  observation does carry.

The all-missing case is reachable straight from the shipped catalog. Six alias
spellings resolve a name that is ALSO a sim-loadable registry robot to a
``*_real`` embodiment whose ``'<motor>.pos'`` state_keys no sim observation
emits - ``so100_follower`` / ``so101_follower`` / ``koch_follower`` /
``bi_so_follower`` / ``lekiwi`` / ``openarm``. ``embodiment="so101"`` driven from
real hardware is the mirror of that mismatch and has always had a ``.pos``
fallback; the sim direction had no report.
"""

import logging
import re

import numpy as np
import pytest

pytest.importorskip("lerobot")

import strands_robots.policies.lerobot_local.embodiment as E
from strands_robots.policies.lerobot_local.embodiment import (
    load_embodiment,
    matching_embodiments,
    observed_state_keys,
)

_LOGGER = "strands_robots.policies.lerobot_local.embodiment"

# What a MuJoCo so101 reports: the asset's numeric joints plus their velocity
# siblings. None of the '<motor>.pos' names a *_real embodiment declares.
SIM_SO101 = {
    k: 0.1 * i
    for i, k in enumerate(["1", "1.vel", "2", "2.vel", "3", "3.vel", "4", "4.vel", "5", "5.vel", "6", "6.vel"])
}
# What a lerobot SOFollower reports, in motor order.
HARDWARE_SO = {
    "shoulder_pan.pos": 1.0,
    "shoulder_lift.pos": 2.0,
    "elbow_flex.pos": 3.0,
    "wrist_flex.pos": 4.0,
    "wrist_roll.pos": 5.0,
    "gripper.pos": 6.0,
}

# The aloha bimanual actuator convention: the gripper ACTUATORS sit at indices 6
# and 13, and the sim exposes finger JOINTS instead - a partial mismatch.
ALOHA_ARMS = [
    f"{side}/{j}"
    for side in ("left", "right")
    for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
]


@pytest.fixture(autouse=True)
def _fresh_warn_dedup():
    """Each test starts with an empty warn-once ledger."""
    E._WARNED_STATE_KEY_MISMATCH.clear()
    yield
    E._WARNED_STATE_KEY_MISMATCH.clear()


def _step(state_keys, **kw):
    Step = E.register_pack_state_step()
    assert Step is not None, "lerobot processor framework unavailable"
    return Step(state_keys=list(state_keys), dim_policy=kw.pop("dim_policy", "pad"), **kw)


def _reports(caplog, step, observation):
    """Run one observation and return the state-mismatch reports it produced."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = step.observation(dict(observation))
    return out, [r.getMessage() for r in caplog.records if "declared state_keys" in r.getMessage()]


def _aloha_sim_observation():
    obs = {k: 100.0 + i for i, k in enumerate(ALOHA_ARMS)}
    for side in ("left", "right"):
        obs[f"{side}/left_finger"] = 0.02
        obs[f"{side}/right_finger"] = 0.02
    return obs


class TestATotalMismatchIsReported:
    def test_it_does_not_return_silently(self, caplog):
        """FAILS pre-fix: the all-missing branch returned the observation with no
        report of any kind, so the only signal was a downstream error that knows
        nothing about embodiments."""
        embodiment = load_embodiment("so101_follower")
        out, reports = _reports(caplog, _step(embodiment.state_keys), SIM_SO101)
        assert "observation.state" not in out, "premise: this configuration packs no state"
        assert len(reports) == 1, f"expected exactly one report, got {reports}"

    def test_it_names_the_declared_keys_and_the_observed_keys(self, caplog):
        """Both halves of the mismatch, so the caller can see which convention
        each side is using. FAILS pre-fix (no message at all)."""
        embodiment = load_embodiment("so101_follower")
        _, reports = _reports(caplog, _step(embodiment.state_keys), SIM_SO101)
        message = reports[0]
        assert "shoulder_pan.pos" in message
        for observed in observed_state_keys(SIM_SO101):
            assert repr(observed) in message, f"observed key {observed!r} not quoted"

    def test_its_remedy_names_an_embodiment_that_binds_this_observation(self, caplog):
        """The remedy must be followable: applying it has to escape the mismatch
        rather than land back on it. FAILS pre-fix (no remedy is offered)."""
        embodiment = load_embodiment("so101_follower")
        _, reports = _reports(caplog, _step(embodiment.state_keys), SIM_SO101)
        offered = re.findall(r"embodiment='([^']+)'", reports[0])
        assert offered, f"no embodiment= remedy in {reports[0]!r}"
        bindable = matching_embodiments(observed_state_keys(SIM_SO101))
        for name in offered:
            assert name in bindable, f"remedy embodiment={name!r} does not bind this observation"
            assert set(load_embodiment(name).state_keys) <= set(SIM_SO101)

    def test_reporting_replaces_the_silence_and_not_the_passthrough(self, caplog):
        """The observation is still handed on untouched for the caller's own
        handling; emitting an all-zero state vector instead would hide the
        mismatch behind plausible numbers."""
        observation = {"unrelated": 5.0}
        out, reports = _reports(caplog, _step(["a", "b"]), observation)
        assert out == observation
        assert len(reports) == 1

    def test_it_is_reported_once_per_distinct_mismatch(self, caplog):
        """The report is deduplicated across the 50Hz control loop, but a second
        mismatch with a different shape still gets its own report."""
        step = _step(load_embodiment("so101_follower").state_keys)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            step.observation(dict(SIM_SO101))
            step.observation(dict(SIM_SO101))
            step.observation({"only_this": 1.0})
        assert len([r for r in caplog.records if "declared state_keys" in r.getMessage()]) == 2


class TestAPartialMismatchGetsTheSameRemedy:
    def test_it_offers_a_registry_checked_remedy(self, caplog):
        """FAILS pre-fix: the partial warning asserted a mimic-gripper cause and
        stopped there, while the generic path appended the same remedy it uses
        for an all-missing binding."""
        aloha = load_embodiment("aloha")
        observation = _aloha_sim_observation()
        _, reports = _reports(caplog, _step(aloha.state_keys), observation)
        assert len(reports) == 1
        assert "set_robot_state_keys" in reports[0], "no followable remedy offered"

    def test_it_quotes_the_keys_the_observation_carries(self, caplog):
        """The absent keys alone do not tell the caller what IS bindable.
        FAILS pre-fix: only the missing keys were named."""
        aloha = load_embodiment("aloha")
        observation = _aloha_sim_observation()
        _, reports = _reports(caplog, _step(aloha.state_keys), observation)
        assert "left/left_finger" in reports[0]

    def test_its_remedy_does_not_invent_an_embodiment(self, caplog):
        """No shipped embodiment declares the aloha finger-joint naming, so
        naming one would send the caller back to this same mismatch."""
        aloha = load_embodiment("aloha")
        observation = _aloha_sim_observation()
        _, reports = _reports(caplog, _step(aloha.state_keys), observation)
        for name in re.findall(r"embodiment='([^']+)'", reports[0]):
            assert set(load_embodiment(name).state_keys) <= set(observation)


class TestTheReportDoesNotChangeWhatIsBound:
    """Controls: every test here passes both pre- and post-fix.

    They pin the behaviour the added reporting must NOT disturb - a bound
    embodiment stays quiet, the mirror mismatch keeps its hardware fallback, and
    a partly-bound vector keeps its in-place zero-fill.
    """

    def test_a_fully_bound_embodiment_reports_nothing(self, caplog):
        out, reports = _reports(caplog, _step(["a", "b", "c"]), {"a": 1.0, "b": 2.0, "c": 3.0})
        np.testing.assert_allclose(out["observation.state"].numpy(), [1.0, 2.0, 3.0], atol=1e-5)
        assert reports == []

    def test_the_hardware_pos_fallback_still_binds_and_stays_quiet(self, caplog):
        """A SIM embodiment driven from real hardware is the mirror mismatch and
        has always had a fallback; it binds, so there is nothing to report."""
        so101 = load_embodiment("so101")
        out, reports = _reports(caplog, _step(so101.state_keys, state_units=so101.state_units), HARDWARE_SO)
        assert "observation.state" in out
        assert len(out["observation.state"]) == len(so101.state_keys)
        assert reports == []

    def test_a_partial_mismatch_still_zero_fills_in_place(self, caplog):
        """The present joints keep their model index and the absent gripper slots
        read 0.0 - unchanged by the added report."""
        aloha = load_embodiment("aloha")
        out, _ = _reports(caplog, _step(aloha.state_keys), _aloha_sim_observation())
        state = out["observation.state"].numpy()
        assert len(state) == 14
        np.testing.assert_allclose(state[0:6], [100.0, 101.0, 102.0, 103.0, 104.0, 105.0], atol=1e-5)
        assert state[6] == 0.0
        np.testing.assert_allclose(state[7:13], [106.0, 107.0, 108.0, 109.0, 110.0, 111.0], atol=1e-5)
        assert state[13] == 0.0


class TestOneRuleAnswersWhatTheObservationCarries:
    """``observed_state_keys`` is the single rule every state diagnostic quotes."""

    def test_it_excludes_the_instruction_and_camera_frames(self):
        keys = observed_state_keys(
            {"1": 0.5, "task": "pick the cube", "front": np.zeros((4, 4, 3), dtype=np.uint8), "pose": np.zeros(7)}
        )
        assert keys == ["1", "pose"], "a 1-D state vector is a candidate; task and a frame are not"

    def test_the_report_quotes_exactly_that_rule(self, caplog):
        """The declarative report and the rule cannot drift apart, because the
        report is built from it rather than from a filter of its own."""
        observation = {**SIM_SO101, "task": "pick", "front": np.zeros((2, 2, 3), dtype=np.uint8)}
        _, reports = _reports(caplog, _step(load_embodiment("so101_follower").state_keys), observation)
        assert f"{observed_state_keys(observation)}" in reports[0]
