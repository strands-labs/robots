# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The all-missing state-key remedy must be one the observation can satisfy.

``_resolve_state_order`` makes an all-missing ``robot_state_keys`` binding loud
(raise under ``strict_keys``, warn-once otherwise). The diagnostic was correct
about the fault but its remedy was a fixed sentence for every caller, ending in
one worked example: ``embodiment='so101'``.

On real SO-101 hardware that example is self-defeating. ``so101`` is the SIM
embodiment and declares numeric MuJoCo actuator names (``'1'..'6'``), while a
lerobot ``SOFollower`` reports ``shoulder_pan.pos`` and friends. An operator who
follows the advice verbatim therefore re-enters the identical branch and is
handed the identical advice - a loop, on the path where the message's own stated
trigger ("a robot/sim that reports named joints") actually occurs. The hardware
embodiment they needed is ``so_real``, which the message never mentioned.

The message's cause sentence had the matching flaw: it asserted the generic-key
cause ("joint_0..joint_N were paired with ...") unconditionally, so it also
mis-described the ``'1'..'6'`` case above, where the configured keys are named.

These tests pin a remedy DERIVED from the observation. The load-bearing one is
:func:`test_every_suggested_embodiment_resolves_the_observation_that_suggested_it`,
a registry-wide invariant: for every embodiment, an observation built from its
own declared keys must only ever be offered suggestions that themselves resolve.
That is the property the fixed example violated, and it holds no matter how
``embodiments.json`` grows.

Ambiguity is preserved rather than resolved: ``so_real``, ``koch_real`` and
``omx_real`` declare identical keys, so an observation of those keys cannot pick
one, and all three are offered. Guessing would trade a loop for a wrong robot.

Behaviour is unchanged throughout - same resolved ordering, same
``generic_state_keys_used`` telemetry, same warn-once. Only the text differs.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from strands_robots.policies.lerobot_local.embodiment import EMBODIMENT_MAP
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

# A lerobot SOFollower reports '<motor>.pos' scalars. This is the observation the
# retired 'so101' example was handed to operators against.
SO101_HARDWARE_OBS = {
    "shoulder_pan.pos": 0.1,
    "shoulder_lift.pos": 0.2,
    "elbow_flex.pos": 0.3,
    "wrist_flex.pos": 0.4,
    "wrist_roll.pos": 0.5,
    "gripper.pos": 0.6,
}

# The keys 'so101' (the SIM embodiment) declares - numeric MuJoCo actuator names.
SO101_SIM_KEYS = ["1", "2", "3", "4", "5", "6"]

GENERIC_KEYS = [f"joint_{i}" for i in range(6)]


def _matching(observation):
    """Canonical embodiments whose declared keys ``observation`` satisfies.

    Computed here from ``EMBODIMENT_MAP`` alone so these tests collect and fail
    on their assertions against pre-fix code, rather than erroring on the import
    of the helper the fix adds.
    """
    present = set(observation)
    return sorted({m.name for m in EMBODIMENT_MAP.values() if m.state_keys and present.issuperset(m.state_keys)})


def _policy(*, keys, strict_keys=False, state_dim=6):
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path=None, policy_type="act", strict_keys=strict_keys)
    policy._input_features = {
        "observation.state": SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(state_dim,))
    }
    policy._device = torch.device("cpu")
    policy.robot_state_keys = list(keys)
    return policy


def _scalars(observation):
    return [k for k in observation if k != "task"]


def _warning(policy, observation):
    """Resolve, returning the single warning emitted (or None)."""
    with patch("strands_robots.policies.lerobot_local.policy.logger") as log:
        order = policy._resolve_state_order(observation, _scalars(observation))
    calls = log.warning.call_args_list
    assert len(calls) <= 1, f"expected warn-once, got {len(calls)}"
    if not calls:
        return None, order
    args = calls[0].args
    return (args[0] % args[1:] if len(args) > 1 else args[0]), order


# The loop the fixed example created


def test_the_retired_worked_example_is_not_offered_for_a_hardware_observation():
    """'so101' declares '1'..'6', which no '.pos' observation contains."""
    message, _ = _warning(_policy(keys=GENERIC_KEYS), SO101_HARDWARE_OBS)

    assert "so101" not in message
    # And the reason it must not be offered: it would not resolve.
    assert not set(SO101_HARDWARE_OBS).issuperset(SO101_SIM_KEYS)


def test_following_the_remedy_does_not_re_enter_the_all_missing_branch():
    """The end-to-end loop: apply what the message says, and it must resolve."""
    message, _ = _warning(_policy(keys=GENERIC_KEYS), SO101_HARDWARE_OBS)

    suggested = [name for name in EMBODIMENT_MAP if f"'{name}'" in message]
    assert suggested, f"message offered no embodiment: {message}"

    for name in suggested:
        keys = list(EMBODIMENT_MAP[name].state_keys)
        retry = _policy(keys=keys)
        retry_message, order = _warning(retry, SO101_HARDWARE_OBS)
        assert retry_message is None, f"embodiment={name!r} re-raised the diagnostic"
        assert retry.generic_state_keys_used is False
        assert order == keys


@pytest.mark.parametrize("name", sorted({m.name for m in EMBODIMENT_MAP.values() if m.state_keys}))
def test_every_suggested_embodiment_resolves_the_observation_that_suggested_it(name):
    """Registry-wide invariant: a suggestion never lands back in this branch.

    Built from each embodiment's own declared keys, so it stays honest as
    ``embodiments.json`` grows rather than pinning today's contents.
    """
    declared = list(EMBODIMENT_MAP[name].state_keys)
    observation = {key: float(i) for i, key in enumerate(declared)}

    candidates = _matching(observation)
    assert name in candidates, f"{name} does not match its own declared keys"

    for candidate in candidates:
        policy = _policy(keys=list(EMBODIMENT_MAP[candidate].state_keys), state_dim=len(declared))
        message, _ = _warning(policy, observation)
        assert message is None, f"{candidate} was suggested for {name}'s observation but does not resolve it"


# Ambiguity is reported, not guessed


def test_embodiments_with_identical_keys_are_all_offered():
    """so_real / koch_real / omx_real declare the same six '.pos' names."""
    candidates = _matching(SO101_HARDWARE_OBS)
    assert candidates == ["koch_real", "omx_real", "so_real"]

    message, _ = _warning(_policy(keys=GENERIC_KEYS), SO101_HARDWARE_OBS)
    for name in candidates:
        assert f"'{name}'" in message
    assert "cannot choose between them" in message


def test_aliases_collapse_to_one_canonical_suggestion():
    """so101_follower / so101_real / so_real are one config, offered once."""
    assert EMBODIMENT_MAP["so101_follower"].name == "so_real"
    candidates = _matching(SO101_HARDWARE_OBS)
    assert candidates == sorted(set(candidates))
    assert "so101_follower" not in candidates


def test_no_matching_embodiment_names_none_at_all():
    """An unrecognised robot gets set_robot_state_keys() and no guess."""
    observation = {"weird_joint_a": 0.1, "weird_joint_b": 0.2}
    assert _matching(observation) == []

    message, order = _warning(_policy(keys=GENERIC_KEYS, state_dim=2), observation)
    assert "No registered embodiment declares the observed keys" in message
    # Both documented mechanisms stay named - only the VALUE is withheld, so no
    # registry name the observation cannot satisfy is ever offered.
    assert "set_robot_state_keys()" in message
    assert "embodiment=" in message
    assert "embodiment='" not in message
    assert not any(f"'{name}'" in message for name in EMBODIMENT_MAP)
    assert order == ["weird_joint_a", "weird_joint_b"]


def test_a_partial_key_overlap_is_not_a_match():
    """A superset check, so a suggestion cannot land in the partial-missing path."""
    partial = dict(list(SO101_HARDWARE_OBS.items())[:5])
    assert "so_real" not in _matching(partial)


# The cause sentence tracks the keys in hand


def test_generic_keys_are_described_as_generic():
    message, _ = _warning(_policy(keys=GENERIC_KEYS), SO101_HARDWARE_OBS)
    assert "generic auto-generated keys (joint_0..joint_N)" in message


def test_named_keys_are_not_described_as_generic():
    """The '1'..'6' sim-vs-hardware case is a different robot, not generic keys."""
    message, _ = _warning(_policy(keys=SO101_SIM_KEYS), SO101_HARDWARE_OBS)
    assert "generic auto-generated keys" not in message
    assert "name a different robot/sim" in message


# Behaviour is unchanged


def test_strict_keys_raises_with_the_same_derived_remedy():
    policy = _policy(keys=GENERIC_KEYS, strict_keys=True)
    with pytest.raises(ValueError) as excinfo:
        policy._resolve_state_order(SO101_HARDWARE_OBS, _scalars(SO101_HARDWARE_OBS))

    text = str(excinfo.value)
    assert text.startswith("strict_keys=True: ")
    assert "'so_real'" in text
    assert "so101'" not in text


def test_fallback_ordering_and_telemetry_are_unchanged():
    policy = _policy(keys=GENERIC_KEYS)
    message, order = _warning(policy, SO101_HARDWARE_OBS)

    assert message is not None
    assert order == list(SO101_HARDWARE_OBS)
    assert policy.generic_state_keys_used is True


def test_warning_is_emitted_at_most_once_per_policy():
    policy = _policy(keys=GENERIC_KEYS)
    with patch("strands_robots.policies.lerobot_local.policy.logger") as log:
        for _ in range(5):
            policy._resolve_state_order(SO101_HARDWARE_OBS, _scalars(SO101_HARDWARE_OBS))
    assert log.warning.call_count == 1


def test_the_registry_helper_matches_the_locally_computed_expectation():
    """The shipped helper agrees with the independent computation above."""
    from strands_robots.policies.lerobot_local.embodiment import embodiments_matching

    for observation in (SO101_HARDWARE_OBS, {"weird_joint_a": 0.1}, dict(list(SO101_HARDWARE_OBS.items())[:5])):
        assert embodiments_matching(observation) == _matching(observation)
