"""A recording's schema carries each frame's measured read time, under a key no LeRobot policy takes as an input."""

from __future__ import annotations

import numpy as np

from strands_robots.recording.features import ACTION, STATE, TIME, features_from
from strands_robots.types import JointState, Observation


def _obs(t: float = 12.5) -> Observation:
    names = ("shoulder_pan", "gripper")
    return Observation(t=t, joints=JointState(names, np.zeros(2), np.zeros(2), np.zeros(2)))


def test_the_schema_has_a_float64_measured_time_per_frame():
    feats = features_from(_obs(), ("shoulder_pan", "gripper"))
    assert feats[TIME] == {"dtype": "float64", "shape": (1,), "names": ["seconds"]}
    assert set(feats) == {STATE, ACTION, TIME}


def test_the_time_key_is_neither_an_observation_nor_an_action_nor_lerobots_own_timestamp():
    # lerobot's dataset_to_policy_features makes any key starting "observation" a state input, "action" an action
    assert not TIME.startswith(("observation", "action"))
    assert TIME not in ("timestamp", "frame_index", "episode_index", "index", "task_index")


def test_each_call_gets_its_own_copy_of_the_time_feature():
    a = features_from(_obs(), ("shoulder_pan", "gripper"))
    a[TIME]["names"] = ["changed"]
    assert features_from(_obs(), ("shoulder_pan", "gripper"))[TIME]["names"] == ["seconds"]
