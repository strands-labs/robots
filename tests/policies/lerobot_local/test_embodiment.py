"""Tests for the declarative EmbodimentMap mapping (SOLUTION.md).

These tests exercise the REAL mapping path (not mocked) to close the gap that
let B7/B12 slip past the mock-heavy existing suite.
"""

import numpy as np
import pytest

from strands_robots.policies.lerobot_local.embodiment import (
    EMBODIMENT_MAP,
    EmbodimentMap,
    load_embodiment,
    reconcile_dim,
    register_pack_state_step,
)


class _Feat:
    """Minimal stand-in for a LeRobot PolicyFeature (just needs .shape)."""

    def __init__(self, shape):
        self.shape = shape


# reconcile_dim


def test_reconcile_dim_exact():
    assert reconcile_dim([1.0, 2.0, 3.0], 3, "strict") == [1.0, 2.0, 3.0]


def test_reconcile_dim_pad():
    assert reconcile_dim([1.0, 2.0], 4, "pad") == [1.0, 2.0, 0.0, 0.0]


def test_reconcile_dim_truncate():
    assert reconcile_dim([1.0, 2.0, 3.0, 4.0], 2, "truncate") == [1.0, 2.0]


def test_reconcile_dim_strict_raises():
    with pytest.raises(ValueError, match="dim_policy"):
        reconcile_dim([1.0, 2.0], 4, "strict")


def test_reconcile_dim_pad_cannot_shrink():
    with pytest.raises(ValueError, match="cannot pad"):
        reconcile_dim([1.0, 2.0, 3.0], 2, "pad")


def test_reconcile_dim_unknown_policy():
    # Use a length != expected so the policy branch is actually reached
    # (an exact-length match short-circuits and returns before policy check).
    with pytest.raises(ValueError, match="Unknown dim_policy"):
        reconcile_dim([1.0, 2.0], 1, "bogus")


# Registry loading + _extends + aliases


def test_registry_loaded():
    assert "panda_libero" in EMBODIMENT_MAP
    assert "so100" in EMBODIMENT_MAP


def test_extends_inheritance():
    # wx250s _extends vx300s (identical Trossen arm joint topology).
    wx = load_embodiment("wx250s")
    vx = load_embodiment("vx300s")
    assert wx.state_keys == vx.state_keys
    assert wx.dim_policy == vx.dim_policy


def test_so100_so101_are_distinct():
    # Regression: so100 (trs_so_arm100 XML: Rotation/Pitch/...) and so101
    # (robotstudio_so101 XML: 1..6) have DIFFERENT sim joint names and must
    # NOT share a schema. The old config wrongly had so101 _extends so100.
    so100 = load_embodiment("so100")
    so101 = load_embodiment("so101")
    assert so100.state_keys == ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    assert so101.state_keys == ["1", "2", "3", "4", "5", "6"]
    assert so100.state_keys != so101.state_keys


def test_aliases():
    assert load_embodiment("franka_libero").name == "panda_libero"
    # panda (joint-space sim) is distinct from panda_libero (EEF task-space).
    assert load_embodiment("panda").name == "panda"
    assert load_embodiment("franka").name == "panda"
    assert load_embodiment("so100_real").name == "so_real"
    assert load_embodiment("kinova").name == "kinova_gen3"


def test_real_hardware_keys_have_pos_suffix():
    # *_real entries must use lerobot driver motor-feature names ('<motor>.pos').
    real = load_embodiment("so_real")
    assert all(k.endswith(".pos") for k in real.state_keys)
    assert "shoulder_pan.pos" in real.state_keys


def test_load_unknown_raises():
    with pytest.raises(ValueError, match="Unknown embodiment"):
        load_embodiment("does_not_exist")


def test_load_inline_dict():
    em = load_embodiment({"obs_rename": {"a": "observation.images.a"}, "state_keys": ["j1"]})
    assert em.obs_rename == {"a": "observation.images.a"}
    assert em.state_keys == ["j1"]


def test_load_passthrough_instance():
    em = EmbodimentMap(name="x")
    assert load_embodiment(em) is em


# validate() fail-fast


def _features():
    inp = {
        "observation.images.image": _Feat((3, 256, 256)),
        "observation.images.wrist_image": _Feat((3, 256, 256)),
        "observation.state": _Feat((7,)),
    }
    out = {"action": _Feat((7,))}
    return inp, out


def test_validate_ok():
    inp, out = _features()
    load_embodiment("panda_libero").validate(inp, out)  # 7 state, 7 action -> OK


def test_validate_unknown_rename_target():
    inp, out = _features()
    em = EmbodimentMap(name="bad", obs_rename={"cam": "observation.images.NOPE"})
    with pytest.raises(ValueError, match="doesn't declare"):
        em.validate(inp, out)


def test_validate_state_dim_mismatch_strict():
    inp, out = _features()
    em = EmbodimentMap(name="bad", state_keys=["a", "b", "c"], dim_policy="strict")
    with pytest.raises(ValueError, match="state_keys"):
        em.validate(inp, out)


def test_validate_state_dim_mismatch_pad_allowed():
    inp, out = _features()
    # pad opts in to adaptation -> no raise
    em = EmbodimentMap(name="ok", state_keys=["a", "b", "c"], dim_policy="pad")
    em.validate(inp, out)


def test_validate_action_dim_mismatch():
    inp, out = _features()
    em = EmbodimentMap(name="bad", action_keys=["a", "b"])  # 2 != 7
    with pytest.raises(ValueError, match="action_keys"):
        em.validate(inp, out)


# PackStateProcessorStep


def _require_pack_state():
    """Return the registered PackState step class, skipping if lerobot absent.

    ``register_pack_state_step`` returns ``None`` when lerobot's processor
    framework is not importable. Without this guard the tests below hard-fail
    (``None`` is not callable) on a minimal env instead of skipping cleanly.
    """
    Step = register_pack_state_step()
    if Step is None:
        pytest.skip("lerobot processor framework unavailable")
    return Step


def test_pack_state_composes_in_order():
    Step = _require_pack_state()
    s = Step(state_keys=["x", "y", "z"], expected_dim=3, dim_policy="strict")
    obs = {"x": 1.0, "y": 2.0, "z": 3.0, "observation.images.image": np.zeros((3, 4, 4))}
    out = s.observation(dict(obs))
    assert list(out["observation.state"]) == [1.0, 2.0, 3.0]
    assert "x" not in out and "y" not in out and "z" not in out
    assert "observation.images.image" in out  # non-state keys preserved


def test_pack_state_idempotent_when_already_packed():
    Step = _require_pack_state()
    s = Step(state_keys=["x"], expected_dim=1, dim_policy="strict")
    pre = {"observation.state": np.array([9.0, 9.0])}
    out = s.observation(dict(pre))
    assert list(out["observation.state"]) == [9.0, 9.0]


def test_pack_state_pads():
    Step = _require_pack_state()
    s = Step(state_keys=["a", "b"], expected_dim=4, dim_policy="pad")
    out = s.observation({"a": 1.0, "b": 2.0})
    assert list(out["observation.state"]) == [1.0, 2.0, 0.0, 0.0]


def test_pack_state_get_config_roundtrips():
    Step = _require_pack_state()
    s = Step(state_keys=["a", "b"], expected_dim=2, dim_policy="pad")
    cfg = s.get_config()
    assert cfg == {"state_keys": ["a", "b"], "expected_dim": 2, "dim_policy": "pad"}


# Full lerobot driver coverage guard


def test_all_lerobot_drivers_have_embodiment():
    """Every robot subclass registered in lerobot.robots must resolve to an
    embodiment (directly or via alias). Guards against a new lerobot driver
    silently lacking a key-mapping. Ground truth: the @RobotConfig
    .register_subclass names in lerobot-src/src/lerobot/robots/*.
    """
    lerobot_drivers = [
        "so100_follower",
        "so101_follower",
        "koch_follower",
        "omx_follower",
        "openarm_follower",
        "bi_openarm_follower",
        "bi_so_follower",
        "rebot_b601_follower",
        "bi_rebot_b601_follower",
        "lekiwi",
        "lekiwi_client",
        "reachy2",
        "hope_jr_hand",
        "hope_jr_arm",
        "earthrover_mini_plus",
        "unitree_g1",
    ]
    missing = []
    for name in lerobot_drivers:
        try:
            load_embodiment(name)
        except ValueError:
            missing.append(name)
    assert not missing, f"lerobot drivers without embodiment mapping: {missing}"


def test_real_hardware_entries_use_pos_or_velocity_keys():
    """All *_real arm/hand entries use '<motor>.pos' driver feature keys
    (the lerobot _motors_ft convention). The mobile rover uses velocity cmds.
    """
    pos_robots = [
        "omx_real",
        "bi_so_real",
        "openarm_real",
        "bi_openarm_real",
        "rebot_b601_real",
        "bi_rebot_b601_real",
        "reachy2_real",
        "hope_jr_arm_real",
        "hope_jr_hand_real",
    ]
    for name in pos_robots:
        em = load_embodiment(name)
        assert all(k.endswith(".pos") for k in em.state_keys), f"{name} non-.pos keys"
    rover = load_embodiment("earthrover_real")
    assert rover.state_keys == ["linear_velocity", "angular_velocity"]
