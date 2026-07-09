"""Unit tests for :mod:`strands_robots.policies.motionbricks` - no GPU, no checkpoints.

These exercise :class:`MotionBricksPolicy` against a stubbed generator (injected
via the ``motion_agent`` seam), so they run on any developer machine without the
``motionbricks`` install, the git-LFS checkpoints, or a CUDA GPU. The live
generator path is covered by ``tests_integ/policies/motionbricks/`` (gated by
``MOTIONBRICKS_CKPT``).

Pinned behaviour (issue #466 MotionBricks acceptance criteria):

* ``create_policy("motionbricks", ...)`` / ``create_policy("motion_bricks", ...)``
  round-trip via the factory + registry.
* ``requires_images is False`` and ``provider_name == "motionbricks"``.
* Config loads + validates (fail-fast on bad knobs); ``controller_dt`` matches
  the upstream ``(8/fps) * generate_dt``.
* Style resolution (int index + str name) and control-signal assembly produce
  the exact upstream control dict; out-of-range / unknown styles raise.
* The action dict carries the G1's 29 joints in ``WBC_G1_ALL_JOINTS`` order
  with python-float values; style switching is forwarded to the generator.
* Missing config + missing ``motion_agent`` raises ``ValueError``; a missing
  checkpoint dir raises ``RuntimeError`` (no silent fallback).
* ``set_robot_state_keys`` validates the G1 joint names by name.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from strands_robots.policies import Policy, create_policy, list_providers
from strands_robots.policies.motionbricks import (
    MotionBricksConfig,
    MotionBricksPolicy,
    build_control_signals,
    resolve_mode,
)
from strands_robots.policies.motionbricks import policy as mb_policy
from strands_robots.policies.motionbricks.config import NUM_REGEN_FRAMES
from strands_robots.policies.motionbricks.observation import allowed_pred_num_tokens
from strands_robots.policies.wbc.policy import WBC_G1_ALL_JOINTS

# G1 clip set (names + per-mode token masks) mirroring upstream clip_holder_G1.
_CLIP_KEYS = [
    "idle",
    "slow_walk",
    "walk",
    "hand_crawling",
    "walk_boxing",
    "elbow_crawling",
    "stealth_walk",
]
_WALK_MASK = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
_CRAWL_MASK = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
_CLIP_SPECS: list[list[int] | None] = [
    _WALK_MASK,
    _WALK_MASK,
    _WALK_MASK,
    _WALK_MASK,
    _CRAWL_MASK,
    _CRAWL_MASK,
    _WALK_MASK,
]


class StubAgent:
    """Minimal :class:`~strands_robots.policies.motionbricks.MotionAgent` stub.

    Returns a deterministic ``qpos`` (root + a per-joint ramp) and records the
    control signals + controller_dt it was fed, so the mapping and style
    plumbing are observable without the real generator.
    """

    clip_keys = _CLIP_KEYS
    clip_token_specs = _CLIP_SPECS
    min_token = 6
    max_token = 16

    def __init__(self, njoints: int = 29) -> None:
        self.njoints = njoints
        self.calls: list[tuple[dict[str, Any], float]] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def next_qpos(self, control_signals: dict[str, Any], controller_dt: float) -> np.ndarray:
        self.calls.append((control_signals, controller_dt))
        q = np.zeros(7 + self.njoints, dtype=np.float64)
        q[7:] = np.arange(self.njoints, dtype=np.float64) * 0.01
        return q


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_from_dict_defaults() -> None:
    cfg = MotionBricksConfig.from_dict({"result_dir": "out"})
    assert cfg.result_dir == "out"
    assert cfg.clips == "G1"
    assert cfg.style == "walk"
    assert cfg.fps == 30
    assert cfg.device == "cuda"
    assert cfg.speed_scale == (1.0, 1.0)


def test_config_controller_dt_matches_upstream() -> None:
    cfg = MotionBricksConfig.from_dict({"result_dir": "out", "fps": 30, "generate_dt": 2.0})
    assert cfg.controller_dt == pytest.approx((NUM_REGEN_FRAMES / 30.0) * 2.0)


def test_config_requires_result_dir() -> None:
    with pytest.raises(ValueError, match="result_dir"):
        MotionBricksConfig.from_dict({"fps": 30})


@pytest.mark.parametrize(
    "bad",
    [
        {"result_dir": "out", "fps": 0},
        {"result_dir": "out", "generate_dt": 0.0},
        {"result_dir": "out", "speed_scale": (2.0, 1.0)},
        {"result_dir": "out", "speed_scale": (1.0,)},
        {"result_dir": "out", "style": 3.5},
    ],
)
def test_config_fail_fast_on_bad_knobs(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MotionBricksConfig.from_dict(bad)


def test_config_from_file_roundtrip(tmp_path: Any) -> None:
    p = tmp_path / "mb.json"
    p.write_text(json.dumps({"result_dir": "out", "style": "stealth_walk", "device": "cpu"}))
    cfg = MotionBricksConfig.from_file(str(p))
    assert cfg.style == "stealth_walk"
    assert cfg.device == "cpu"


def test_config_from_file_missing(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError):
        MotionBricksConfig.from_file(str(tmp_path / "nope.json"))


def test_config_rejects_empty_result_dir() -> None:
    # result_dir is the path to the checkpoint tree; an empty string would
    # silently misbehave deep in the generator, so __post_init__ fails fast.
    with pytest.raises(ValueError, match="result_dir must be a non-empty path"):
        MotionBricksConfig(result_dir="")


def test_config_from_file_rejects_non_json_extension(tmp_path: Any) -> None:
    p = tmp_path / "mb.yaml"
    p.write_text('{"result_dir": "out"}')
    with pytest.raises(ValueError, match="unsupported extension"):
        MotionBricksConfig.from_file(str(p))


def test_config_from_file_rejects_invalid_json(tmp_path: Any) -> None:
    p = tmp_path / "mb.json"
    p.write_text("{not valid json,,,}")
    with pytest.raises(ValueError, match="not valid JSON"):
        MotionBricksConfig.from_file(str(p))


def test_config_from_file_rejects_non_mapping(tmp_path: Any) -> None:
    # Valid JSON, but a top-level array/scalar is not a config mapping.
    p = tmp_path / "mb.json"
    p.write_text('["result_dir", "out"]')
    with pytest.raises(ValueError, match="must contain a mapping"):
        MotionBricksConfig.from_file(str(p))


# ---------------------------------------------------------------------------
# Style resolution + control-signal assembly
# ---------------------------------------------------------------------------
def test_resolve_mode_by_name_and_index() -> None:
    assert resolve_mode("walk", _CLIP_KEYS) == 2
    assert resolve_mode("stealth_walk", _CLIP_KEYS) == 6
    assert resolve_mode(4, _CLIP_KEYS) == 4


def test_resolve_mode_rejects_unknown_and_out_of_range() -> None:
    with pytest.raises(ValueError, match="not a known mode"):
        resolve_mode("backflip", _CLIP_KEYS)
    with pytest.raises(ValueError, match="out of range"):
        resolve_mode(99, _CLIP_KEYS)
    with pytest.raises(ValueError, match="bool"):
        resolve_mode(True, _CLIP_KEYS)


def test_allowed_pred_num_tokens_explicit_and_default() -> None:
    assert allowed_pred_num_tokens(2, _CLIP_SPECS, 6, 16) == _WALK_MASK
    # A mode with no explicit mask -> all-ones of width max-min+1.
    assert allowed_pred_num_tokens(0, [None], 6, 16) == [1] * 11


def test_build_control_signals_default_direction() -> None:
    cs = build_control_signals(mode_idx=2, clip_token_specs=_CLIP_SPECS, min_token=6, max_token=16, kwargs={})
    assert cs["mode"] == 2
    assert cs["movement_direction"] == [1.0, 0.0, 0.0]
    assert cs["facing_direction"] == [1.0, 0.0, 0.0]
    assert cs["allowed_pred_num_tokens"] == _WALK_MASK


def test_build_control_signals_velocity_and_heading() -> None:
    cs = build_control_signals(
        mode_idx=2,
        clip_token_specs=_CLIP_SPECS,
        min_token=6,
        max_token=16,
        kwargs={"target_velocity": [0.0, 2.0], "target_heading": [1.0, 0.0]},
    )
    # Movement normalised to +y, facing to +x; z dropped.
    assert cs["movement_direction"] == pytest.approx([0.0, 1.0, 0.0])
    assert cs["facing_direction"] == pytest.approx([1.0, 0.0, 0.0])


def test_resolve_mode_rejects_non_int_non_str() -> None:
    # A value that is neither a mode index (int) nor a mode name (str) is a
    # caller error, not a silent no-op: the wrong type surfaces with the
    # accepted forms so the call can be corrected (AGENTS.md: raise on fatal).
    with pytest.raises(ValueError, match="int index or str name"):
        resolve_mode(1.5, _CLIP_KEYS)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="int index or str name"):
        resolve_mode([2], _CLIP_KEYS)  # type: ignore[arg-type]


def test_allowed_pred_num_tokens_rejects_bad_mode_and_range() -> None:
    # An out-of-range mode index and a degenerate token range are both fatal:
    # a wrong horizon mask would silently corrupt the generation window, so
    # each raises rather than clamping to a plausible-but-wrong value.
    with pytest.raises(ValueError, match="out of range"):
        allowed_pred_num_tokens(len(_CLIP_SPECS), _CLIP_SPECS, 6, 16)
    with pytest.raises(ValueError, match="must be >= min_token"):
        allowed_pred_num_tokens(0, [None], 16, 6)


def test_build_control_signals_rejects_short_direction() -> None:
    # A single-component target_velocity cannot define a planar direction; the
    # builder rejects it instead of fabricating a NaN/garbage heading.
    with pytest.raises(ValueError, match="2 or 3 entries"):
        build_control_signals(
            mode_idx=2,
            clip_token_specs=_CLIP_SPECS,
            min_token=6,
            max_token=16,
            kwargs={"target_velocity": [1.0]},
        )


def test_build_control_signals_zero_velocity_falls_back_to_forward() -> None:
    # An all-zero command must not yield a NaN direction: it falls back to the
    # default forward (+x) heading for both movement and facing.
    cs = build_control_signals(
        mode_idx=2,
        clip_token_specs=_CLIP_SPECS,
        min_token=6,
        max_token=16,
        kwargs={"target_velocity": [0.0, 0.0]},
    )
    assert cs["movement_direction"] == [1.0, 0.0, 0.0]
    assert cs["facing_direction"] == [1.0, 0.0, 0.0]


def test_build_control_signals_heading_angle_sets_facing() -> None:
    # target_heading_angle drives facing independently of movement: a 90-degree
    # heading faces +y while the body still moves along the default +x.
    cs = build_control_signals(
        mode_idx=2,
        clip_token_specs=_CLIP_SPECS,
        min_token=6,
        max_token=16,
        kwargs={"target_heading_angle": math.pi / 2},
    )
    assert cs["movement_direction"] == [1.0, 0.0, 0.0]
    assert cs["facing_direction"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


# ---------------------------------------------------------------------------
# Policy via the stub seam
# ---------------------------------------------------------------------------
def test_policy_basic_props_and_action_keys() -> None:
    pol = MotionBricksPolicy(motion_agent=StubAgent(), style="walk")
    assert isinstance(pol, Policy)
    assert pol.provider_name == "motionbricks"
    assert pol.requires_images is False

    actions = pol.get_actions_sync({}, "")
    assert len(actions) == 1
    act = actions[0]
    # 29 joints, exact WBC ordering, python floats (no ndarray scalars).
    assert list(act.keys()) == list(WBC_G1_ALL_JOINTS)
    assert len(act) == 29
    assert all(isinstance(v, float) for v in act.values())
    # qpos[7:] ramp -> first joint 0.0, last 0.28.
    assert act["left_hip_pitch_joint"] == pytest.approx(0.0)
    assert act["right_wrist_yaw_joint"] == pytest.approx(0.28)


def test_policy_style_switching_forwarded_to_generator() -> None:
    stub = StubAgent()
    pol = MotionBricksPolicy(motion_agent=stub, style="walk")
    pol.get_actions_sync({}, "", style="stealth_walk")
    assert stub.calls[-1][0]["mode"] == 6
    pol.get_actions_sync({}, "", mode=4)  # int via the `mode` alias
    assert stub.calls[-1][0]["mode"] == 4
    # controller_dt forwarded matches the config default (8/30)*2.0.
    assert stub.calls[-1][1] == pytest.approx((NUM_REGEN_FRAMES / 30.0) * 2.0)


def test_policy_unknown_style_raises() -> None:
    pol = MotionBricksPolicy(motion_agent=StubAgent())
    with pytest.raises(ValueError, match="not a known mode"):
        pol.get_actions_sync({}, "", style="moonwalk")


def test_policy_reset_forwards_to_generator() -> None:
    stub = StubAgent()
    pol = MotionBricksPolicy(motion_agent=stub)
    pol.reset(seed=7)
    assert stub.reset_count == 1


def test_policy_short_qpos_raises() -> None:
    pol = MotionBricksPolicy(motion_agent=StubAgent(njoints=10))  # 7+10 < 7+29
    with pytest.raises(RuntimeError, match="qpos of length"):
        pol.get_actions_sync({}, "")


def test_policy_requires_config_or_agent() -> None:
    with pytest.raises(ValueError, match="config/result_dir.*motion_agent|motion_agent"):
        MotionBricksPolicy()


def test_policy_missing_checkpoint_dir_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Neutralise the optional-dep gate so the test is independent of whether the
    # `motionbricks` package happens to be installed in the test env.
    monkeypatch.setattr(mb_policy, "require_optional", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="result_dir not found"):
        MotionBricksPolicy(result_dir="/nonexistent/motionbricks/out")


def test_set_robot_state_keys_validates_g1_joints() -> None:
    pol = MotionBricksPolicy(motion_agent=StubAgent())
    full = ["floating_base_joint", *WBC_G1_ALL_JOINTS]
    pol.set_robot_state_keys(full)  # ok
    with pytest.raises(ValueError, match="missing expected G1 joints"):
        pol.set_robot_state_keys(["floating_base_joint", "left_hip_pitch_joint"])


# ---------------------------------------------------------------------------
# Factory + registry
# ---------------------------------------------------------------------------
def test_registry_lists_motionbricks() -> None:
    assert "motionbricks" in list_providers()


@pytest.mark.parametrize("name", ["motionbricks", "motion_bricks"])
def test_create_policy_roundtrip_via_seam(name: str) -> None:
    pol = create_policy(name, motion_agent=StubAgent())
    assert isinstance(pol, MotionBricksPolicy)
    assert pol.provider_name == "motionbricks"
    actions = pol.get_actions_sync({}, "", style="walk")
    assert len(actions[0]) == 29


def test_policy_last_qpos_exposes_full_body_pose() -> None:
    pol = MotionBricksPolicy(motion_agent=StubAgent())
    assert pol.last_qpos is None  # before first call
    pol.get_actions_sync({}, "")
    qpos = pol.last_qpos
    assert qpos is not None
    assert qpos.shape == (7 + 29,)  # root(7) + 29 joints
    # It is a copy: mutating the returned array does not corrupt policy state.
    qpos[0] = 123.0
    assert pol.last_qpos[0] != 123.0


# ---------------------------------------------------------------------------
# locomotion_style goal channel: a caller-supplied locomotion_style steers
# MotionBricks' gait.
#
# Callers pass a fixed SONIC locomotion style vocabulary via
# run_policy(policy_kwargs={"locomotion_style": ...}); MotionBricks names its
# clips differently. These pin that locomotion_style selects the matching clip.
# ---------------------------------------------------------------------------

# A clip set covering every mappable locomotion style (a superset of the
# upstream clip_holder_G1 names the bridge maps onto).
_FULL_CLIP_KEYS = [
    "idle",
    "walk",
    "walk_happy_dance",
    "stealth_walk",
    "injured_walk",
    "hand_crawling",
    "elbow_crawling",
    "walk_boxing",
]


class _FullClipAgent(StubAgent):
    """A stub whose clip set covers every mappable locomotion style."""

    clip_keys = _FULL_CLIP_KEYS
    clip_token_specs = [None] * len(_FULL_CLIP_KEYS)


def test_resolve_locomotion_style_maps_every_mappable_style() -> None:
    from strands_robots.policies.motionbricks import LOCOMOTION_STYLE_TO_G1_CLIP, resolve_locomotion_style

    for locomotion_style, clip in LOCOMOTION_STYLE_TO_G1_CLIP.items():
        assert resolve_locomotion_style(locomotion_style, _FULL_CLIP_KEYS) == clip


def test_resolve_locomotion_style_passes_native_clip_name_through() -> None:
    from strands_robots.policies.motionbricks import resolve_locomotion_style

    # A value already in the clip set is used as-is (a caller may emit a clip name).
    assert resolve_locomotion_style("stealth_walk", _FULL_CLIP_KEYS) == "stealth_walk"


def test_resolve_locomotion_style_override_map() -> None:
    from strands_robots.policies.motionbricks import resolve_locomotion_style

    # An override remaps a locomotion style onto a different clip in the set.
    assert resolve_locomotion_style("run", _FULL_CLIP_KEYS, {"run": "idle"}) == "idle"


def test_resolve_locomotion_style_unmapped_raises() -> None:
    from strands_robots.policies.motionbricks import resolve_locomotion_style

    # "kneeling" has no G1 clip: a clear error, never a silent wrong gait.
    with pytest.raises(ValueError, match="kneeling.*no MotionBricks clip mapping"):
        resolve_locomotion_style("kneeling", _FULL_CLIP_KEYS)


def test_resolve_locomotion_style_mapped_clip_absent_from_set_raises() -> None:
    from strands_robots.policies.motionbricks import resolve_locomotion_style

    # "boxing" maps to "walk_boxing", which a reduced clip set does not provide.
    with pytest.raises(ValueError, match="walk_boxing.*does not provide"):
        resolve_locomotion_style("boxing", ["idle", "walk"])


def test_resolve_locomotion_style_rejects_non_string() -> None:
    from strands_robots.policies.motionbricks import resolve_locomotion_style

    with pytest.raises(ValueError, match="must be a clip/style name"):
        resolve_locomotion_style(2, _FULL_CLIP_KEYS)  # type: ignore[arg-type]


def test_get_actions_consumes_locomotion_style() -> None:
    # locomotion_style selects the clip when no
    # explicit style=/mode= is pinned.
    agent = _FullClipAgent()
    pol = MotionBricksPolicy(motion_agent=agent, style="walk")
    pol.get_actions_sync({}, "", target_velocity=[0.5, 0.0, 0.0], locomotion_style="stealth")
    signals, _ = agent.calls[-1]
    assert agent.clip_keys[signals["mode"]] == "stealth_walk"


def test_explicit_style_overrides_locomotion_style() -> None:
    # An explicit style=/mode= pins the clip even when a locomotion_style is given.
    agent = _FullClipAgent()
    pol = MotionBricksPolicy(motion_agent=agent, style="walk")
    pol.get_actions_sync({}, "", style="walk_boxing", locomotion_style="stealth")
    signals, _ = agent.calls[-1]
    assert agent.clip_keys[signals["mode"]] == "walk_boxing"


def test_locomotion_style_falls_back_to_default_when_absent() -> None:
    # No style/mode/locomotion_style -> the configured default clip is used.
    agent = _FullClipAgent()
    pol = MotionBricksPolicy(motion_agent=agent, style="injured_walk")
    pol.get_actions_sync({}, "", target_velocity=[1.0, 0.0, 0.0])
    signals, _ = agent.calls[-1]
    assert agent.clip_keys[signals["mode"]] == "injured_walk"


def test_policy_style_map_override_applied() -> None:
    # A style_map on the policy remaps a locomotion style onto another clip.
    agent = _FullClipAgent()
    pol = MotionBricksPolicy(motion_agent=agent, style="walk", style_map={"run": "idle"})
    pol.get_actions_sync({}, "", locomotion_style="run")
    signals, _ = agent.calls[-1]
    assert agent.clip_keys[signals["mode"]] == "idle"


def test_policy_kwargs_goal_channel_drives_motionbricks_end_to_end() -> None:
    # Full path: the well-known policy_kwargs goal channel feeds get_actions, the
    # way run_policy(policy_kwargs={...}) forwards it verbatim each tick.
    agent = _FullClipAgent()
    pol = MotionBricksPolicy(motion_agent=agent, style="walk")
    kwargs = {"target_velocity": [0.4, 0.0, 0.0], "locomotion_style": "boxing"}
    pol.get_actions_sync({}, "", **kwargs)
    signals, _ = agent.calls[-1]
    assert agent.clip_keys[signals["mode"]] == "walk_boxing"
    # target_velocity still composes (movement direction is the +x unit vector).
    assert signals["movement_direction"][0] == pytest.approx(1.0)


def test_locomotion_styles_all_mapped_or_intentionally_absent() -> None:
    # Keep the bridge in sync with the accepted vocabulary: every locomotion
    # style is either mapped to a clip or the one documented gap (kneeling has no
    # G1 clip). LOCOMOTION_STYLES is owned by MotionBricks (no external module).
    from strands_robots.policies.motionbricks import (
        LOCOMOTION_STYLE_TO_G1_CLIP,
        LOCOMOTION_STYLES,
    )

    assert set(LOCOMOTION_STYLE_TO_G1_CLIP) | {"kneeling"} == set(LOCOMOTION_STYLES)


def test_config_style_map_roundtrip_and_validation() -> None:
    cfg = MotionBricksConfig.from_dict({"result_dir": "out", "style_map": {"run": "idle"}})
    assert cfg.style_map == {"run": "idle"}
    with pytest.raises(ValueError, match="style_map must be a dict"):
        MotionBricksConfig.from_dict({"result_dir": "out", "style_map": {"run": 3}})


def test_config_style_map_feeds_policy() -> None:
    agent = _FullClipAgent()
    cfg = MotionBricksConfig.from_dict({"result_dir": "out", "style_map": {"run": "idle"}})
    pol = MotionBricksPolicy(config=cfg, motion_agent=agent)
    pol.get_actions_sync({}, "", locomotion_style="run")
    signals, _ = agent.calls[-1]
    assert agent.clip_keys[signals["mode"]] == "idle"


# ---------------------------------------------------------------------------
# Config-input normalisation + constructor seams
# ---------------------------------------------------------------------------
class TestConfigInputNormalisation:
    """The constructor accepts a MotionBricksConfig, a dict, a JSON path, or a
    ``result_dir`` shortcut, exposes the resolved config via ``.config``, and
    rejects anything else.

    Verified through the public constructor (the stub-agent seam keeps it
    CPU-only, no checkpoints): the ``config=`` input is polymorphic, so the
    normalisation contract is what callers actually depend on.
    """

    def test_dict_config_resolved_and_exposed(self) -> None:
        pol = MotionBricksPolicy(config={"result_dir": "out", "style": "walk"}, motion_agent=StubAgent())
        assert isinstance(pol.config, MotionBricksConfig)
        assert pol.config.result_dir == "out"
        assert pol.config.style == "walk"

    def test_json_path_config_resolved(self, tmp_path: Any) -> None:
        p = tmp_path / "mb.json"
        p.write_text(json.dumps({"result_dir": "out", "device": "cpu"}))
        pol = MotionBricksPolicy(config=str(p), motion_agent=StubAgent())
        assert isinstance(pol.config, MotionBricksConfig)
        assert pol.config.device == "cpu"

    def test_result_dir_shortcut_applies_style_and_device(self) -> None:
        pol = MotionBricksPolicy(result_dir="out", style="stealth_walk", device="cpu", motion_agent=StubAgent())
        assert isinstance(pol.config, MotionBricksConfig)
        assert pol.config.result_dir == "out"
        assert pol.config.style == "stealth_walk"
        assert pol.config.device == "cpu"

    def test_no_config_leaves_config_none(self) -> None:
        pol = MotionBricksPolicy(motion_agent=StubAgent())
        assert pol.config is None

    def test_unsupported_config_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported config type"):
            MotionBricksPolicy(config=123, motion_agent=StubAgent())  # type: ignore[arg-type]

    def test_unknown_constructor_kwargs_are_ignored(self) -> None:
        # Unknown kwargs are tolerated (logged at debug), so a caller passing a
        # forward-compat option does not crash the policy.
        pol = MotionBricksPolicy(motion_agent=StubAgent(), some_future_option=True)
        assert pol.get_actions_sync({}, "")  # still functional


def test_default_target_velocity_injected_when_call_omits_it() -> None:
    # A constructor ``target_velocity`` is the standing movement direction used
    # when a get_actions call passes none; an explicit per-call value overrides.
    stub = StubAgent()
    pol = MotionBricksPolicy(motion_agent=stub, target_velocity=[0.0, 2.0])
    pol.get_actions_sync({}, "")
    # +y default normalised into the control signal.
    assert stub.calls[-1][0]["movement_direction"] == pytest.approx([0.0, 1.0, 0.0])
    # Per-call target_velocity wins over the constructor default.
    pol.get_actions_sync({}, "", target_velocity=[2.0, 0.0])
    assert stub.calls[-1][0]["movement_direction"] == pytest.approx([1.0, 0.0, 0.0])
