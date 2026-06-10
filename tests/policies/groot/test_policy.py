"""Tests for Gr00tPolicy - unit tests WITHOUT Isaac-GR00T installed."""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

msgpack = pytest.importorskip("msgpack", reason="msgpack not installed - pip install 'strands-robots[groot-service]'")
zmq = pytest.importorskip("zmq", reason="zmq not installed - pip install 'strands-robots[groot-service]'")

# All tests in this file require groot-service extras
pytestmark = pytest.mark.skipif(
    not msgpack or not zmq,
    reason="groot-service extras not installed",
)

from strands_robots.policies.groot import DATA_CONFIG_MAP, ActionMapping, Gr00tPolicy, ObservationMapping  # noqa: E402
from strands_robots.policies.groot.data_config import Gr00tDataConfig  # noqa: E402
from strands_robots.policies.groot.policy import (  # noqa: E402
    _auto_infer_action_mapping,
    _auto_infer_observation_mapping,
    _detect_groot_version,
    _parse_action_mapping,
    _parse_observation_mapping,
    _reference_video_shape,
    _to_state_batch,
    _to_video_batch,
)

# (section)
# Helpers
# (section)


_KNOWN_DOF = {
    "single_arm": 5,
    "gripper": 1,
    "webcam": None,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 6,
    "right_hand": 6,
    "waist": 3,
    "ego_view_bg_crop_pad_res256_freq20": None,
}


def _mock_mmc(video_keys=None, state_keys=None, action_keys=None, language_keys=None):
    configs = {}
    for name, keys in [
        ("video", video_keys or ["webcam"]),
        ("state", state_keys or ["single_arm", "gripper"]),
        ("action", action_keys or ["single_arm", "gripper"]),
        ("language", language_keys or ["annotation.human.task_description"]),
    ]:
        mc = MagicMock()
        mc.modality_keys = keys
        configs[name] = mc
    return configs


SO100_MMC = _mock_mmc()
GR1_MMC = _mock_mmc(
    video_keys=["ego_view_bg_crop_pad_res256_freq20"],
    state_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
    action_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
    language_keys=["task"],
)


def _make_policy(data_config="so100", version="n1.6", obs_mapping=None, action_mapping=None, mmc=None):
    p = Gr00tPolicy.__new__(Gr00tPolicy)
    p.data_config = DATA_CONFIG_MAP[data_config]
    p.data_config_name = data_config
    p._mode = "local"
    p._groot_version = version
    p._strict = False
    p._client = None
    p._local_policy = MagicMock()
    p._raw_obs_mapping = None
    p._raw_action_mapping = None
    p._language_key_override = None

    mc = mmc or SO100_MMC
    p._local_policy.modality_configs = mc  # Direct N16Policy
    p._local_policy.policy.modality_configs = mc  # Wrapped
    p._local_policy.modality_config = mc  # N1.5

    # Simulate discovered DOF
    p._model_state_dof = {k: _KNOWN_DOF[k] for k in mc["state"].modality_keys if _KNOWN_DOF.get(k) is not None}
    p._obs_mapping = obs_mapping
    p._action_mapping = action_mapping
    return p


def _require_real_torch():
    """Return real torch, or skip when the numpy-backed torch mock is active.

    The RNG-reproducibility tests below assert that ``policy.reset(seed=...)``
    reseeds torch deterministically. That contract is only observable against
    real torch: the test-suite torch mock (tests/mocks/torch_mock.py) has a
    no-op ``manual_seed`` and its tensors have no ``tolist``, so the assertion
    is both meaningless and crash-prone under the mock. ``importorskip`` alone
    is insufficient because the mock registers a ``torch`` module in
    ``sys.modules`` (so the import succeeds); the mock never sets
    ``__version__``, which is the discriminator used here.
    """
    torch = pytest.importorskip("torch", reason="torch not installed")
    if not hasattr(torch, "__version__"):
        pytest.skip("requires real torch (mock active); RNG reproducibility is real-torch-only")
    return torch


# (section)
# Construction
# (section)


class TestConstruction:
    def test_service_mode(self):
        p = Gr00tPolicy()
        assert p._mode == "service" and p._client is not None and p.provider_name == "groot"

    def test_config_name(self):
        assert Gr00tPolicy(data_config="so100").data_config_name == "so100"

    def test_config_object(self):
        cfg = Gr00tDataConfig(
            name="t",
            video_keys=["video.c"],
            state_keys=["state.a"],
            action_keys=["action.a"],
            language_keys=["annotation.human.task_description"],
        )
        assert Gr00tPolicy(data_config=cfg).data_config is cfg

    def test_strict(self):
        assert Gr00tPolicy()._strict is False
        assert Gr00tPolicy(strict=True)._strict is True

    def test_api_token(self):
        assert Gr00tPolicy(api_token="t")._client.api_token == "t"

    def test_api_token_from_env(self, monkeypatch):
        """GROOT_API_TOKEN env var is used when api_token param is None."""
        monkeypatch.setenv("GROOT_API_TOKEN", "env-secret")
        p = Gr00tPolicy()
        assert p._client.api_token == "env-secret"

    def test_api_token_param_overrides_env(self, monkeypatch):
        """Explicit api_token param takes precedence over env var."""
        monkeypatch.setenv("GROOT_API_TOKEN", "env-secret")
        p = Gr00tPolicy(api_token="explicit")
        assert p._client.api_token == "explicit"

    def test_api_token_none_without_env(self, monkeypatch):
        """When no env var and no param, api_token is None."""
        monkeypatch.delenv("GROOT_API_TOKEN", raising=False)
        p = Gr00tPolicy()
        assert p._client.api_token is None

    def test_unknown_config(self):
        with pytest.raises(ValueError):
            Gr00tPolicy(data_config="nope")

    def test_no_isaac(self):
        p = _make_policy()
        p._groot_version = None
        with pytest.raises(ImportError):
            p._load_local_policy("/f", "NEW_EMBODIMENT", "cpu")

    def test_all_configs(self):
        for name in DATA_CONFIG_MAP:
            assert Gr00tPolicy(data_config=name)._mode == "service"

    def test_no_denoising_steps_param(self):
        """denoising_steps was removed from __init__ - kwargs swallows it."""
        p = Gr00tPolicy(denoising_steps=8)
        assert p._mode == "service"  # no error, just ignored via **kwargs

    def test_set_robot_state_keys_is_noop(self):
        """set_robot_state_keys is a documented no-op."""
        p = Gr00tPolicy()
        p.set_robot_state_keys(["a", "b"])  # should not raise


# (section)
# Version detection
# (section)


class TestVersion:
    def test_cached(self):
        import strands_robots.policies.groot.policy as pm

        orig = pm._GROOT_VERSION
        pm._GROOT_VERSION = "cached"
        try:
            assert _detect_groot_version() == "cached"
        finally:
            pm._GROOT_VERSION = orig

    def test_force_redetect(self):
        """force=True should bypass cache and re-detect."""
        import strands_robots.policies.groot.policy as pm

        orig = pm._GROOT_VERSION
        pm._GROOT_VERSION = "stale"
        try:
            with patch("importlib.util.find_spec", return_value=None):
                result = _detect_groot_version(force=True)
            # Should have re-detected (None since find_spec returns None)
            assert result is None
            assert pm._GROOT_VERSION is None
        finally:
            pm._GROOT_VERSION = orig

    def test_force_false_uses_cache(self):
        """force=False (default) should use cached value."""
        import strands_robots.policies.groot.policy as pm

        orig = pm._GROOT_VERSION
        pm._GROOT_VERSION = "n1.6"
        try:
            assert _detect_groot_version(force=False) == "n1.6"
        finally:
            pm._GROOT_VERSION = orig

    def test_detect_n17(self):
        """N1.7 is detected when the ``gr00t.model.gr00t_n1d7`` subpackage exists.

        N1.6 and N1.7 share ``gr00t.policy.gr00t_policy`` - so we need a
        version-specific probe.  ``gr00t_n1d7`` was introduced in N1.7.
        """
        import strands_robots.policies.groot.policy as pm

        orig = pm._GROOT_VERSION
        pm._GROOT_VERSION = None
        try:

            def fake_find_spec(name: str):
                if name == "gr00t.model.gr00t_n1d7":
                    return MagicMock()  # N1.7 subpackage found
                if name == "gr00t.policy.gr00t_policy":
                    return MagicMock()  # Also present in N1.7, but N1.7 wins first
                return None

            with patch("importlib.util.find_spec", side_effect=fake_find_spec):
                assert _detect_groot_version(force=True) == "n1.7"
            assert pm._GROOT_VERSION == "n1.7"
        finally:
            pm._GROOT_VERSION = orig

    def test_detect_n16_when_no_n17_subpackage(self):
        """N1.6 is reported when ``gr00t.policy.gr00t_policy`` exists but no N1.7 subpackage."""
        import strands_robots.policies.groot.policy as pm

        orig = pm._GROOT_VERSION
        pm._GROOT_VERSION = None
        try:

            def fake_find_spec(name: str):
                if name == "gr00t.model.gr00t_n1d7":
                    return None  # N1.7 subpackage absent
                if name == "gr00t.policy.gr00t_policy":
                    return MagicMock()  # N1.6 entry point present
                return None

            with patch("importlib.util.find_spec", side_effect=fake_find_spec):
                assert _detect_groot_version(force=True) == "n1.6"
            assert pm._GROOT_VERSION == "n1.6"
        finally:
            pm._GROOT_VERSION = orig

    def test_detect_order_prefers_n17(self):
        """When both N1.7 and N1.6 probes would succeed, N1.7 must win."""
        import strands_robots.policies.groot.policy as pm

        orig = pm._GROOT_VERSION
        pm._GROOT_VERSION = None
        try:
            # All three probes return a spec - N1.7 must come first.
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert _detect_groot_version(force=True) == "n1.7"
        finally:
            pm._GROOT_VERSION = orig

    def test_detect_n15_legacy_only(self):
        """Only ``gr00t.model.policy`` => N1.5."""
        import strands_robots.policies.groot.policy as pm

        orig = pm._GROOT_VERSION
        pm._GROOT_VERSION = None
        try:

            def fake_find_spec(name: str):
                if name == "gr00t.model.policy":
                    return MagicMock()
                return None

            with patch("importlib.util.find_spec", side_effect=fake_find_spec):
                assert _detect_groot_version(force=True) == "n1.5"
        finally:
            pm._GROOT_VERSION = orig


# (section)
# ObservationMapping
# (section)


class TestObsMapping:
    def test_empty(self):
        m = ObservationMapping()
        assert m.video == {} and m.state == {} and m.language_key == "task"

    def test_frozen(self):
        with pytest.raises(AttributeError):
            ObservationMapping().language_key = "x"

    def test_validate_ok(self):
        ObservationMapping(
            video={"c": "ego_view_bg_crop_pad_res256_freq20"},
            state={"j": "left_arm"},
            language_key="task",
        ).validate(GR1_MMC)

    def test_bad_video(self):
        with pytest.raises(ValueError, match="video"):
            ObservationMapping(video={"c": "nope"}).validate(GR1_MMC)

    def test_bad_state(self):
        with pytest.raises(ValueError, match="state"):
            ObservationMapping(state={"j": "nope"}, language_key="task").validate(GR1_MMC)

    def test_bad_lang(self):
        with pytest.raises(ValueError, match="language"):
            ObservationMapping(language_key="nope").validate(GR1_MMC)


# (section)
# ActionMapping
# (section)


class TestActionMapping:
    def test_empty(self):
        assert ActionMapping().actions == {}

    def test_validate_ok(self):
        ActionMapping(actions={"left_arm": "j"}).validate(GR1_MMC)

    def test_bad(self):
        with pytest.raises(ValueError):
            ActionMapping(actions={"nope": "j"}).validate(GR1_MMC)


# (section)
# Parsing
# (section)


class TestParsing:
    def test_obs(self):
        m = _parse_observation_mapping(
            {"cam": "video.ego_view_bg_crop_pad_res256_freq20", "j": "state.left_arm"},
            GR1_MMC,
        )
        assert m.video == {"cam": "ego_view_bg_crop_pad_res256_freq20"}
        assert m.state == {"j": "left_arm"}
        assert m.language_key == "task"

    def test_obs_default_lang_without_mmc(self):
        """Without modality_configs, language_key defaults to 'task'."""
        m = _parse_observation_mapping({"c": "video.cam"})
        assert m.language_key == "task"

    def test_bad_prefix(self):
        with pytest.raises(ValueError, match="must start with"):
            _parse_observation_mapping({"x": "audio.mic"})

    def test_action(self):
        m = _parse_action_mapping({"action.left_arm": "j", "action.left_hand": "g"})
        assert m.actions == {"left_arm": "j", "left_hand": "g"}


# (section)
# Auto-inference
# (section)


class TestAutoInfer:
    def test_obs_exact(self):
        m = _auto_infer_observation_mapping(DATA_CONFIG_MAP["so100"], SO100_MMC)
        assert m.video == {"webcam": "webcam"}
        assert m.state["single_arm"] == "single_arm"

    def test_obs_positional(self):
        m = _auto_infer_observation_mapping(DATA_CONFIG_MAP["so100"], GR1_MMC)
        assert m.video.get("webcam") == "ego_view_bg_crop_pad_res256_freq20"

    def test_action_exact(self):
        m = _auto_infer_action_mapping(DATA_CONFIG_MAP["so100"], SO100_MMC)
        assert m.actions["single_arm"] == "single_arm"


# (section)
# Shape helpers
# (section)


class TestShapes:
    def test_video_3d(self):
        assert _to_video_batch(np.zeros((64, 64, 3))).shape == (1, 1, 64, 64, 3)

    def test_video_4d(self):
        assert _to_video_batch(np.zeros((3, 64, 64, 3))).shape == (1, 3, 64, 64, 3)

    def test_video_5d_passthrough(self):
        assert _to_video_batch(np.zeros((1, 1, 64, 64, 3))).shape == (1, 1, 64, 64, 3)

    def test_state_1d(self):
        r = _to_state_batch(np.zeros(5))
        assert r.shape == (1, 1, 5) and r.dtype == np.float32

    def test_state_2d(self):
        assert _to_state_batch(np.zeros((3, 5))).shape == (1, 3, 5)

    def test_state_from_list(self):
        r = _to_state_batch([1.0, 2.0, 3.0])
        assert r.shape == (1, 1, 3) and r.dtype == np.float32

    def test_ref_from_mapped_video_keys(self):
        """Should only look at keys in the video_keys set."""
        obs = {
            "cam": np.zeros((128, 128, 3)),
            "state_3d": np.zeros((10, 10, 3)),  # 3D state - should NOT match
        }
        assert _reference_video_shape(obs, video_keys={"cam"}) == (128, 128, 3)

    def test_ref_ignores_non_video_keys(self):
        """State array with shape[-1]==3 should not be picked when video_keys given."""
        obs = {"state_vals": np.zeros((10, 10, 3))}
        assert _reference_video_shape(obs, video_keys={"cam"}) == (256, 256, 3)

    def test_ref_default(self):
        assert _reference_video_shape({"j": np.zeros(5)}) == (256, 256, 3)

    def test_ref_legacy_heuristic_when_no_video_keys(self):
        """Without video_keys, falls back to heuristic scan."""
        obs = {"c": np.zeros((128, 128, 3))}
        assert _reference_video_shape(obs, video_keys=None) == (128, 128, 3)


# (section)
# _prepare_observation - nested dict format
# (section)


class TestPrepareObs:
    def test_nested_structure(self):
        p = _make_policy(
            obs_mapping=ObservationMapping(
                video={"cam": "ego_view_bg_crop_pad_res256_freq20"},
                state={"j": "left_arm", "g": "left_hand"},
                language_key="task",
            ),
            mmc=GR1_MMC,
        )
        b = p._prepare_observation(
            {
                "cam": np.zeros((256, 256, 3), dtype=np.uint8),
                "j": np.zeros(7),
                "g": np.zeros(6),
            },
            "pick cube",
        )
        assert "video" in b and "state" in b and "language" in b
        assert b["video"]["ego_view_bg_crop_pad_res256_freq20"].shape == (1, 1, 256, 256, 3)
        assert b["state"]["left_arm"].shape == (1, 1, 7)
        assert b["language"]["task"] == [["pick cube"]]

    def test_zero_fills_unmapped_state(self):
        p = _make_policy(
            obs_mapping=ObservationMapping(
                video={"cam": "ego_view_bg_crop_pad_res256_freq20"},
                state={"j": "left_arm"},
                language_key="task",
            ),
            mmc=GR1_MMC,
        )
        b = p._prepare_observation({"cam": np.zeros((64, 64, 3), dtype=np.uint8), "j": np.zeros(7)}, "t")
        assert "right_arm" in b["state"]
        np.testing.assert_array_equal(b["state"]["right_arm"][0, 0], np.zeros(7))
        assert "waist" in b["state"]
        np.testing.assert_array_equal(b["state"]["waist"][0, 0], np.zeros(3))

    def test_skips_zero_fill_unknown_dof(self):
        """When DOF is not discoverable, the key should be skipped."""
        p = _make_policy(
            obs_mapping=ObservationMapping(
                video={"cam": "webcam"}, state={"arm": "single_arm"}, language_key="annotation.human.task_description"
            ),
        )
        # Clear DOF for gripper - simulate unknown
        p._model_state_dof = {"single_arm": 5}
        b = p._prepare_observation({"cam": np.zeros((64, 64, 3), dtype=np.uint8), "arm": np.zeros(5)}, "t")
        # gripper DOF unknown → should NOT be in state dict
        assert "gripper" not in b["state"]
        assert "single_arm" in b["state"]


# (section)
# _unpack_actions
# (section)


class TestUnpackActions:
    def test_maps(self):
        p = _make_policy(action_mapping=ActionMapping(actions={"left_arm": "j", "left_hand": "g"}), mmc=GR1_MMC)
        acts = p._unpack_actions({"left_arm": np.ones((1, 16, 7)), "left_hand": np.ones((1, 16, 6))})
        assert len(acts) == 16 and acts[0]["j"].shape == (7,) and acts[0]["g"].shape == (6,)

    def test_unmapped(self):
        p = _make_policy(action_mapping=ActionMapping(actions={"left_arm": "j"}), mmc=GR1_MMC)
        acts = p._unpack_actions({"left_arm": np.ones((1, 4, 7)), "waist": np.ones((1, 4, 3))})
        assert "unmapped.waist" in acts[0]

    def test_empty(self):
        assert _make_policy(action_mapping=ActionMapping())._unpack_actions({}) == []


# (section)
# Full local flow
# (section)


class TestLocalFlow:
    def test_n16(self):
        p = _make_policy(
            obs_mapping=ObservationMapping(
                video={"cam": "ego_view_bg_crop_pad_res256_freq20"},
                state={"j": "left_arm", "g": "left_hand"},
                language_key="task",
            ),
            action_mapping=ActionMapping(actions={"left_arm": "j", "left_hand": "g"}),
            mmc=GR1_MMC,
        )
        p._local_policy.get_action.return_value = (
            {"left_arm": np.ones((1, 16, 7)), "left_hand": np.ones((1, 16, 6))},
            {},
        )
        acts = p._local_get_actions(
            {"cam": np.zeros((256, 256, 3), dtype=np.uint8), "j": np.zeros(7), "g": np.zeros(6)},
            "pick",
        )
        assert len(acts) == 16
        call = p._local_policy.get_action.call_args[0][0]
        assert "video" in call and "state" in call and "language" in call

    def test_n15(self):
        p = _make_policy(
            version="n1.5",
            obs_mapping=ObservationMapping(
                video={"cam": "webcam"},
                state={"arm": "single_arm", "grip": "gripper"},
                language_key="annotation.human.task_description",
            ),
            action_mapping=ActionMapping(actions={"single_arm": "arm", "gripper": "grip"}),
        )
        p._local_policy.get_action.return_value = {
            "single_arm": np.ones((1, 16, 5)),
            "gripper": np.ones((1, 16, 1)),
        }
        acts = p._local_get_actions(
            {"cam": np.zeros((64, 64, 3), dtype=np.uint8), "arm": np.zeros(5), "grip": np.array([0.5])},
            "t",
        )
        assert len(acts) == 16 and "arm" in acts[0]

    def test_bad_version(self):
        p = _make_policy(obs_mapping=ObservationMapping(), action_mapping=ActionMapping())
        p._groot_version = "n9.9"
        with pytest.raises(RuntimeError):
            p._local_get_actions({}, "t")


# (section)
# get_actions routing
# (section)


class TestGetActions:
    def test_local(self):
        p = _make_policy(
            obs_mapping=ObservationMapping(
                video={"cam": "webcam"},
                state={"arm": "single_arm"},
                language_key="annotation.human.task_description",
            ),
            action_mapping=ActionMapping(actions={"single_arm": "arm"}),
        )
        p._local_policy.get_action.return_value = ({"single_arm": np.ones((1, 4, 5))}, {})
        acts = asyncio.run(p.get_actions({"cam": np.zeros((64, 64, 3), dtype=np.uint8), "arm": np.zeros(5)}, "t"))
        assert len(acts) == 4

    def test_service(self):
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._client.get_action = MagicMock(
            return_value={
                "action.single_arm": np.ones((1, 16, 5)),
                "action.gripper": np.ones((1, 16, 1)),
            }
        )
        acts = asyncio.run(p.get_actions({"webcam": np.zeros((64, 64, 3), dtype=np.uint8)}, "t"))
        assert len(acts) == 16


class TestPolicyReset:
    """#187: ``Gr00tPolicy.reset(seed=...)`` forwards a per-episode seed
    to the GR00T inference server (via the standard ``reset`` endpoint
    that ``server_client.PolicyServer`` registers by default).

    Without this, ``set_eval_seed`` only seeds the client-side process,
    leaving the server's diffusion sampler RNG drifting across calls and
    breaking reproducibility — same input gives different actions on
    successive calls. Pin the surface so future refactors can't silently
    drop the per-episode reset that the bisection plan in #187 confirmed
    is required.
    """

    def test_service_reset_calls_endpoint_with_seed(self):
        """``policy.reset(seed=42)`` in SERVICE mode must call the server's
        ``reset`` endpoint with ``options={"seed": 42}`` so the server-side
        wrapper can apply the seed before subsequent ``get_action`` calls.
        """
        from unittest.mock import MagicMock

        p = Gr00tPolicy(data_config="libero_panda", host="localhost", port=19999)
        p._client.call_endpoint = MagicMock(return_value={})

        p.reset(seed=42)

        p._client.call_endpoint.assert_called_once()
        endpoint, payload = p._client.call_endpoint.call_args.args
        assert endpoint == "reset"
        assert payload == {"options": {"seed": 42}}

    def test_service_reset_without_seed_omits_options(self):
        """``policy.reset()`` (no seed) calls the endpoint with ``data=None``,
        so the server-side wrapper falls back to its default seed."""
        from unittest.mock import MagicMock

        p = Gr00tPolicy(data_config="libero_panda", host="localhost", port=19999)
        p._client.call_endpoint = MagicMock(return_value={})

        p.reset()

        p._client.call_endpoint.assert_called_once()
        endpoint, payload = p._client.call_endpoint.call_args.args
        assert endpoint == "reset"
        # When no seed is provided we forward None — lets server use its
        # initial seed (matches "no-op" semantics for servers that don't
        # implement seed-aware reset).
        assert payload is None

    def test_service_reset_swallows_server_errors(self, caplog):
        """A server that rejects ``reset`` (older NVIDIA images, wrappers
        without the patch) must not crash the eval. Reset is best-effort —
        log INFO and continue.
        """
        import logging as _logging
        from unittest.mock import MagicMock

        p = Gr00tPolicy(data_config="libero_panda", host="localhost", port=19999)
        p._client.call_endpoint = MagicMock(side_effect=RuntimeError("Server error: Unknown endpoint: reset"))

        with caplog.at_level(_logging.INFO, logger="strands_robots.policies.groot.policy"):
            p.reset(seed=42)  # must not raise

        # We logged the failure but continued. The test passes as long as
        # reset returns without an exception.
        assert any("did not accept reset" in r.getMessage() or "Server error" in r.getMessage() for r in caplog.records)

    def test_local_reset_reseeds_torch(self):
        """``policy.reset(seed=42)`` in LOCAL mode reseeds Python/NumPy/torch
        the same way ``set_eval_seed`` does, so the in-process diffusion
        sampler is deterministic across reset boundaries.
        """
        torch = _require_real_torch()

        # Construct a LOCAL-mode Gr00tPolicy via the test's __new__-bypass
        # fixture to skip model loading. The reset path doesn't depend on
        # _local_policy being a real model.
        from strands_robots.policies.groot.policy import Gr00tPolicy

        p = Gr00tPolicy.__new__(Gr00tPolicy)
        p._mode = "local"
        p._client = None
        p._groot_version = "n1.7"
        p.data_config_name = "libero_panda"

        # Reseeded twice with the same seed → same torch.randn draw.
        p.reset(seed=42)
        first = torch.randn(3).tolist()
        p.reset(seed=42)
        second = torch.randn(3).tolist()
        assert first == second, f"local reset(seed=42) must be reproducible; got {first} vs {second}"

    def test_local_reset_without_seed_is_noop(self):
        """``policy.reset()`` in LOCAL mode without a seed leaves torch RNG
        state untouched. Reset is a hint, not a hard reset; without an
        explicit seed there's nothing to apply.
        """
        torch = _require_real_torch()
        from strands_robots.policies.groot.policy import Gr00tPolicy

        p = Gr00tPolicy.__new__(Gr00tPolicy)
        p._mode = "local"
        p._client = None
        p._groot_version = "n1.7"
        p.data_config_name = "libero_panda"

        # Take a draw, then reset(no seed), then take another draw. They
        # should differ because the RNG advanced naturally.
        torch.manual_seed(42)
        first = torch.randn(3).tolist()
        p.reset()  # no seed → no-op
        second = torch.randn(3).tolist()
        assert first != second, f"reset() with no seed must NOT reseed; first={first}, second={second}"


# (section)
# Service observation + action unpack
# (section)


class TestServiceObs:
    def test_flat_keys(self):
        p = Gr00tPolicy(data_config="so100_dualcam", host="localhost", port=19999)
        obs = p._build_service_observation({"front": np.zeros((64, 64, 3)), "wrist": np.zeros((64, 64, 3))}, "t")
        assert "video.front" in obs and obs["video.front"].shape[0] == 1

    def test_language(self):
        obs = Gr00tPolicy(data_config="so100")._build_service_observation({}, "t")
        assert "annotation.human.task_description" in obs

    # GH #148 / Failure 2 - regressions for the N1.7 wire format.
    #
    # The N1.5 / N1.6 inference servers accept (B, ...) tensors. The N1.7
    # ``run_gr00t_server`` entrypoint adds an explicit time axis and rejects
    # float64 state, so video must be (B, T, H, W, C) and state must be
    # (B, T, D) float32.

    def test_n15_default_video_shape_is_4d(self):
        """Pre-fix behaviour preserved when groot_version != 'n1.7' (back-compat)."""
        p = Gr00tPolicy(data_config="so100_dualcam", host="localhost", port=19999)
        p._groot_version = None  # mimic env where gr00t isn't installed
        obs = p._build_service_observation({"front": np.zeros((64, 64, 3), dtype=np.uint8)}, "t")
        # (B=1, H=64, W=64, C=3) - no T axis.
        assert obs["video.front"].shape == (1, 64, 64, 3)

    def test_n17_video_shape_is_5d_with_T(self):
        """N1.7 servers require (B, T, H, W, C) - one extra leading axis."""
        p = Gr00tPolicy(data_config="so100_dualcam", host="localhost", port=19999)
        p._groot_version = "n1.7"
        obs = p._build_service_observation({"front": np.zeros((64, 64, 3), dtype=np.uint8)}, "t")
        # (B=1, T=1, H=64, W=64, C=3).
        assert obs["video.front"].shape == (1, 1, 64, 64, 3)

    def test_n17_state_is_float32_and_3d(self):
        """N1.7 server rejects float64 state and requires (B, T, D) shape."""
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._groot_version = "n1.7"
        # ``state.single_arm`` is on so100; provide a Python float so the
        # promotion-to-float32 path is exercised (not just np.float32 input).
        obs = p._build_service_observation({"single_arm": [0.1, 0.2, 0.3]}, "t")
        arr = obs["state.single_arm"]
        assert arr.dtype == np.float32
        # (B=1, T=1, D=3).
        assert arr.shape == (1, 1, 3)

    def test_n17_scalar_state_promoted_to_3d(self):
        """A 0-D / scalar joint reading must surface as (B=1, T=1, D=1) in n1.7."""
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._groot_version = "n1.7"
        obs = p._build_service_observation({"gripper": 0.5}, "t")
        arr = obs["state.gripper"]
        assert arr.dtype == np.float32
        assert arr.shape == (1, 1, 1)

    def test_n17_language_remains_b_length_list_str(self):
        """Language is matched against the batch axis - same shape as N1.5/6."""
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._groot_version = "n1.7"
        obs = p._build_service_observation({}, "pick the cube")
        v = obs["annotation.human.task_description"]
        assert v == ["pick the cube"]

    def test_n15_state_remains_2d(self):
        """Pre-fix shape preserved: (B, D), no T axis. Float32 dtype unchanged."""
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._groot_version = "n1.6"
        obs = p._build_service_observation({"gripper": 0.5}, "t")
        arr = obs["state.gripper"]
        assert arr.dtype == np.float32
        # (B=1, D=1) - no T axis on n1.6.
        assert arr.shape == (1, 1)


class TestServiceObsImageRotation180:
    """``Gr00tDataConfig.image_rotation_180`` triggers a 180-deg rotation of
    every video tensor at observation-build time (#168 round-7 bug H).

    The ``nvidia/GR00T-N1.7-LIBERO`` checkpoint was trained on data that
    Isaac-GR00T's ``examples/Libero/eval/utils.py:get_libero_image()``
    rotates 180 deg at preprocessing time. Without this rotation at eval
    time the policy sees every observation upside-down relative to its
    training distribution and the success rate collapses to 0.

    The flag is gated by data_config so other configs (so100, oxe_droid,
    etc.) don't pay any rotation cost - it's enabled in
    ``data_configs.json`` only on the ``libero_panda`` entry.
    """

    def test_libero_panda_rotates_video_180(self):
        """End-to-end: ``data_config='libero_panda'`` rotates ``image`` 180."""
        p = Gr00tPolicy(data_config="libero_panda", host="localhost", port=19999)
        p._groot_version = None  # n1.5/1.6 wire shape so we get (B, H, W, C)
        # Build an obviously-asymmetric image so a 180 rotation flips
        # corner intensities in a verifiable way.
        h, w = 16, 8
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        original_top_left = img[0, 0].copy()
        original_bottom_right = img[h - 1, w - 1].copy()

        obs = p._build_service_observation(
            {"image": img.copy(), "wrist_image": img.copy()},
            "pick the cube",
        )
        out_image = obs["video.image"]
        assert out_image.shape == (1, h, w, 3)  # (B, H, W, C)
        # After 180 rotation: top-left of output == bottom-right of input,
        # bottom-right of output == top-left of input.
        np.testing.assert_array_equal(out_image[0, 0, 0], original_bottom_right)
        np.testing.assert_array_equal(out_image[0, h - 1, w - 1], original_top_left)
        # Wrist channel must rotate too - otherwise GR00T-LIBERO sees a
        # mismatched-orientation pair.
        out_wrist = obs["video.wrist_image"]
        np.testing.assert_array_equal(out_wrist[0, 0, 0], original_bottom_right)

    def test_libero_panda_rotation_works_under_n17_wire_shape(self):
        """N1.7 servers add a leading time axis - rotation must apply
        BEFORE the newaxis fanout so the slice indexes the H/W axes,
        not the leading B/T axes."""
        p = Gr00tPolicy(data_config="libero_panda", host="localhost", port=19999)
        p._groot_version = "n1.7"
        h, w = 8, 8
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        original_top_left = img[0, 0].copy()
        obs = p._build_service_observation({"image": img.copy()}, "t")
        out = obs["video.image"]
        # (B=1, T=1, H, W, C)
        assert out.shape == (1, 1, h, w, 3)
        # Bottom-right of rotated == top-left of original.
        np.testing.assert_array_equal(out[0, 0, h - 1, w - 1], original_top_left)

    def test_so100_does_not_rotate_video(self):
        """Configs without ``image_rotation_180`` flag keep the legacy
        no-rotation behaviour. Pin so so100 / oxe_droid / etc. are
        unaffected by the new flag - the bug-H fix is gated, not global."""
        p = Gr00tPolicy(data_config="so100_dualcam", host="localhost", port=19999)
        p._groot_version = None
        h, w = 8, 8
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        original_top_left = img[0, 0].copy()
        obs = p._build_service_observation(
            {"front": img.copy(), "wrist": img.copy()},
            "t",
        )
        out = obs["video.front"]
        # No rotation: top-left should match original top-left.
        np.testing.assert_array_equal(out[0, 0, 0], original_top_left)

    def test_rotation_output_is_contiguous(self):
        """``np.ascontiguousarray`` materialises the flipped view as a
        fresh contiguous buffer. Required because msgpack serialisation
        and ``ndarray.tobytes()`` produce wrong bytes for non-contiguous
        views; reversed views are NOT C-contiguous."""
        p = Gr00tPolicy(data_config="libero_panda", host="localhost", port=19999)
        p._groot_version = None
        img = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        obs = p._build_service_observation({"image": img}, "t")
        out = obs["video.image"]
        assert out.flags["C_CONTIGUOUS"], "rotated image must be C-contiguous for serialisation"

    def test_rotation_does_not_mutate_input_obs(self):
        """The rotation builds a fresh ndarray; the original
        ``robot_obs[bare]`` must be unchanged. Required because the
        observation dict may be reused (e.g. recorded for replay,
        or passed to a second policy)."""
        p = Gr00tPolicy(data_config="libero_panda", host="localhost", port=19999)
        p._groot_version = None
        img = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        original_copy = img.copy()
        robot_obs = {"image": img}
        p._build_service_observation(robot_obs, "t")
        np.testing.assert_array_equal(robot_obs["image"], original_copy)

    def test_libero_panda_data_config_has_rotation_flag(self):
        """Sanity: the ``libero_panda`` entry in data_configs.json has
        ``image_rotation_180=True``. Without this the runtime check in
        ``_build_service_observation`` no-ops and bug H reappears."""
        from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP

        cfg = DATA_CONFIG_MAP["libero_panda"]
        assert cfg.image_rotation_180 is True

    def test_other_data_configs_default_to_no_rotation(self):
        """Spot-check that the flag is OFF for non-libero embodiments.
        If somebody accidentally sets it globally, GR00T policies on
        other robots would silently rotate their inputs and break."""
        from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP

        for name in ("so100", "so100_dualcam"):
            cfg = DATA_CONFIG_MAP[name]
            assert cfg.image_rotation_180 is False, f"{name} unexpectedly has rotation enabled"


class TestLocalObsImageRotation180:
    """#169: ``image_rotation_180`` is applied in LOCAL inference mode too,
    not just SERVICE mode.

    Pre-#169 the rotation lived only in ``_build_service_observation``,
    which made:

    - LOCAL + ``LiberoAdapter`` (V-flipped → OpenGL): policy got
      OpenGL convention images (no rotation), training expected
      training convention. Silent OOD.

    The fix moves the rotation into a shared helper called from BOTH
    LOCAL and SERVICE paths so both inference modes consistently see
    training-convention images regardless of transport.
    """

    def _libero_panda_local_policy(self):
        """Build a LOCAL-mode policy with libero_panda's rotation flag."""
        # Use _make_policy helper but with libero_panda's data_config
        # AND a custom obs_mapping that maps the test image keys.
        p = _make_policy(
            data_config="libero_panda",
            obs_mapping=ObservationMapping(
                video={"image": "video.image", "wrist_image": "video.wrist_image"},
                state={"x": "state.x"},
                language_key="annotation.human.action.task_description",
            ),
            mmc=_mock_mmc(
                video_keys=["video.image", "video.wrist_image"],
                state_keys=["state.x"],
                action_keys=["action.x"],
                language_keys=["annotation.human.action.task_description"],
            ),
        )
        return p

    def test_local_rotates_video_180_when_flag_is_true(self):
        """LOCAL mode applies the same rotation SERVICE mode does, when
        ``data_config.image_rotation_180`` is True (e.g. libero_panda).

        Sentinel: top-left pixel of input → bottom-right of output, and
        vice versa. This is a strict 180° rotation.
        """
        p = self._libero_panda_local_policy()
        h, w = 16, 8
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        original_top_left = img[0, 0].copy()
        original_bottom_right = img[h - 1, w - 1].copy()

        out = p._prepare_observation(
            {"image": img.copy(), "wrist_image": img.copy(), "x": 0.0},
            "pick the cube",
        )
        # video_dict shape is (1, 1, H, W, C) post-fanout
        out_image = out["video"]["video.image"]
        assert out_image.shape == (1, 1, h, w, 3)
        np.testing.assert_array_equal(out_image[0, 0, 0, 0], original_bottom_right)
        np.testing.assert_array_equal(out_image[0, 0, h - 1, w - 1], original_top_left)
        # Wrist channel rotates too.
        out_wrist = out["video"]["video.wrist_image"]
        np.testing.assert_array_equal(out_wrist[0, 0, 0, 0], original_bottom_right)

    def test_local_does_not_rotate_when_flag_is_false(self):
        """LOCAL mode skips rotation when the data_config flag is False
        (the default; e.g. so100). Pin so non-LIBERO embodiments don't
        get silently rotated by the new local-mode wiring."""
        p = _make_policy(
            data_config="so100",
            obs_mapping=ObservationMapping(
                video={"webcam": "webcam"},
                state={"single_arm": "single_arm"},
                language_key="annotation.human.task_description",
            ),
        )
        h, w = 8, 8
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        original_top_left = img[0, 0].copy()

        out = p._prepare_observation(
            {"webcam": img.copy(), "single_arm": np.zeros(6)},
            "t",
        )
        out_image = out["video"]["webcam"]
        # No rotation: top-left should match original top-left.
        np.testing.assert_array_equal(out_image[0, 0, 0, 0], original_top_left)

    def test_local_and_service_produce_identical_image_after_rotation(self):
        """#169 invariant: for the same input, LOCAL and SERVICE modes
        deliver the SAME image content to the policy after rotation.

        This is the strongest test that the two paths can't drift again.
        Prior to #169, LOCAL skipped rotation while SERVICE applied it,
        so the same observation produced different model inputs depending
        on transport — exactly the bug class issue #169 covers."""
        p_local = self._libero_panda_local_policy()
        p_svc = Gr00tPolicy(data_config="libero_panda", host="localhost", port=19999)
        p_svc._groot_version = "n1.7"

        h, w = 16, 16
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)

        local_out = p_local._prepare_observation(
            {"image": img.copy(), "wrist_image": img.copy(), "x": 0.0},
            "t",
        )
        svc_out = p_svc._build_service_observation(
            {"image": img.copy(), "wrist_image": img.copy(), "x": 0.0},
            "t",
        )

        local_img = local_out["video"]["video.image"]  # (1, 1, H, W, C)
        svc_img = svc_out["video.image"]  # (1, 1, H, W, C) for n1.7
        # Squeeze leading axes for comparison.
        np.testing.assert_array_equal(local_img.squeeze(), svc_img.squeeze())

    def test_local_rotation_output_is_contiguous(self):
        """``np.ascontiguousarray`` materialises the rotated view as a
        fresh contiguous buffer. The local-mode pass-through to
        ``Gr00tPolicy.get_action`` doesn't strictly need contiguous
        memory (no msgpack), but consistency with service mode + any
        downstream code that calls ``.tobytes()`` matters."""
        p = self._libero_panda_local_policy()
        img = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        out = p._prepare_observation(
            {"image": img, "wrist_image": img, "x": 0.0},
            "t",
        )
        out_image = out["video"]["video.image"]
        assert out_image.flags["C_CONTIGUOUS"], "rotated image must be C-contiguous for serialisation"


class TestApplyImageRotation180Helper:
    """#169: the shared ``_apply_image_rotation_180_inplace`` helper
    handles 3D, 4D, and 5D inputs uniformly via negative-axis indexing.

    Pin the multi-shape behaviour so future refactors that change tensor
    shapes don't silently break either inference path.
    """

    def test_3d_input_rotated(self):
        """Bare ``(H, W, C)`` input rotates correctly."""
        from strands_robots.policies.groot.policy import _apply_image_rotation_180_inplace

        h, w = 4, 4
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        original_top_left = img[0, 0].copy()
        obs = {"video.image": img.copy()}
        _apply_image_rotation_180_inplace(obs, ["video.image"])
        np.testing.assert_array_equal(obs["video.image"][h - 1, w - 1], original_top_left)

    def test_4d_input_rotated_on_hw_axes(self):
        """``(B, H, W, C)`` rotates H/W (axes 1/2), leaves B alone."""
        from strands_robots.policies.groot.policy import _apply_image_rotation_180_inplace

        b, h, w = 2, 4, 4
        img = np.arange(b * h * w * 3, dtype=np.uint8).reshape(b, h, w, 3)
        # Put a unique marker at (b=1, top-left) so we can detect axis confusion.
        img[1, 0, 0] = [200, 201, 202]
        obs = {"video.image": img.copy()}
        _apply_image_rotation_180_inplace(obs, ["video.image"])
        # After 180° on H/W: marker at b=1 should now be at bottom-right.
        np.testing.assert_array_equal(obs["video.image"][1, h - 1, w - 1], [200, 201, 202])
        # And b=0 should NOT have that marker (would happen if B axis was flipped).
        assert not np.array_equal(obs["video.image"][0, h - 1, w - 1], [200, 201, 202])

    def test_5d_input_rotated_on_hw_axes(self):
        """``(B, T, H, W, C)`` rotates H/W (axes 2/3), leaves B/T alone."""
        from strands_robots.policies.groot.policy import _apply_image_rotation_180_inplace

        h, w = 4, 4
        img = np.arange(h * w * 3, dtype=np.uint8).reshape(1, 1, h, w, 3)
        original_top_left = img[0, 0, 0, 0].copy()
        obs = {"video.image": img.copy()}
        _apply_image_rotation_180_inplace(obs, ["video.image"])
        np.testing.assert_array_equal(obs["video.image"][0, 0, h - 1, w - 1], original_top_left)

    def test_skips_non_ndarray_values(self):
        """Best-effort: non-ndarray entries (e.g. zero-fills, mocks) are
        passed through unchanged. The helper is opt-in via the keys list,
        so a mistyped key produces a no-op rather than a crash."""
        from strands_robots.policies.groot.policy import _apply_image_rotation_180_inplace

        obs = {"video.image": "not an array", "video.wrist": [1, 2, 3]}
        _apply_image_rotation_180_inplace(obs, ["video.image", "video.wrist"])
        assert obs["video.image"] == "not an array"
        assert obs["video.wrist"] == [1, 2, 3]

    def test_skips_dim_lt_3(self):
        """``(H, W)`` shape (2D) without channel axis is skipped — the
        helper only rotates when there are ≥3 dims so the H/W axes are
        unambiguous via negative indexing."""
        from strands_robots.policies.groot.policy import _apply_image_rotation_180_inplace

        img_2d = np.arange(16, dtype=np.uint8).reshape(4, 4)
        obs = {"video.image": img_2d.copy()}
        _apply_image_rotation_180_inplace(obs, ["video.image"])
        # 2D unchanged.
        np.testing.assert_array_equal(obs["video.image"], img_2d)


class TestServiceUnpackWithMapping:
    """_unpack_service_actions should apply _action_mapping when available."""

    def test_with_mapping(self):
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._action_mapping = ActionMapping(actions={"single_arm": "joints", "gripper": "grip"})
        result = p._unpack_service_actions(
            {
                "action.single_arm": np.ones((1, 4, 5)),
                "action.gripper": np.ones((1, 4, 1)),
            }
        )
        assert len(result) == 4
        assert "joints" in result[0]
        assert "grip" in result[0]
        assert "single_arm" not in result[0]

    def test_with_mapping_unmapped_keys(self):
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._action_mapping = ActionMapping(actions={"single_arm": "joints"})
        result = p._unpack_service_actions(
            {
                "action.single_arm": np.ones((1, 4, 5)),
                "action.gripper": np.ones((1, 4, 1)),
            }
        )
        assert "joints" in result[0]
        assert "unmapped.gripper" in result[0]

    def test_without_mapping(self):
        """No mapping → bare model keys."""
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._action_mapping = None
        result = p._unpack_service_actions({"action.single_arm": np.ones((1, 4, 5))})
        assert "single_arm" in result[0]

    def test_empty_mapping(self):
        """Empty mapping → bare model keys."""
        p = Gr00tPolicy(data_config="so100", host="localhost", port=19999)
        p._action_mapping = ActionMapping(actions={})
        result = p._unpack_service_actions({"action.single_arm": np.ones((1, 2, 5))})
        assert "single_arm" in result[0]


# (section)
# Exports
# (section)


class TestExports:
    def test_all(self):
        import strands_robots.policies.groot as mod

        for name in mod.__all__:
            assert hasattr(mod, name)
