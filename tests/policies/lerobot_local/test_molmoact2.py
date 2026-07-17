"""Tests for ``strands_robots.policies.lerobot_local.molmoact2`` -- the
transformers-native MolmoAct2 load-path helpers used by ``LerobotLocalPolicy``
to support ``allenai/MolmoAct2-*`` checkpoints that have no lerobot draccus
``type`` key.

These tests are dependency-light: they exercise detection, norm-tag discovery,
and image-key derivation by stubbing ``config.json`` / ``norm_stats.json``
reads. They do NOT download the 21GB checkpoint or import lerobot's heavy
modeling code (build_policy is covered by the hardware/e2e validation).
"""

from __future__ import annotations

import json

import pytest

from strands_robots.policies.lerobot_local import molmoact2


def _require_lerobot_configs_features() -> None:
    """Skip unless ``lerobot.configs`` exposes the typed-feature API.

    ``build_policy`` imports ``FeatureType``/``PolicyFeature`` from
    ``lerobot.configs``; those symbols only exist in lerobot >= 0.5.2 (from
    source). A bare ``importorskip("lerobot")`` passes on the PyPI 0.5.1 wheel,
    where the import then raises and the test *errors* instead of *skips*. Gate
    on the symbol the code path actually needs so older lerobot cleanly skips.
    """
    pytest.importorskip("lerobot")
    configs = pytest.importorskip("lerobot.configs")
    if not hasattr(configs, "FeatureType") or not hasattr(configs, "PolicyFeature"):
        pytest.skip("lerobot.configs lacks FeatureType/PolicyFeature (needs lerobot >= 0.5.2)")


def test_is_molmoact2_explicit_type():
    """Explicit policy_type='molmoact2' short-circuits to True without any I/O."""
    assert molmoact2.is_molmoact2("anything/at-all", "molmoact2") is True
    assert molmoact2.is_molmoact2("anything/at-all", "MolmoAct2") is True  # case-insensitive


def test_is_molmoact2_empty_path_no_type():
    """No path and no type → not molmoact2 (avoids spurious hub calls)."""
    assert molmoact2.is_molmoact2("", None) is False


def test_is_molmoact2_explicit_nonmolmoact2_type_skips_hub_probe(monkeypatch):
    """An explicit non-molmoact2 policy_type is authoritative: return False
    WITHOUT any config.json probe.

    A declared lerobot-native type (``"act"``, ``"diffusion"``, ...) already
    tells us this is not a transformers-native MolmoAct2 checkpoint. The
    config.json read is a Hub round-trip, so falling through to it on every
    explicitly-typed load stalls (or fails) whenever the Hub is slow/unreachable
    -- e.g. loading ``LerobotLocalPolicy(pretrained_name_or_path="user/act_ckpt",
    policy_type="act")`` must not touch the network just to rule out molmoact2.
    """
    probed: list[str] = []

    def _tracking_read(path):
        probed.append(path)
        return {"type": "act"}

    monkeypatch.setattr(molmoact2, "_read_config_json", _tracking_read)

    for declared in ("act", "diffusion", "pi0", "smolvla"):
        assert molmoact2.is_molmoact2("user/some-checkpoint", declared) is False

    assert probed == [], f"explicit type must not probe config.json, but read {probed}"


def test_is_molmoact2_from_config_transformers_native(monkeypatch):
    """A transformers-native ckpt (model_type=molmoact2, no lerobot type) → True."""
    monkeypatch.setattr(
        molmoact2,
        "_read_config_json",
        lambda _p: {"model_type": "molmoact2", "hidden_size": 4096},
    )
    assert molmoact2.is_molmoact2("allenai/MolmoAct2-SO100_101", None) is True


def test_is_molmoact2_lerobot_native_is_false(monkeypatch):
    """A lerobot-native molmoact2 (has draccus 'type') goes through the normal
    resolution path, NOT this wrapper → False."""
    monkeypatch.setattr(
        molmoact2,
        "_read_config_json",
        lambda _p: {"model_type": "molmoact2", "type": "molmoact2"},
    )
    assert molmoact2.is_molmoact2("some/lerobot-native-molmoact2", None) is False


def test_is_molmoact2_other_model_is_false(monkeypatch):
    """An ACT/Pi0/etc. checkpoint is not molmoact2."""
    monkeypatch.setattr(molmoact2, "_read_config_json", lambda _p: {"type": "act"})
    assert molmoact2.is_molmoact2("lerobot/act_aloha", None) is False


def test_auto_norm_tag_explicit_wins():
    """An explicitly requested norm_tag is returned verbatim (no I/O)."""
    assert molmoact2.auto_norm_tag("any/repo", "my_custom_tag") == "my_custom_tag"


def test_auto_norm_tag_single_tag(tmp_path):
    """A norm_stats.json with exactly one tag → that tag is auto-selected."""
    norm = {"metadata_by_tag": {"so100_so101_molmoact2": {"action_horizon": 30}}}
    (tmp_path / "norm_stats.json").write_text(json.dumps(norm))
    assert molmoact2.auto_norm_tag(str(tmp_path), None) == "so100_so101_molmoact2"


def test_auto_norm_tag_multiple_tags_returns_none(tmp_path):
    """Multiple tags → None (refuse to guess; caller must pass norm_tag=)."""
    norm = {"metadata_by_tag": {"tag_a": {}, "tag_b": {}}}
    (tmp_path / "norm_stats.json").write_text(json.dumps(norm))
    assert molmoact2.auto_norm_tag(str(tmp_path), None) is None


def test_auto_norm_tag_missing_file_returns_none(tmp_path):
    """No norm_stats.json locally and offline → None, not a crash."""
    assert molmoact2.auto_norm_tag(str(tmp_path), None) is None


def test_derive_image_keys_explicit_wins():
    """Explicit image_keys are returned unchanged."""
    keys = ["observation.images.top", "observation.images.side"]
    assert molmoact2.derive_image_keys(keys, "so_real") == keys


def test_derive_image_keys_default_when_none():
    """No keys and no embodiment → the documented default image keys."""
    assert molmoact2.derive_image_keys(None, None) == molmoact2.DEFAULT_IMAGE_KEYS


def test_derive_image_keys_from_embodiment():
    """Image rename targets are pulled from the embodiment's obs_rename."""
    pytest.importorskip("lerobot")
    # so_real renames front->observation.images.image, wrist->...wrist_image
    keys = molmoact2.derive_image_keys(None, "so_real")
    assert "observation.images.image" in keys
    assert all(k.startswith("observation.images.") for k in keys)


class TestReadConfigJsonLocal:
    """``_read_config_json`` / ``is_molmoact2`` reading a local ``config.json``
    (no Hub call) — the on-disk checkpoint path."""

    def test_reads_local_config_json(self, tmp_path):
        """A local dir with a valid config.json is parsed without hitting the Hub."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "molmoact2", "hidden_size": 4096}))
        assert molmoact2._read_config_json(str(tmp_path)) == {"model_type": "molmoact2", "hidden_size": 4096}

    def test_is_molmoact2_unreadable_config_is_false(self, monkeypatch):
        """A non-empty path whose config.json cannot be read → not molmoact2."""
        monkeypatch.setattr(molmoact2, "_read_config_json", lambda _p: None)
        assert molmoact2.is_molmoact2("some/unreachable-repo", None) is False

    def test_is_molmoact2_end_to_end_from_local_dir(self, tmp_path):
        """is_molmoact2 detects a transformers-native ckpt straight from a local dir."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "molmoact2"}))
        assert molmoact2.is_molmoact2(str(tmp_path), None) is True

    def test_malformed_local_config_returns_none(self, tmp_path):
        """A corrupt config.json yields None (ValueError swallowed), not a crash."""
        (tmp_path / "config.json").write_text("{not valid json")
        assert molmoact2._read_config_json(str(tmp_path)) is None

    def test_no_local_config_falls_through_to_hub(self, tmp_path, monkeypatch):
        """A dir without config.json does not short-circuit; it tries the Hub."""
        calls: list[tuple[str, str]] = []

        def fake_download(repo, filename):
            calls.append((repo, filename))
            raise FileNotFoundError("offline")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        assert molmoact2._read_config_json(str(tmp_path)) is None
        # The empty dir has no config.json so the hub branch ran with the dir path.
        assert calls == [(str(tmp_path), "config.json")]


class TestReadConfigJsonHub:
    """``_read_config_json`` resolving a repo id via the HF Hub."""

    def test_reads_config_from_hub(self, tmp_path, monkeypatch):
        """A repo id with no local dir downloads + parses config.json from the Hub."""
        cfg_file = tmp_path / "downloaded_config.json"
        cfg_file.write_text(json.dumps({"model_type": "molmoact2", "type": "molmoact2"}))
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, filename: str(cfg_file))
        config = molmoact2._read_config_json("allenai/MolmoAct2-SO100_101")
        assert config == {"model_type": "molmoact2", "type": "molmoact2"}

    def test_hub_download_failure_returns_none(self, monkeypatch):
        """Network/repo errors during Hub fetch are non-fatal → None."""

        def boom(repo, filename):
            raise OSError("network down")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
        assert molmoact2._read_config_json("nonexistent/repo") is None


class TestDeriveImageKeysEmbodimentFallback:
    """``derive_image_keys`` falling back to defaults when the embodiment spec
    cannot be resolved (the ``_embodiment_image_targets`` exception path)."""

    def test_unknown_embodiment_falls_back_to_defaults(self):
        """An unresolvable embodiment name does not raise; it yields the defaults."""
        pytest.importorskip("lerobot")
        keys = molmoact2.derive_image_keys(None, "definitely_not_a_real_embodiment_xyz")
        assert keys == molmoact2.DEFAULT_IMAGE_KEYS

    def test_embodiment_image_targets_returns_empty_on_bad_spec(self):
        """_embodiment_image_targets swallows resolution errors and returns []."""
        pytest.importorskip("lerobot")
        assert molmoact2._embodiment_image_targets("definitely_not_a_real_embodiment_xyz") == []

    def test_embodiment_image_targets_none_spec(self):
        """A None spec short-circuits to [] before any import."""
        assert molmoact2._embodiment_image_targets(None) == []


class TestBuildPolicy:
    """``build_policy`` wiring of lerobot's PUBLIC factory contract.

    These tests exercise the load path hardware-free by stubbing the three
    ``lerobot.policies.factory`` entry points the wrapper rides
    (``make_policy_config`` / ``get_policy_class`` / ``make_pre_post_processors``)
    plus the norm-tag/image-key helpers. They pin the kwargs the wrapper hands
    lerobot so an upstream factory-signature drift is caught here rather than
    only on a 21GB hardware run.
    """

    @pytest.fixture
    def fake_factory(self, monkeypatch):
        """Stub lerobot's factory + record what build_policy passes through it."""
        _require_lerobot_configs_features()
        import lerobot.policies.factory as factory

        calls: dict[str, object] = {}

        class FakePolicy:
            def __init__(self, cfg):
                self.cfg = cfg
                self.to_device = None
                self.eval_called = False

            def to(self, device):
                self.to_device = device
                return self

            def eval(self):
                self.eval_called = True
                return self

        def fake_make_policy_config(policy_type, **kwargs):
            calls["config_type"] = policy_type
            calls["config_kwargs"] = kwargs
            return {"_cfg": True, **kwargs}

        def fake_get_policy_class(policy_type):
            calls["policy_class_type"] = policy_type
            return FakePolicy

        def fake_make_pre_post(cfg):
            calls["processors_cfg"] = cfg
            return ("PRE", "POST")

        monkeypatch.setattr(factory, "make_policy_config", fake_make_policy_config)
        monkeypatch.setattr(factory, "get_policy_class", fake_get_policy_class)
        monkeypatch.setattr(factory, "make_pre_post_processors", fake_make_pre_post)
        # Pin helper outputs so the test asserts wiring, not discovery I/O.
        monkeypatch.setattr(molmoact2, "auto_norm_tag", lambda _p, tag: tag or "resolved_tag")
        monkeypatch.setattr(
            molmoact2,
            "derive_image_keys",
            lambda keys, _emb: keys or ["observation.images.cam"],
        )
        return calls, FakePolicy

    def test_returns_policy_processors_and_config(self, fake_factory):
        """build_policy returns the (policy, pre, post, cfg) 4-tuple from the factory."""
        policy, pre, post, cfg = molmoact2.build_policy(
            "allenai/MolmoAct2-SO100_101",
            device="cpu",
            norm_tag="so100_so101_molmoact2",
            inference_action_mode="continuous",
            image_keys=None,
            embodiment_spec=None,
        )
        assert pre == "PRE"
        assert post == "POST"
        assert cfg["_cfg"] is True
        # The policy was instantiated, moved to device, and put in eval mode.
        assert policy.to_device == "cpu"
        assert policy.eval_called is True

    def test_passes_checkpoint_and_mode_to_config(self, fake_factory):
        """The HF repo, resolved norm tag and action mode reach make_policy_config."""
        calls, _ = fake_factory
        molmoact2.build_policy(
            "allenai/MolmoAct2-SO100_101",
            device="cpu",
            norm_tag=None,  # -> auto_norm_tag stub yields "resolved_tag"
            inference_action_mode="discrete",
            image_keys=None,
            embodiment_spec=None,
        )
        assert calls["config_type"] == molmoact2.MOLMOACT2_TYPE
        kw = calls["config_kwargs"]
        assert kw["checkpoint_path"] == "allenai/MolmoAct2-SO100_101"
        assert kw["norm_tag"] == "resolved_tag"
        assert kw["inference_action_mode"] == "discrete"
        assert kw["device"] == "cpu"

    def test_builds_visual_state_and_action_features(self, fake_factory):
        """Image keys become VISUAL features; state/action features pin the dims."""
        from lerobot.configs import FeatureType

        calls, _ = fake_factory
        molmoact2.build_policy(
            "repo",
            device="cpu",
            norm_tag="t",
            inference_action_mode="continuous",
            image_keys=["observation.images.top", "observation.images.wrist"],
            embodiment_spec=None,
            state_dim=7,
            action_dim=7,
        )
        kw = calls["config_kwargs"]
        in_feats = kw["input_features"]
        assert in_feats["observation.images.top"].type == FeatureType.VISUAL
        assert in_feats["observation.images.top"].shape == (3, 224, 224)
        assert in_feats["observation.state"].type == FeatureType.STATE
        assert in_feats["observation.state"].shape == (7,)
        out_feats = kw["output_features"]
        assert out_feats["action"].type == FeatureType.ACTION
        assert out_feats["action"].shape == (7,)

    def test_device_none_resolves_via_torch(self, fake_factory, monkeypatch):
        """device=None resolves to cpu when CUDA is unavailable (no silent crash)."""
        import torch

        calls, _ = fake_factory
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        policy, _pre, _post, _cfg = molmoact2.build_policy(
            "repo",
            device=None,
            norm_tag="t",
            inference_action_mode="continuous",
            image_keys=["observation.images.cam"],
            embodiment_spec=None,
        )
        assert calls["config_kwargs"]["device"] == "cpu"
        assert policy.to_device == "cpu"

    def test_processors_built_from_resolved_config(self, fake_factory):
        """make_pre_post_processors is dispatched with the config build_policy made."""
        calls, _ = fake_factory
        _policy, _pre, _post, cfg = molmoact2.build_policy(
            "repo",
            device="cpu",
            norm_tag="t",
            inference_action_mode="continuous",
            image_keys=["observation.images.cam"],
            embodiment_spec=None,
        )
        assert calls["processors_cfg"] is cfg

    def test_missing_lerobot_factory_raises_install_hint(self, monkeypatch):
        """If lerobot.policies.factory is absent, a clear install hint is raised."""
        pytest.importorskip("lerobot")
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "lerobot.policies.factory":
                raise ImportError("no factory")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        with pytest.raises(ImportError, match="MolmoAct2 requires lerobot") as excinfo:
            molmoact2.build_policy(
                "repo",
                device="cpu",
                norm_tag="t",
                inference_action_mode="continuous",
                image_keys=["observation.images.cam"],
                embodiment_spec=None,
            )
        # The hint must name the install extra and the upstream lerobot PR so an
        # operator can act on it without spelunking (issue #52 fail-loud branch).
        msg = str(excinfo.value)
        assert "strands-robots[molmoact2]" in msg
        assert "PR #3604" in msg

    def test_missing_lerobot_configs_raises_install_hint(self, monkeypatch):
        """If lerobot.configs is absent, the first import guard raises the hint."""
        pytest.importorskip("lerobot")
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            # The first import guard pulls FeatureType/PolicyFeature from the
            # lerobot.configs package; the concrete submodule moved to
            # lerobot.configs.types in 0.5.2+. Block the whole configs subtree
            # so this guard stays pinned to the import target wherever inside
            # lerobot.configs it lives.
            if name == "lerobot.configs" or name.startswith("lerobot.configs."):
                raise ImportError("no configs")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        with pytest.raises(ImportError, match="MolmoAct2 requires lerobot") as excinfo:
            molmoact2.build_policy(
                "repo",
                device="cpu",
                norm_tag="t",
                inference_action_mode="continuous",
                image_keys=["observation.images.cam"],
                embodiment_spec=None,
            )
        msg = str(excinfo.value)
        assert "strands-robots[molmoact2]" in msg
        assert "PR #3604" in msg

    def test_transitive_dep_failure_names_real_dep_not_lerobot(self, monkeypatch):
        """A missing *transitive* dep must surface THAT dep, not a lerobot hint.

        When importing ``lerobot.configs`` fails because a package it pulls in
        is missing (``ModuleNotFoundError`` whose ``name`` is the transitive
        package, e.g. ``einops``), telling the caller to reinstall lerobot from
        source is a dead end -- lerobot is fine; the missing package is the fix.
        The error must name the real package and how to install it.
        """
        pytest.importorskip("lerobot")
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            # Block the configs subtree: the actual import target is
            # lerobot.configs.types since the 0.5.2 migration.
            if name == "lerobot.configs" or name.startswith("lerobot.configs."):
                raise ModuleNotFoundError("No module named 'einops'", name="einops")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        with pytest.raises(ImportError) as excinfo:
            molmoact2.build_policy(
                "repo",
                device="cpu",
                norm_tag="t",
                inference_action_mode="continuous",
                image_keys=["observation.images.cam"],
                embodiment_spec=None,
            )
        msg = str(excinfo.value)
        assert "einops" in msg
        assert "pip install einops" in msg
        # Must NOT misattribute to an outdated lerobot.
        assert "PR #3604" not in msg
        assert "lerobot >= 0.5.2" not in msg

    def test_transitive_dep_failure_in_factory_import_names_real_dep(self, monkeypatch):
        """Same discrimination for the second guard (lerobot.policies.factory).

        The first guard (``from lerobot.configs import FeatureType, PolicyFeature``)
        is stubbed so this test isolates the SECOND guard's discrimination. Those
        symbols are only re-exported from ``lerobot.configs`` on lerobot >= 0.5.2;
        on lerobot 0.5.1 (the latest PyPI release) ``lerobot.configs`` is a
        namespace package without them, so the real first-guard import would fail
        first and mask the factory-guard behaviour under test.
        """
        pytest.importorskip("lerobot")
        import builtins
        import types

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "lerobot.policies.factory":
                raise ModuleNotFoundError("No module named 'qwen_vl_utils'", name="qwen_vl_utils")
            # Stub the configs subtree so the first guard passes and we
            # isolate the factory guard.
            if name == "lerobot.configs" or name.startswith("lerobot.configs."):
                return types.SimpleNamespace(FeatureType=object, PolicyFeature=object)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        with pytest.raises(ImportError) as excinfo:
            molmoact2.build_policy(
                "repo",
                device="cpu",
                norm_tag="t",
                inference_action_mode="continuous",
                image_keys=["observation.images.cam"],
                embodiment_spec=None,
            )
        msg = str(excinfo.value)
        assert "qwen_vl_utils" in msg
        assert "PR #3604" not in msg


class TestFactoryImportErrorTranslation:
    """Unit-level coverage of the import-error discriminator.

    ``_factory_import_error`` maps a caught ``ImportError`` to the actionable
    remedy by inspecting ``ImportError.name``: a missing/old lerobot keeps the
    from-source version hint; a missing transitive dependency names that
    package instead.
    """

    def test_transitive_dep_named(self):
        exc = ModuleNotFoundError("No module named 'einops'", name="einops")
        msg = str(molmoact2._factory_import_error(exc))
        assert "einops" in msg
        assert "pip install einops" in msg
        assert "PR #3604" not in msg

    def test_missing_lerobot_module_keeps_version_hint(self):
        exc = ModuleNotFoundError("No module named 'lerobot'", name="lerobot")
        msg = str(molmoact2._factory_import_error(exc))
        assert "PR #3604" in msg
        assert "strands-robots[molmoact2]" in msg

    def test_missing_lerobot_symbol_keeps_version_hint(self):
        # Old lerobot: module present, FeatureType symbol absent. ``from
        # lerobot.configs import FeatureType`` reports name="lerobot.configs".
        exc = ImportError(
            "cannot import name 'FeatureType' from 'lerobot.configs'",
            name="lerobot.configs",
        )
        msg = str(molmoact2._factory_import_error(exc))
        assert "PR #3604" in msg

    def test_no_name_falls_back_to_version_hint(self):
        # Defensive: an ImportError with no ``.name`` keeps the lerobot hint
        # rather than emitting a misleading "None is not installed".
        msg = str(molmoact2._factory_import_error(ImportError("opaque")))
        assert "PR #3604" in msg
        assert "None" not in msg


class TestBuildPolicyDependencyGuard:
    """``build_policy`` must guard its auxiliary deps up front with an error that
    names the ``strands-robots[molmoact2]`` extra AND lists every missing dep.

    Without the guard the first missing dep (e.g. ``transformers``) only surfaces
    deep inside lerobot's own model construction, where ``require_package``
    raises ``'transformers' is required ... pip install 'lerobot[transformers-dep]'``
    -- a dead end for a caller who installed ``strands-robots[molmoact2]`` and
    never touched lerobot's extras directly. The guard turns that into an
    actionable, correctly-attributed error before any heavy import runs.

    The guard uses ``require_optionals`` (plural), which probes ALL deps and
    reports every missing one at once -- so a partially-provisioned env (e.g.
    transformers present but peft and scipy both absent) is fixed in a single
    install instead of one reinstall-and-retry per missing dep.
    """

    def test_missing_dep_raises_naming_strands_extra(self, monkeypatch):
        """A missing runtime dep aborts with the strands-robots extra in the hint."""
        seen: list[object] = []

        def fake_require_optionals(module_names, *, extra=None, purpose=""):
            seen.append((tuple(module_names), extra))
            raise ImportError(
                f"'transformers' is required for {purpose}\n"
                "Install with:\n"
                f"  pip install 'strands-robots[{extra}]'\n"
                "  pip install transformers"
            )

        monkeypatch.setattr(molmoact2, "require_optionals", fake_require_optionals)

        with pytest.raises(ImportError) as exc:
            molmoact2.build_policy(
                "allenai/MolmoAct2-SO100_101",
                device="cpu",
                norm_tag="t",
                inference_action_mode="continuous",
                image_keys=None,
                embodiment_spec=None,
            )

        msg = str(exc.value)
        assert "strands-robots[molmoact2]" in msg
        assert "transformers" in msg
        # The guard runs before any heavy torch/lerobot import, against the
        # molmoact2 extra, and covers the full runtime-dep set in one call.
        assert seen[0] == (molmoact2._MOLMOACT2_RUNTIME_DEPS, "molmoact2")

    def test_all_runtime_deps_guarded_in_one_call(self, monkeypatch):
        """Every auxiliary dep is gated against the molmoact2 extra in a single call."""
        seen: list[object] = []

        def fake_require_optionals(module_names, *, extra=None, purpose=""):
            seen.append((tuple(module_names), extra))

        monkeypatch.setattr(molmoact2, "require_optionals", fake_require_optionals)
        # Make the post-guard lerobot import fail predictably so the test does
        # not reach real model construction.
        monkeypatch.setattr(molmoact2, "auto_norm_tag", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("stop")))

        with pytest.raises((RuntimeError, ImportError)):
            molmoact2.build_policy(
                "repo",
                device="cpu",
                norm_tag="t",
                inference_action_mode="continuous",
                image_keys=None,
                embodiment_spec=None,
            )

        assert len(seen) == 1, "deps must be gated in one aggregate call, not a loop"
        names, extra = seen[0]
        assert names == molmoact2._MOLMOACT2_RUNTIME_DEPS
        assert extra == "molmoact2"

    def test_reports_every_missing_dep_at_once(self, monkeypatch):
        """Regression: a partial env (peft + scipy both missing) names BOTH in one error.

        Pre-fix the guard looped ``require_optional`` and raised on the first
        missing dep only, so a caller fixed peft, re-ran the heavy load path, and
        only then learned scipy was also missing -- a reinstall treadmill. The
        aggregate guard must surface every missing dep in a single message.
        """
        import strands_robots.utils as u

        absent = {"peft", "scipy"}
        real_import = u.importlib.import_module

        def fake_import(name, *a, **k):
            if name.split(".")[0] in absent:
                raise ImportError(f"no module named {name}")
            return real_import(name, *a, **k)

        monkeypatch.setattr(u.importlib, "import_module", fake_import)
        # Drop any cached real peft/scipy so the probe actually fails.
        for mod in list(u._lazy_modules):
            if mod in absent:
                del u._lazy_modules[mod]

        with pytest.raises(ImportError) as exc:
            molmoact2.build_policy(
                "allenai/MolmoAct2-SO100_101",
                device="cpu",
                norm_tag="t",
                inference_action_mode="continuous",
                image_keys=None,
                embodiment_spec=None,
            )

        msg = str(exc.value)
        assert "peft" in msg
        assert "scipy" in msg
        assert "strands-robots[molmoact2]" in msg
