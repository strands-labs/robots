"""Tests for the Trainer abstraction: ABC contract, factory, mock lifecycle."""

import ast
import json
import os
import re

import pytest

from strands_robots.training import (
    Trainer,
    TrainResult,
    TrainSpec,
    create_trainer,
    import_trainer_class,
    list_trainers,
    register_trainer,
)
from strands_robots.training.factory import _runtime_registry
from strands_robots.training.mock import MockTrainer


@pytest.fixture
def dataset_root(tmp_path):
    """A minimal LeRobotDataset v3 root (just meta/info.json)."""
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 5}))
    return str(tmp_path)


@pytest.fixture
def spec(dataset_root, tmp_path):
    out = tmp_path / "ft_out"
    return TrainSpec(
        dataset_root=dataset_root,
        base_model="mock/base",
        output_dir=str(out),
        steps=100,
    )


class TestFactory:
    def test_create_from_registry(self):
        """`mock` resolves via its policies.json trainer block."""
        t = create_trainer("mock")
        assert isinstance(t, MockTrainer)
        assert t.provider_name == "mock"

    def test_list_trainers_includes_mock(self):
        assert "mock" in list_trainers()

    def test_builtin_rl_trainers_coexist(self):
        """Both from-scratch RL trainers stay registered side by side.

        ``training.__init__`` wires the on-policy ``ppo`` and off-policy
        ``fast_sac`` providers through separate lazy loaders; a regression that
        drops either registration would silently strip one RL backend. Pin that
        both are discoverable and resolve to distinct trainer classes.
        """
        registered = list_trainers()
        assert "ppo" in registered
        assert "fast_sac" in registered
        ppo = create_trainer("ppo")
        fast_sac = create_trainer("fast_sac")
        assert ppo.provider_name == "ppo"
        assert fast_sac.provider_name == "fast_sac"
        assert type(ppo) is not type(fast_sac)

    def test_import_trainer_class(self):
        assert import_trainer_class("mock") is MockTrainer

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="No trainer registered"):
            create_trainer("does_not_exist_xyz")

    def test_runtime_register_and_alias(self):
        register_trainer("custom_x", lambda: MockTrainer, aliases=["cx"])
        assert isinstance(create_trainer("custom_x"), MockTrainer)
        assert isinstance(create_trainer("cx"), MockTrainer)
        assert "custom_x" in list_trainers()

    def test_trainer_is_subclass(self):
        assert issubclass(MockTrainer, Trainer)


class TestValidate:
    def test_clean_spec_has_no_problems(self, spec):
        assert create_trainer("mock").validate(spec) == []

    def test_missing_dataset_reported(self, tmp_path):
        t = create_trainer("mock")
        s = TrainSpec(
            dataset_root=str(tmp_path / "nope"),
            base_model="m",
            output_dir=str(tmp_path / "o"),
        )
        problems = t.validate(s)
        assert any("LeRobotDataset v3" in p for p in problems)

    def test_bad_method_reported(self, spec):
        spec.method = "banana"
        problems = create_trainer("mock").validate(spec)
        assert any("unsupported method" in p for p in problems)

    def test_lora_expert_only_mutually_exclusive(self, spec):
        spec.method = "lora"
        spec.tune = {"expert_only": True}
        problems = create_trainer("mock").validate(spec)
        assert any("mutually exclusive" in p for p in problems)

    def test_nonpositive_steps_reported(self, spec):
        spec.steps = 0
        problems = create_trainer("mock").validate(spec)
        assert any("steps must be a positive integer" in p for p in problems)


class TestLifecycle:
    def test_train_writes_checkpoint_and_succeeds(self, spec):
        t = create_trainer("mock")
        res = t.train(spec)
        assert isinstance(res, TrainResult)
        assert res.status == "success"
        assert res.job_id
        assert res.checkpoint_dir and os.path.isfile(os.path.join(res.checkpoint_dir, "config.json"))
        assert res.metrics["learning"] is True

    def test_train_refuses_invalid_spec(self, tmp_path):
        t = create_trainer("mock")
        bad = TrainSpec(dataset_root="/nope", base_model="", output_dir="", steps=0)
        res = t.train(bad)
        assert res.status == "error"
        assert "validation failed" in res.message

    def test_export_default_is_passthrough(self, spec):
        t = create_trainer("mock")
        res = t.train(spec)
        assert t.export(spec, res.checkpoint_dir) == res.checkpoint_dir

    def test_latest_checkpoint_after_train(self, spec):
        # MockTrainer writes checkpoints/last; latest_checkpoint must find it.
        t = create_trainer("mock")
        res = t.train(spec)
        ckpt = t.latest_checkpoint(spec.output_dir)
        assert ckpt is not None
        assert ckpt == res.checkpoint_dir

    def test_latest_checkpoint_none_before_train(self, tmp_path):
        t = create_trainer("mock")
        assert t.latest_checkpoint(str(tmp_path / "never_trained")) is None

    def test_status_reports_learning(self, spec):
        t = create_trainer("mock")
        res = t.train(spec)
        st = t.status(res.job_id)
        assert st.status == "success"
        assert st.metrics["learning"] is True

    def test_hardware_floor_default(self):
        floor = create_trainer("mock").hardware_floor
        assert floor["min_gpus"] == 1
        assert floor["multinode"] is False


class TestHardwareFloorContract:
    """Every registered trainer's ``hardware_floor`` honors the ABC contract.

    ``Trainer.hardware_floor`` powers the ``plan`` advisor: :meth:`Trainer.validate`
    checks a spec's requested resources against it, so a floor that omits a key
    or returns the wrong type would break feasibility checking for the whole
    provider family. The advisory keys are ``min_gpus`` (int), ``min_vram_gb``
    (int), and ``multinode`` (bool); GPU/VRAM counts must be non-negative.
    """

    @pytest.mark.parametrize("provider", list_trainers())
    def test_floor_shape_and_types(self, provider):
        floor = create_trainer(provider).hardware_floor

        assert isinstance(floor, dict), f"{provider} hardware_floor is not a dict"
        assert {"min_gpus", "min_vram_gb", "multinode"} <= set(floor), (
            f"{provider} hardware_floor is missing advisory keys: {sorted(floor)}"
        )

        # bool is a subclass of int, so reject it explicitly for the counts.
        assert isinstance(floor["min_gpus"], int) and not isinstance(floor["min_gpus"], bool)
        assert isinstance(floor["min_vram_gb"], int) and not isinstance(floor["min_vram_gb"], bool)
        assert isinstance(floor["multinode"], bool)

        assert floor["min_gpus"] >= 0
        assert floor["min_vram_gb"] >= 0


class TestSpecTolerance:
    def test_unknown_extra_keys_do_not_break_validate(self, spec):
        """The **kwargs-style tolerance rule: unknown extras are ignored."""
        spec.extra = {"some_future_flag": "value", "another": 123}
        assert create_trainer("mock").validate(spec) == []


class TestAutoDiscoveryFallback:
    """Last resolution rung of ``import_trainer_class``: when a provider is in
    neither the runtime registry nor a ``trainer`` block in policies.json, the
    factory falls back to importing ``strands_robots.training.<provider>`` and
    resolving a Trainer subclass.
    """

    def test_resolves_named_provider_trainer_class(self, monkeypatch):
        """A module exposing ``<Provider>Trainer`` is resolved by name."""
        import sys
        import types

        mod = types.ModuleType("strands_robots.training.autoprov")

        class AutoprovTrainer(MockTrainer):
            pass

        mod.AutoprovTrainer = AutoprovTrainer
        monkeypatch.setitem(sys.modules, "strands_robots.training.autoprov", mod)

        assert import_trainer_class("autoprov") is AutoprovTrainer
        assert isinstance(create_trainer("autoprov"), AutoprovTrainer)

    def test_scans_for_first_trainer_subclass_when_name_mismatched(self, monkeypatch):
        """When no ``<Provider>Trainer`` exists, the first Trainer subclass wins."""
        import sys
        import types

        mod = types.ModuleType("strands_robots.training.scanprov")

        class CustomBackendTrainer(MockTrainer):
            pass

        mod.CustomBackendTrainer = CustomBackendTrainer
        monkeypatch.setitem(sys.modules, "strands_robots.training.scanprov", mod)

        assert import_trainer_class("scanprov") is CustomBackendTrainer

    def test_importable_module_without_trainer_raises(self, monkeypatch):
        """A module that imports cleanly but exposes no Trainer subclass still
        raises ValueError with the available-trainers list (not ImportError)."""
        import sys
        import types

        mod = types.ModuleType("strands_robots.training.emptyprov")
        monkeypatch.setitem(sys.modules, "strands_robots.training.emptyprov", mod)

        with pytest.raises(ValueError, match="No trainer registered"):
            import_trainer_class("emptyprov")


class TestBothResolversServeEveryListedTrainer:
    """``import_trainer_class`` resolves every name ``list_trainers`` advertises.

    The factory has two entry points onto one question - which ``Trainer``
    subclass a provider name means. :func:`create_trainer` answers it to build
    an instance; :func:`import_trainer_class` is the public answer for a caller
    that wants the class without paying for construction. A name only one of
    them can serve makes the pair a coin flip on which door the caller used,
    and the refusal ``import_trainer_class`` raises builds its available list
    from ``list_trainers()`` - so a name it cannot serve is advertised by the
    very message that rejects it.
    """

    def test_every_listed_trainer_resolves(self):
        """No advertised provider is refused by the public resolver."""
        refused = {}
        for name in list_trainers():
            try:
                import_trainer_class(name)
            except Exception as e:  # noqa: BLE001 - report every failure, not the first
                refused[name] = f"{type(e).__name__}: {e}"
        assert not refused, f"list_trainers() advertises providers import_trainer_class refuses: {refused}"

    def test_the_refusal_advertises_only_names_it_can_serve(self):
        """A refusal must not enumerate the provider it just rejected."""
        with pytest.raises(ValueError) as exc:
            import_trainer_class("no_such_trainer_xyz")
        message = str(exc.value)
        advertised = ast.literal_eval(re.search(r"Available trainers: (\[[^\]]*\])", message).group(1))
        assert advertised, "premise: the refusal names an available-trainers list to grade"
        unservable = []
        for name in advertised:
            try:
                import_trainer_class(name)
            except Exception:  # noqa: BLE001 - any failure means the list over-promises
                unservable.append(name)
        assert not unservable, f"the refusal offers providers it cannot resolve: {unservable}\n  message: {message}"

    def test_the_two_entry_points_agree_on_every_listed_name(self):
        """Resolving a class and building an instance answer the same question."""
        disagree = []
        for name in list_trainers():
            try:
                import_trainer_class(name)
                imports = True
            except Exception:  # noqa: BLE001
                imports = False
            try:
                create_trainer(name)
                creates = True
            except Exception:  # noqa: BLE001
                creates = False
            if imports is not creates:
                disagree.append((name, imports, creates))
        assert not disagree, f"(name, import_trainer_class, create_trainer) disagree: {disagree}"

    def test_the_builtin_rl_trainers_are_reachable_through_the_public_resolver(self):
        """``ppo`` and ``fast_sac`` register at runtime, not in policies.json.

        Their modules live in the ``training.rl`` subpackage, so neither the
        JSON rung nor auto-discovery on ``strands_robots.training.<provider>``
        finds them - the runtime registry is the only rung that can.
        """
        from strands_robots.training.rl.fast_sac import FastSacTrainer
        from strands_robots.training.rl.ppo import PpoTrainer

        assert import_trainer_class("ppo") is PpoTrainer
        assert import_trainer_class("fast_sac") is FastSacTrainer

    def test_a_runtime_registered_trainer_and_its_alias_resolve(self):
        """The documented ``register_trainer`` route reaches the public resolver."""

        class ResolverProbeTrainer(MockTrainer):
            pass

        register_trainer("resolver_probe", lambda: ResolverProbeTrainer, aliases=["rp"])
        assert import_trainer_class("resolver_probe") is ResolverProbeTrainer
        assert import_trainer_class("rp") is ResolverProbeTrainer

    def test_at_least_one_listed_trainer_is_runtime_only(self):
        """Non-vacuity: the graded set spans both registries.

        Were every provider declared in policies.json, the properties above
        would hold for a resolver that consults only the JSON rung.
        """
        from strands_robots.registry.policies import get_policy_provider

        runtime_only = [name for name in list_trainers() if not (get_policy_provider(name) or {}).get("trainer")]
        assert runtime_only, "expected at least one provider registered outside policies.json"


class TestTheRuntimeRungKeepsItsPrecedence:
    """A runtime registration shadows a JSON ``trainer`` block, as before.

    ``create_trainer`` consulted the runtime registry ahead of the registry
    lookup, so a caller could already override a shipped provider's trainer by
    re-registering the name. Sharing one resolver has to keep that ordering:
    demoting the runtime rung below the JSON rung would silently ignore such an
    override instead of honoring it.
    """

    def test_a_runtime_registration_overrides_a_json_trainer_block(self):
        class ShadowingTrainer(MockTrainer):
            pass

        register_trainer("mock", lambda: ShadowingTrainer)
        try:
            assert import_trainer_class("mock") is ShadowingTrainer
            assert isinstance(create_trainer("mock"), ShadowingTrainer)
        finally:
            _runtime_registry.pop("mock", None)
        # The JSON rung answers again once the override is gone.
        assert import_trainer_class("mock") is MockTrainer

    def test_an_unknown_provider_is_still_refused(self):
        with pytest.raises(ValueError, match="No trainer registered"):
            import_trainer_class("definitely_not_a_trainer_xyz")
        with pytest.raises(ValueError, match="No trainer registered"):
            create_trainer("definitely_not_a_trainer_xyz")
