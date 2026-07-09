"""RewardModelConfig parity guard: every lerobot reward model reaches strands.

LeRobot registers its reward models on a single draccus ChoiceRegistry,
``RewardModelConfig`` (``@RewardModelConfig.register_subclass("<name>")`` in each
``lerobot/rewards/<type>/configuration_<type>.py``). The strands
:class:`~strands_robots.training.lerobot.LerobotTrainer` reward-model path must
stay in lock-step with that registry with ZERO hardcoding, the same way Robot /
Teleop / Camera / Policy discovery already does - any reward model lerobot ships
(or a plugin registers) must be reachable through ``extra['reward_model']``
without editing strands.

This is a source-level guard: it AST-scans the INSTALLED lerobot's reward
sources for the ``register_subclass`` decorators (the ground truth, independent
of import side effects) and asserts strands' dynamic discovery sees exactly the
same set and can validate + build a config for each. It ``importorskip``s
``lerobot.rewards`` so it self-skips on a lerobot too old to ship it
(< 0.5.2 / PyPI), where reward-model training cannot run anyway.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from strands_robots.training.base import TrainSpec
from strands_robots.training.lerobot import (
    LerobotTrainer,
    _reward_friendly_fields,
    _reward_model_types,
)


def _registered_reward_types_from_source() -> set[str]:
    """Ground-truth reward type names from the installed lerobot's source.

    Walks ``lerobot/rewards`` for ``configuration_*.py`` files and AST-parses
    each for an ``@RewardModelConfig.register_subclass(<name>)`` decorator,
    returning the registered names. Source parsing (not the runtime registry)
    is deliberate: it catches a type that lerobot ships but that strands' own
    discovery fails to import/register.
    """
    import lerobot.rewards

    rewards_root = Path(lerobot.rewards.__file__).parent
    names: set[str] = set()
    for cfg_path in rewards_root.rglob("configuration_*.py"):
        tree = ast.parse(cfg_path.read_text(encoding="utf-8"), filename=str(cfg_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for deco in node.decorator_list:
                name = _register_subclass_name(deco)
                if name is not None:
                    names.add(name)
    return names


def _register_subclass_name(deco: ast.expr) -> str | None:
    """Extract the name from a ``RewardModelConfig.register_subclass(...)`` call.

    Handles both the positional (``register_subclass("sarm")``) and keyword
    (``register_subclass(name="reward_classifier")``) forms lerobot uses.
    Returns ``None`` for any other decorator.
    """
    if not isinstance(deco, ast.Call):
        return None
    func = deco.func
    if not (isinstance(func, ast.Attribute) and func.attr == "register_subclass"):
        return None
    for arg in deco.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    for kw in deco.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


@pytest.fixture
def dataset_root(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return str(tmp_path)


class TestRewardModelConfigParity:
    """LerobotTrainer reaches every lerobot RewardModelConfig subclass."""

    def test_strands_discovery_matches_lerobot_source(self):
        """strands' dynamic reward-type discovery == lerobot's registered set.

        Equality (not just superset) in both directions: strands must not miss a
        type lerobot ships, and must not advertise a type lerobot does not.
        """
        pytest.importorskip("lerobot.rewards")
        source_types = _registered_reward_types_from_source()
        # Sanity: the audit baseline - lerobot ships at least these four.
        assert {"sarm", "robometer", "topreward", "reward_classifier"} <= source_types
        assert _reward_model_types() == source_types

    @pytest.mark.parametrize("rtype", ["sarm", "robometer", "topreward", "reward_classifier"])
    def test_every_reward_type_validates_and_builds(self, rtype, dataset_root, tmp_path):
        """Each reward type is reachable: validate() accepts it and build_config
        targets ``cfg.reward_model`` (lerobot's is_reward_model_training path)."""
        pytest.importorskip("lerobot.rewards")
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / f"{rtype}_out"),
            steps=100,
            extra={"reward_model": {"type": rtype}},
        )
        trainer = LerobotTrainer(device="cpu")
        assert trainer.validate(spec) == [], f"{rtype} failed validation"
        cfg = trainer.build_config(spec)
        assert cfg.is_reward_model_training is True
        assert cfg.policy is None
        assert cfg.reward_model.type == rtype

    @pytest.mark.parametrize("rtype", ["sarm", "robometer", "topreward", "reward_classifier"])
    def test_own_field_passthrough_per_type(self, rtype, dataset_root, tmp_path):
        """A type's own config knob flows through to the built config.

        Picks one subclass-declared field per type and asserts it both passes
        validation (the friendly surface is per-type, not SARM-only) and lands on
        the built ``cfg.reward_model`` - the dynamic-passthrough contract that
        makes all four types configurable, not just SARM.
        """
        pytest.importorskip("lerobot.rewards")
        # normalization_mapping is declared by every reward subclass; use a
        # simpler per-type scalar knob to assert real value passthrough.
        knob = {
            "sarm": ("annotation_mode", "single_stage"),
            "robometer": ("default_task", "pick up the cube"),
            "topreward": ("default_task", "pick up the cube"),
            "reward_classifier": ("num_classes", 3),
        }[rtype]
        field, value = knob
        assert field in _reward_friendly_fields(rtype)
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / f"{rtype}_out"),
            steps=100,
            extra={"reward_model": {"type": rtype, field: value}},
        )
        trainer = LerobotTrainer(device="cpu")
        assert trainer.validate(spec) == [], f"{rtype}.{field} rejected"
        cfg = trainer.build_config(spec)
        assert getattr(cfg.reward_model, field) == value

    def test_cross_type_field_is_rejected(self, dataset_root, tmp_path):
        """SARM's annotation_mode is not a robometer field -> rejected.

        Guards the per-type field validation: before this, the friendly key set
        was a single SARM-biased list that wrongly accepted annotation_mode for
        every type (then failed deep in make_reward_model_config).
        """
        pytest.importorskip("lerobot.rewards")
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "robometer_out"),
            steps=100,
            extra={"reward_model": {"type": "robometer", "annotation_mode": "single_stage"}},
        )
        problems = LerobotTrainer(device="cpu").validate(spec)
        assert any("does not support field" in p and "annotation_mode" in p for p in problems)

    def test_ctor_rejection_becomes_actionable_error(self, dataset_root, tmp_path, monkeypatch):
        """A field the reward-config CONSTRUCTOR rejects surfaces an actionable
        ValueError naming the fields, not a raw TypeError.

        ``_reward_friendly_fields`` introspects the resolved config dataclass to
        decide which ``extra['reward_model']`` keys to forward. If the installed
        lerobot's constructor and that introspection ever diverge (an API drift),
        the forwarded kwargs can be rejected by ``make_reward_model_config`` with
        a ``TypeError``. The trainer must translate that into a ``ValueError``
        that names the rejected fields so the drift is diagnosable, and chain the
        original ``TypeError`` as the cause - not leak a bare, contextless error.
        """
        pytest.importorskip("lerobot.rewards")
        import lerobot.rewards as lr

        def _reject(rtype, **kwargs):
            raise TypeError("unexpected keyword argument 'device'")

        # The trainer imports make_reward_model_config from lerobot.rewards at
        # call time, so patching the module attribute intercepts the real call.
        monkeypatch.setattr(lr, "make_reward_model_config", _reject)

        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "sarm_out"),
            steps=100,
            extra={"reward_model": {"type": "sarm"}},
        )
        trainer = LerobotTrainer(device="cpu")
        with pytest.raises(ValueError) as excinfo:
            trainer.build_config(spec)

        msg = str(excinfo.value)
        assert "reward_model type 'sarm' rejected field(s)" in msg
        # The forwarded-field list (always includes the managed 'device') is
        # named so the drift is diagnosable from the message alone.
        assert "device" in msg
        # The original TypeError is chained for debugging, not swallowed.
        assert isinstance(excinfo.value.__cause__, TypeError)
