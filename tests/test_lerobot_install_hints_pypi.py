"""Currency contract for the lerobot "too old / absent" install hints.

Since ``strands-robots`` pins ``lerobot[feetech,dataset]>=0.6.0`` (pyproject),
lerobot 0.6 -- including ``MolmoAct2Policy`` (lerobot PR #3604) and the
``lerobot.rewards`` package -- ships straight from PyPI through the
``strands-robots[lerobot]`` / ``[molmoact2]`` extras. The user-facing "your
lerobot is too old / missing" error hints must therefore point the caller at a
plain PyPI (re)install of the extra, NOT a from-source / ``git+`` install of an
"unreleased" lerobot. These are the runtime-error siblings of the docs corrected
in the >=0.6 dependency-guidance pass; this pins that they stay in sync so a
broken-install user is never sent chasing a remedy that no longer applies (and
would conflict with the pinned >=0.6.0 floor).
"""

from __future__ import annotations

import importlib.util

import pytest

_STALE = ("from source", "git+", "not yet on PyPI", "0.5.1", "0.5.2")


def _assert_currency(text: str) -> None:
    """A lerobot install hint must name the >=0.6 PyPI floor, not from-source."""
    assert "lerobot >= 0.6" in text, f"hint should name the >=0.6 floor: {text!r}"
    for token in _STALE:
        assert token not in text, f"stale install advice ({token!r}) in hint: {text!r}"


class TestMolmoAct2VersionHint:
    def test_constant_names_pypi_extra_not_from_source(self) -> None:
        from strands_robots.policies.lerobot_local import molmoact2

        hint = molmoact2._LEROBOT_VERSION_HINT
        _assert_currency(hint)
        # The remedy is the PyPI extra, not a git+ lerobot install.
        assert "strands-robots[molmoact2]" in hint

    def test_factory_import_error_for_missing_lerobot_uses_currency_hint(self) -> None:
        from strands_robots.policies.lerobot_local import molmoact2

        # A missing/too-old lerobot reports name "lerobot" (or "lerobot.*"),
        # which routes to the version hint (not the transitive-dep branch).
        err = molmoact2._factory_import_error(ImportError("no lerobot", name="lerobot"))
        _assert_currency(str(err))

    def test_factory_import_error_transitive_dep_is_unaffected(self) -> None:
        from strands_robots.policies.lerobot_local import molmoact2

        # A missing transitive dep keeps its own targeted remedy (regression guard
        # that the currency edit did not collapse the two distinct branches).
        err = molmoact2._factory_import_error(ImportError("no einops", name="einops"))
        msg = str(err)
        # Remedy is to install the named transitive dep, not (re)install lerobot.
        assert "pip install einops" in msg
        assert "strands-robots" not in msg
        assert "git+" not in msg


class TestRewardModelHints:
    def test_load_reward_model_missing_rewards_uses_currency_hint(self, monkeypatch) -> None:
        from strands_robots.training import reward as reward_mod

        real = importlib.util.find_spec
        monkeypatch.setattr(
            reward_mod.importlib.util,
            "find_spec",
            lambda name: None if name == "lerobot.rewards" else real(name),
        )
        with pytest.raises(ImportError) as excinfo:
            reward_mod.load_reward_model("/ckpt/sarm", device="cpu")
        _assert_currency(str(excinfo.value))

    def test_trainer_validate_missing_rewards_uses_currency_hint(self, tmp_path, monkeypatch) -> None:
        from strands_robots.training.base import TrainSpec
        from strands_robots.training.lerobot import LerobotTrainer

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: None if name == "lerobot.rewards" else real(name),
        )
        spec = TrainSpec(
            dataset_root=str(tmp_path),
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=100,
            extra={"reward_model": {"type": "sarm", "annotation_mode": "single_stage"}},
        )
        problems = LerobotTrainer().validate(spec)
        reward_problems = [p for p in problems if "lerobot.rewards" in p]
        assert reward_problems, f"expected a reward-support problem, got: {problems}"
        for p in reward_problems:
            assert "lerobot >= 0.6" in p
            assert "from source" not in p
            assert "0.5.2" not in p
