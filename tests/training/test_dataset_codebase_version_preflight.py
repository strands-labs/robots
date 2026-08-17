"""A local dataset root whose format version lerobot cannot read is refused by
``validate()``, not by the loader.

lerobot compares a dataset's declared ``codebase_version`` against its own
``CODEBASE_VERSION`` and refuses an older MAJOR. Learning that from the loader is
poor: only a v2.1 root gets a message naming the dataset and the converter, and
every other older major dies inside ``BackwardCompatibilityError.__init__`` with a
bare ``NotImplementedError: Contact the maintainer on [Discord](...)`` that names
neither the dataset, the version, nor the problem.

The version is declared in the same ``meta/info.json`` this trainer already reads
for the episode count, so the mismatch is decidable offline - no network, no
download - which is what lets it move ahead of the run.
"""

import json

import pytest

from strands_robots.training import TrainSpec
from strands_robots.training.lerobot import LerobotTrainer

#: Substring that identifies this preflight's problem among the others.
_VERSION_HINT = "codebase_version"

#: The converter lerobot ships, named by the v2.1 remedy.
_CONVERTER_HINT = "convert_dataset_v21_to_v30"


def _write_root(tmp_path, version, *, name="ds"):
    """A minimal local dataset root declaring ``version`` (or none when None)."""
    root = tmp_path / name
    (root / "meta").mkdir(parents=True)
    info = {"total_episodes": 10, "robot_type": "so101", "fps": 30}
    if version is not None:
        info["codebase_version"] = version
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return str(root)


def _spec(root, tmp_path, **kw):
    return TrainSpec(
        dataset_root=root,
        base_model="",
        output_dir=str(tmp_path / "out"),
        steps=200,
        global_batch_size=4,
        extra={"policy_type": "act", "job_name": "t", **kw.pop("extra", {})},
        **kw,
    )


def _offending(problems):
    return [p for p in problems if _VERSION_HINT in p]


class TestAnUnreadableFormatVersionIsRefusedAtValidateTime:
    """The refusal names the root, both versions, and a remedy that exists."""

    def test_a_v21_root_is_refused_and_pointed_at_lerobots_converter(self, tmp_path):
        root = _write_root(tmp_path, "v2.1")
        problems = LerobotTrainer().validate(_spec(root, tmp_path))
        offending = _offending(problems)
        assert offending, f"a v2.1 root is not loadable by lerobot v3; got {problems}"
        message = offending[0]
        assert root in message, message
        assert "v2.1" in message, message
        assert _CONVERTER_HINT in message, message

    @pytest.mark.parametrize("version", ["v2.0", "v1.6"])
    def test_an_older_root_is_refused_without_claiming_a_converter_for_it(self, tmp_path, version):
        """These are the versions whose loader failure names nothing at all.

        lerobot's only conversion is v2.1 -> v3.0, so the remedy must not offer
        to convert a version the converter does not accept.
        """
        root = _write_root(tmp_path, version)
        problems = LerobotTrainer().validate(_spec(root, tmp_path))
        offending = _offending(problems)
        assert offending, f"{version} is not loadable by lerobot v3; got {problems}"
        message = offending[0]
        assert version in message, message
        assert f"--repo-id={version}" not in message, message
        assert "no converter" in message, message

    def test_the_remedy_never_passes_the_local_sentinel_as_a_repo_id(self, tmp_path):
        """``_dataset_source`` calls a local-only root "local".

        That is lerobot's placeholder for "not a Hub dataset", not a repo id, so
        a command built from it would name a repo nobody can convert. The
        quantile-stats preflight makes the same substitution.
        """
        root = _write_root(tmp_path, "v2.1")
        problems = LerobotTrainer().validate(_spec(root, tmp_path))
        offending = _offending(problems)
        assert offending, problems
        assert "--repo-id=local" not in offending[0], offending[0]

    def test_a_reward_model_run_is_gated_too(self, tmp_path):
        """Both run types load the same dataset, so both are refused.

        The gate belongs to ``validate()`` rather than to the policy path: a
        reward-model run reads the same ``meta/info.json``.
        """
        root = _write_root(tmp_path, "v2.1")
        spec = _spec(root, tmp_path, extra={"reward_model": {"type": "sarm"}})
        assert _offending(LerobotTrainer().validate(spec)), "reward-model runs load the same dataset"


class TestItFlagsOnlyADefiniteMismatch:
    """Controls: every one of these is launchable, or not knowable offline."""

    @pytest.mark.parametrize("version", ["v3.0", "v3.1", "v4.0"])
    def test_a_current_or_newer_format_root_is_clean(self, tmp_path, version):
        """Only an older MAJOR is a refusal; an older minor merely warns."""
        root = _write_root(tmp_path, version)
        problems = LerobotTrainer().validate(_spec(root, tmp_path))
        assert not _offending(problems), problems

    @pytest.mark.parametrize("version", [None, "banana", ""])
    def test_an_unreadable_version_fails_open(self, tmp_path, version):
        """A version this cannot parse must not block a possibly-loadable root."""
        root = _write_root(tmp_path, version)
        problems = LerobotTrainer().validate(_spec(root, tmp_path))
        assert not _offending(problems), problems

    def test_a_hub_dataset_with_no_local_cache_is_not_flagged(self, tmp_path):
        """``validate()`` does not reach the network, so Hub metadata is unknown.

        Its format version is checked by lerobot when the shards load.
        """
        spec = TrainSpec(
            dataset_repo_id="acme/some-dataset",
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            global_batch_size=4,
            extra={"policy_type": "act", "job_name": "t"},
        )
        assert not _offending(LerobotTrainer().validate(spec))


class TestTheVersionHelpers:
    def test_the_declared_version_probe_is_tri_state(self, tmp_path):
        from strands_robots.training.lerobot import _dataset_codebase_version

        assert _dataset_codebase_version(str(tmp_path)) is None  # no meta/info.json
        root = _write_root(tmp_path, "v2.1")
        assert _dataset_codebase_version(root) == "v2.1"
        bare = _write_root(tmp_path, None, name="bare")
        assert _dataset_codebase_version(bare) is None
        broken = tmp_path / "broken" / "meta"
        broken.mkdir(parents=True)
        (broken / "info.json").write_text("{not json", encoding="utf-8")
        assert _dataset_codebase_version(str(broken.parent)) is None

    def test_the_major_is_read_off_either_spelling(self):
        from strands_robots.training.lerobot import _format_version_major

        assert _format_version_major("v3.0") == 3
        assert _format_version_major("3.0") == 3
        assert _format_version_major("v12.4") == 12
        assert _format_version_major("banana") is None
        assert _format_version_major("") is None

    def test_the_offline_fallback_matches_the_version_lerobot_enforces(self):
        """The fallback stands in for lerobot's own constant when it is absent.

        Pinned against the installed lerobot so the offline verdict cannot drift
        from the version the loader actually compares against.
        """
        from strands_robots.training.lerobot import (
            _LEROBOT_CODEBASE_VERSION_FALLBACK,
            _lerobot_codebase_version,
        )

        pytest.importorskip("lerobot.datasets.dataset_metadata")
        from lerobot.datasets.dataset_metadata import CODEBASE_VERSION

        assert _lerobot_codebase_version() == CODEBASE_VERSION
        assert _LEROBOT_CODEBASE_VERSION_FALLBACK == CODEBASE_VERSION

    def test_the_converter_source_version_is_the_one_lerobot_messages(self):
        """lerobot builds a real message for exactly this version.

        Every other older major raises ``NotImplementedError`` from the exception
        constructor instead, which is why the remedy branches on it.
        """
        from strands_robots.training.lerobot import _V21_TO_V30_SOURCE_VERSION

        packaging_version = pytest.importorskip("packaging.version")
        utils = pytest.importorskip("lerobot.datasets.utils")

        supported = packaging_version.parse(_V21_TO_V30_SOURCE_VERSION)
        assert utils.BackwardCompatibilityError("acme/ds", supported).args
        with pytest.raises(NotImplementedError):
            utils.BackwardCompatibilityError("acme/ds", packaging_version.parse("2.0"))
