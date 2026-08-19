"""``DatasetTransform.validate`` refuses a spec it cannot honor, before anything is read.

The shared preflight (``DatasetTransform._spec_problems``) is the
provider-agnostic half of the transform contract, so it is tested here once
against the reference :class:`~strands_robots.transforms.mock.MockTransform`
rather than per backend. Three rules from AGENTS.md shape it and each is
pinned:

* the ``episodes`` field is a SUBSET SELECTOR, read ``is None`` / by
  membership - an empty selection is refused rather than widened to "all";
* the numeric knobs (``variants_per_episode``, ``seed``, each episode index)
  are checked against the SHARED value domains
  (:func:`~strands_robots.utils.positive_whole_number_error` /
  :func:`~strands_robots.utils.non_negative_whole_number_error`), parametrized
  over the helpers themselves so a value added to the shared domain is covered
  without an edit here;
* ``overwrite`` is a posture flag, checked as a strict boolean
  (:func:`~strands_robots.utils.boolean_flag_error`) - ``"false"`` must not
  select dataset deletion by truthiness.

One refusal is deliberately unconditional: ``output_root == source_root`` is
refused whatever ``overwrite`` says, because that spelling plus
``overwrite=True`` would delete the recorded source before reading it.
"""

import json
from typing import Any

import pytest

from strands_robots.transforms import TransformSpec
from strands_robots.transforms.mock import MockTransform
from strands_robots.utils import (
    boolean_flag_error,
    non_negative_whole_number_error,
    positive_whole_number_error,
)

CTX = "mock.validate"


@pytest.fixture
def source_root(tmp_path):
    """A minimal LeRobotDataset v3 root (just meta/info.json)."""
    meta = tmp_path / "source" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 3}))
    return str(tmp_path / "source")


@pytest.fixture
def spec(source_root, tmp_path):
    """A runnable spec every test below breaks one field of."""
    return TransformSpec(source_root=source_root, output_root=str(tmp_path / "aug"))


def _validate(spec: TransformSpec) -> list[str]:
    return MockTransform().validate(spec)


class TestRoots:
    """Source and output roots: presence, dataset-ness, distinctness."""

    def test_runnable_spec_has_no_problems(self, spec):
        assert _validate(spec) == []

    def test_missing_source_root(self, tmp_path):
        problems = _validate(TransformSpec(output_root=str(tmp_path / "aug")))
        assert "source_root is required" in problems

    def test_source_root_not_a_dataset(self, tmp_path):
        (tmp_path / "not_a_dataset").mkdir()
        problems = _validate(
            TransformSpec(source_root=str(tmp_path / "not_a_dataset"), output_root=str(tmp_path / "aug"))
        )
        assert any("not a LeRobotDataset v3 root" in p for p in problems)

    def test_missing_output_root(self, source_root):
        problems = _validate(TransformSpec(source_root=source_root))
        assert "output_root is required" in problems

    @pytest.mark.parametrize("overwrite", [False, True], ids=["default", "overwrite"])
    def test_output_equal_to_source_refused_whatever_the_flag_says(self, source_root, overwrite):
        """The one spelling that deletes the input is refused unconditionally."""
        problems = _validate(TransformSpec(source_root=source_root, output_root=source_root, overwrite=overwrite))
        assert any("must not be the source_root" in p for p in problems)

    def test_existing_output_dataset_needs_overwrite(self, source_root, tmp_path):
        out = tmp_path / "aug"
        (out / "meta").mkdir(parents=True)
        problems = _validate(TransformSpec(source_root=source_root, output_root=str(out)))
        assert any("already holds a dataset" in p and "overwrite=True" in p for p in problems)
        assert _validate(TransformSpec(source_root=source_root, output_root=str(out), overwrite=True)) == []


class TestEpisodesSelector:
    """``episodes`` is a subset selector: ``is None`` / membership, never truthiness."""

    def test_none_selects_all(self, spec):
        spec.episodes = None
        assert _validate(spec) == []

    def test_real_subset_passes(self, spec):
        spec.episodes = [0, 2]
        assert _validate(spec) == []

    def test_empty_selection_is_refused_not_widened(self, spec):
        spec.episodes = []
        problems = _validate(spec)
        assert any("EMPTY subset" in p for p in problems)

    @pytest.mark.parametrize("value", [0, (0, 1), {0: True}, "0"], ids=["bare-int", "tuple", "dict", "str"])
    def test_non_list_selector_is_refused(self, spec, value):
        spec.episodes = value
        problems = _validate(spec)
        assert any("episodes must be a list" in p for p in problems)

    def test_repeated_index_is_refused(self, spec):
        spec.episodes = [1, 1]
        problems = _validate(spec)
        assert any("repeated index" in p for p in problems)

    @pytest.mark.parametrize("value", [-1, 1.5, True, None], ids=["negative", "fractional", "bool", "none"])
    def test_unusable_index_reports_the_shared_domain_verdict(self, spec, value):
        """Each index is judged by the one shared non-negative-count domain."""
        expected = non_negative_whole_number_error(value, "episodes[]", CTX)
        assert expected is not None  # the domain itself rejects it
        spec.episodes = [value]
        assert expected in _validate(spec)


class TestNumericKnobs:
    """``variants_per_episode`` / ``seed`` are checked by the shared domains."""

    @pytest.mark.parametrize(
        "value",
        [0, -1, 2.7, True, "2", None, float("nan")],
        ids=["zero", "negative", "fractional", "bool", "str", "none", "nan"],
    )
    def test_unusable_variant_count(self, spec, value: Any):
        expected = positive_whole_number_error(value, "variants_per_episode", CTX)
        assert expected is not None
        spec.variants_per_episode = value
        assert expected in _validate(spec)

    def test_seed_none_is_the_opt_out(self, spec):
        spec.seed = None
        assert _validate(spec) == []

    @pytest.mark.parametrize("value", [-1, 0.5, True, "7"], ids=["negative", "fractional", "bool", "str"])
    def test_unusable_seed(self, spec, value: Any):
        expected = non_negative_whole_number_error(value, "seed", CTX)
        assert expected is not None
        spec.seed = value
        assert expected in _validate(spec)


class TestPostureAndHooks:
    """``overwrite`` is a strict boolean; ``revalidate`` a callable; ``prompt`` a string."""

    @pytest.mark.parametrize(
        "value", ["false", "no", 0, 1, None, []], ids=["str-false", "str-no", "zero", "one", "none", "list"]
    )
    def test_overwrite_is_checked_not_read_by_truthiness(self, spec, value: Any):
        expected = boolean_flag_error(value, "overwrite", CTX)
        assert expected is not None
        spec.overwrite = value
        assert expected in _validate(spec)

    def test_non_callable_revalidate_is_refused(self, spec):
        spec.revalidate = "predicates"
        problems = _validate(spec)
        assert any("revalidate must be a callable" in p for p in problems)

    def test_non_string_prompt_is_refused(self, spec):
        spec.prompt = 7
        problems = _validate(spec)
        assert any("prompt must be a string" in p for p in problems)

    def test_validate_reads_nothing_and_writes_nothing(self, spec, tmp_path):
        """Preflight is pure: a passing validate leaves the output root absent."""
        assert _validate(spec) == []
        assert not (tmp_path / "aug").exists()
