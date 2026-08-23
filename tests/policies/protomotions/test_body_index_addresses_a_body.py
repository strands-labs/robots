# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tracker body index is refused unless it addresses a row of ``body_names``.

:attr:`~strands_robots.policies.protomotions.config.ProtoMotionsConfig.anchor_body_index`
and ``root_body_index`` are the only two fields the config resolves a body NAME
from, and both are offsets into ``body_names``. An offset that misses is a
dimension error the control path cannot report, because the tracker consumes the
resolved body CONSISTENTLY: ``required_bodies`` declares that link, the runtime
merges its ``body.<name>.quat`` into every observation, and the future-reference
window slices the same row out of the motion cache. Every stage agrees on a link
that is not the anchor.

The two ways an offset misses are not symmetric, which is why the guard covers
both:

* A NEGATIVE index wraps. ``body_names[-1]`` is ``right_rubber_hand`` - a real
  link name, so nothing raises anywhere and the tracker anchors on the hand.
  Measured on the tracker's own G1 embodiment, that link's world orientation is
  20.2 degrees from ``torso_link`` with the waist turned and an arm raised.
* A POSITIVE out-of-range index raises ``IndexError: tuple index out of range``
  from the property that resolves the name, naming neither the field, the value,
  the valid range, nor the sidecar it came from - and only once something reads
  the property, not when the config was built.

``load_config_from_yaml`` documents the returned config as "validated for
consistent dimensions" and promises ``ValueError`` for "an inconsistent
dimension"; the module docstring promises such an error "surfaces at policy build
time with a clean message". Both indices are checked here so those hold for the
two dimensions that were unchecked.

The domain runs BEFORE the ``int()`` normalisation, which is what the four
sibling policy configs (``kimodo``, ``motionbricks``, ``vera``, ``wbc``) already
do - ``MotionBricksConfig.__post_init__`` states the reason in as many words.
Coercing first laundered a yaml ``anchor_body_index: true`` into row 1 (``head``)
and a ``2.7`` into row 2 (``left_hip_pitch_link``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from strands_robots.policies.protomotions.config import (
    GTP_G1_ANCHOR_BODY_INDEX,
    GTP_G1_BODY_NAMES,
    GTP_G1_JOINT_NAMES,
    GTP_G1_ROOT_BODY_INDEX,
    ProtoMotionsConfig,
    load_config_from_yaml,
)

_INDEX_FIELDS = ("anchor_body_index", "root_body_index")

# Values that name no row of ``body_names``. The wrapping negatives are the
# reachable slip (a 1-based sidecar, or an index copied from a list that counted
# from the end); ``True``/``2.7`` are the values the removed ``int()`` used to
# turn into a different, in-range row.
_UNADDRESSABLE: list[Any] = [
    pytest.param(-1, id="minus_one_wraps_to_the_last_body"),
    pytest.param(-17, id="negative_wrap_that_lands_on_the_right_body_by_accident"),
    pytest.param(len(GTP_G1_BODY_NAMES), id="one_past_the_end"),
    pytest.param(99, id="far_out_of_range"),
    pytest.param(True, id="boolean_reads_as_row_one"),
    pytest.param(2.7, id="non_integral_float_truncates_to_row_two"),
    pytest.param("16", id="string_spelling_of_a_valid_row"),
    pytest.param(None, id="none"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
]


def _built(**overrides: Any) -> ProtoMotionsConfig:
    """Build a config with ``overrides`` applied.

    Deliberately typed loose: half the cases below pass values the signature
    refuses, which is the point, and splatting a ``dict[str, Any]`` through the
    real constructor is what a type checker reports rather than what the guard
    reports.
    """
    return ProtoMotionsConfig(**overrides)


def _sidecar(tmp_path: Path, **robot: Any) -> Path:
    """Write a minimal but complete ``unified_pipeline.yaml`` sidecar."""
    yaml = pytest.importorskip("yaml")
    body = {
        "joint_names": list(GTP_G1_JOINT_NAMES),
        "body_names": list(GTP_G1_BODY_NAMES),
        "robot": {
            "anchor_body_index": GTP_G1_ANCHOR_BODY_INDEX,
            "root_body_index": GTP_G1_ROOT_BODY_INDEX,
            **robot,
        },
    }
    path = tmp_path / "unified_pipeline.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


class TestABodyIndexMustAddressABody:
    """Both indices are refused unless they name a row of ``body_names``."""

    @pytest.mark.parametrize("field_name", _INDEX_FIELDS)
    @pytest.mark.parametrize("value", _UNADDRESSABLE)
    def test_an_unaddressable_index_is_refused_at_construction(self, field_name: str, value: Any) -> None:
        """A config carrying an index that names no body must not be built.

        Pre-fix every one of these built a config. The wrapping negatives then
        resolved a real but different link and nothing raised at all; the rest
        either resolved a different in-range row or deferred an ``IndexError``
        to whoever first read the name.
        """
        with pytest.raises(ValueError, match=field_name):
            _built(**{field_name: value})

    @pytest.mark.parametrize("field_name", _INDEX_FIELDS)
    def test_the_refusal_names_the_body_count_it_was_checked_against(self, field_name: str) -> None:
        """An out-of-range refusal quotes the range, so it is actionable.

        The pre-fix report was ``IndexError: tuple index out of range`` from a
        property read, which names neither the field nor the bound.
        """
        with pytest.raises(ValueError) as refused:
            _built(**{field_name: 99})
        message = str(refused.value)
        assert field_name in message
        assert "body_names" in message
        assert f"0..{len(GTP_G1_BODY_NAMES) - 1}" in message
        assert str(len(GTP_G1_BODY_NAMES)) in message

    @pytest.mark.parametrize("field_name", _INDEX_FIELDS)
    @pytest.mark.parametrize("value", [-1, 99, True, 2.7])
    def test_a_yaml_sidecar_is_refused_the_same_way(self, tmp_path: Path, field_name: str, value: Any) -> None:
        """The loader reports the same verdict as a hand-built config.

        ``load_config_from_yaml`` used to coerce the two indices with ``int()``
        before the dataclass saw them, so the yaml route and the direct route
        disagreed about ``true`` and ``2.7``.
        """
        with pytest.raises(ValueError, match=field_name):
            load_config_from_yaml(_sidecar(tmp_path, **{field_name: value}))

    def test_a_negative_index_no_longer_resolves_a_real_but_wrong_body(self) -> None:
        """The headline: -1 named ``right_rubber_hand`` and the tracker used it.

        Guards against a fix that only bounded the upper end: ``body_names[-1]``
        is a valid tuple lookup, so an upper-bound-only check leaves the silent
        case in place. The assertion states what the wrong answer WAS, so it
        fails with that name rather than only with "did not raise".
        """
        assert GTP_G1_BODY_NAMES[-1] == "right_rubber_hand"
        assert GTP_G1_BODY_NAMES[GTP_G1_ANCHOR_BODY_INDEX] == "torso_link"
        with pytest.raises(ValueError, match="anchor_body_index"):
            built = _built(anchor_body_index=-1)
            pytest.fail(
                "ProtoMotionsConfig accepted anchor_body_index=-1 and resolved "
                f"{built.anchor_body_name!r} as the anchor, where the config's own "
                f"index 16 is {GTP_G1_BODY_NAMES[GTP_G1_ANCHOR_BODY_INDEX]!r}."
            )

    def test_an_empty_body_list_cannot_carry_an_anchor(self) -> None:
        """No row exists, so the shipped default index addresses nothing."""
        with pytest.raises(ValueError, match="body_names"):
            ProtoMotionsConfig(body_names=())

    def test_a_yaml_boolean_no_longer_becomes_the_head_row(self) -> None:
        """A yaml ``anchor_body_index: true`` used to load as row 1.

        ``bool`` is an ``int`` subclass, so the loader's ``int()`` turned it into
        a row that addresses ``head`` - 34.4 degrees from ``torso_link`` with the
        waist turned. Stated as its own case because it is the value the removed
        coercion laundered, not one a range check alone would catch.
        """
        assert GTP_G1_BODY_NAMES[int(True)] == "head"
        with pytest.raises(ValueError, match="anchor_body_index"):
            built = _built(anchor_body_index=True)
            pytest.fail(
                "ProtoMotionsConfig accepted anchor_body_index=True and resolved "
                f"{built.anchor_body_name!r} as the anchor."
            )

    def test_the_bound_is_the_configs_own_body_list(self) -> None:
        """A trimmed embodiment bounds the index to ITS rows, not the G1's.

        Fails for a fix that hard-codes ``len(GTP_G1_BODY_NAMES)`` instead of
        reading the config's own list.
        """
        trimmed = GTP_G1_BODY_NAMES[:20]
        with pytest.raises(ValueError, match=r"0\.\.19"):
            _built(body_names=trimmed, anchor_body_index=20)

    @pytest.mark.parametrize("field_name", _INDEX_FIELDS)
    def test_an_integral_float_is_normalised_to_the_row_number(self, field_name: str) -> None:
        """``3.0`` addresses a row, so it is normalised rather than refused.

        The shared whole-number domain admits an integral float; storing it as an
        ``int`` matters because both consumers index with it - a tuple lookup for
        the name and a numpy slice for the future-reference window.
        """
        cfg = _built(**{field_name: 3.0})
        assert getattr(cfg, field_name) == 3
        assert isinstance(getattr(cfg, field_name), int)
        assert not isinstance(getattr(cfg, field_name), bool)


class TestTheAddressableIndicesAreUnchanged:
    """Controls: nothing that could address a body before is refused now."""

    def test_the_shipped_defaults_still_build(self) -> None:
        """The default config is the pinned G1 tracker embodiment."""
        cfg = ProtoMotionsConfig()
        assert cfg.anchor_body_index == GTP_G1_ANCHOR_BODY_INDEX
        assert cfg.root_body_index == GTP_G1_ROOT_BODY_INDEX
        assert cfg.anchor_body_name == "torso_link"
        assert cfg.root_body_name == "pelvis"

    @pytest.mark.parametrize("field_name", _INDEX_FIELDS)
    @pytest.mark.parametrize("value", [0, 1, 16, len(GTP_G1_BODY_NAMES) - 1])
    def test_every_row_of_body_names_is_accepted(self, field_name: str, value: int) -> None:
        """Every in-range row stays addressable, including both endpoints."""
        cfg = _built(**{field_name: value})
        assert getattr(cfg, field_name) == value
        assert cfg.body_names[value] == GTP_G1_BODY_NAMES[value]

    def test_dataclasses_replace_still_repoints_the_anchor(self) -> None:
        """The runtime-contract suite repoints the anchor onto the root."""
        base = ProtoMotionsConfig()
        root_anchored = dataclasses.replace(base, anchor_body_index=base.root_body_index)
        assert root_anchored.anchor_is_root
        assert root_anchored.anchor_body_name == base.root_body_name

    def test_a_shorter_embodiment_still_addresses_its_own_rows(self) -> None:
        """The bound is the config's own ``body_names``, not the G1 constant.

        A fingerless or otherwise trimmed embodiment is a legitimate config and
        index 16 addresses a body there too; only where the range ENDS changes.
        """
        trimmed = GTP_G1_BODY_NAMES[:20]
        cfg = _built(body_names=trimmed, anchor_body_index=16)
        assert cfg.anchor_body_name == trimmed[16]

    def test_the_shipped_sidecar_shape_loads_unchanged(self, tmp_path: Path) -> None:
        """A well-formed sidecar is unaffected by the widened check."""
        cfg = load_config_from_yaml(_sidecar(tmp_path))
        assert cfg.anchor_body_name == "torso_link"
        assert cfg.root_body_name == "pelvis"
        assert cfg.num_bodies == len(GTP_G1_BODY_NAMES)
        assert cfg.num_dofs == len(GTP_G1_JOINT_NAMES)

    def test_the_existing_joint_count_refusal_is_untouched(self, tmp_path: Path) -> None:
        """The two dimensions the loader already checked report as before."""
        yaml = pytest.importorskip("yaml")
        path = tmp_path / "unified_pipeline.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "joint_names": list(GTP_G1_JOINT_NAMES),
                    "body_names": list(GTP_G1_BODY_NAMES),
                    "control": {"stiffness": [1.0, 2.0, 3.0]},
                }
            )
        )
        with pytest.raises(ValueError, match=r"stiffness length \(3\) != joint count \(29\)"):
            load_config_from_yaml(path)


class TestTheResolvedNameIsWhatTheTrackerConsumes:
    """The index reaches the policy's declared body and its cache slice."""

    def test_required_bodies_is_the_resolved_anchor_name(self) -> None:
        """A wrong index would have propagated through this declaration.

        ``required_bodies`` is what the runtime resolves once per rollout to
        merge ``body.<name>.quat``, so the index and the observation key cannot
        disagree - which is exactly why a wrong index is silent.
        """
        pytest.importorskip("numpy")
        from strands_robots.policies.protomotions.policy import ProtoMotionsPolicy

        cfg = ProtoMotionsConfig()
        policy = ProtoMotionsPolicy.__new__(ProtoMotionsPolicy)
        policy._config = cfg  # noqa: SLF001 - reading the property, not building a session
        assert policy.required_bodies == (cfg.anchor_body_name,)
        assert policy.required_bodies == ("torso_link",)
