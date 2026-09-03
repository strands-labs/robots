# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A source column's ``names`` are read as LeRobot writes them, or refused.

``_SourceDataset`` derives the output dataset's schema from the source's
``observation.state`` / ``action`` names -- ``create_output_recorder`` documents
itself as "the SOURCE schema (parity by construction)" -- and ``_write_episode``
fills those columns by pairing each name with the source row. So the name list
decides how wide the output trajectory is, and reading it wrongly writes a
different trajectory than the one recorded.

LeRobot writes a vector column's ``names`` in two spellings. Most datasets use a
flat list. But its own shipped teleoperators use a mapping:
``teleop_keyboard.action_features`` declares ``{"motors": list(self.arm.motors)}``
and ``teleop_gamepad`` declares ``{"delta_x": 0, "delta_y": 1, ...}``. The reader
did ``list(feature.get("names") or [])``, which on a mapping yields its KEYS --
one name per GROUP, not per column:

=====================================================  ==========  ==============
source ``action.names``                                6-col ->    outcome
=====================================================  ==========  ==============
``["shoulder_pan", ..., "gripper"]``                   6 -> 6      unaffected
``{"motors": [6 motors]}`` (``teleop_keyboard``)       6 -> **1**  5 dropped
``{"left": [6], "right": [6]}`` (bimanual)             12 -> **2** 10 dropped
``{"delta_x": 0, ...}`` (``teleop_gamepad``)           6 -> 6      unaffected
=====================================================  ==========  ==============

Every one of those transformed with ``status="success"``, wrote provenance
claiming the episode is its source, and reported it in ``episodes_written``.
``docs/data/transforms.md`` contract item 1 promises a generated episode is "the
*same trajectory* rendered differently"; a six-motor arm recorded as its
shoulder-pan column alone is not that.

The two mapping spellings do not mean the same thing, which is why they are read
differently rather than flattened alike: a list value holds the components (its
key is a group label), an integer value is the component's index (the key is the
column name). LeRobot's own ``_flatten_feature_names`` renders the second shape's
indices as the names, so following it would name a six-motor arm ``0..5``.

A count that still disagrees with the declared column width cannot be repaired by
reading it differently, so it is refused -- the fourth refusal in
:meth:`~strands_robots.transforms.base._SourceDataset.open`, beside the three it
already makes for the same reason (the output could not be the source rendered
differently). Both directions are silent otherwise: names short of the width drop
the trailing components, and names past it declare a column no frame supplies,
which ``DatasetRecorder.add_frame`` writes as ``0.0`` -- itself a travel-to-zero
command for an absolute-position actuator, per
:func:`~strands_robots.dataset_recorder.unrecordable_action_columns_error`.
``verify_dataset._declared_column_blocks`` already declined to use a ``names``
whose length differs from the vector's width ("a ``names`` list of a different
length does not describe this vector"); this holds the transform reader to the
same rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")

from strands_robots.transforms import TransformSpec, create_transform  # noqa: E402

_ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
_BIMANUAL = [f"left_{joint}" for joint in _ARM] + [f"right_{joint}" for joint in _ARM]


@pytest.fixture
def source_declaring(tmp_path):
    """Record a real source dataset, then rewrite its declared column ``names``.

    Returns a callable ``(joints, names) -> root``. ``names`` is written verbatim
    into ``meta/info.json`` for both ``observation.state`` and ``action``, which
    is how a dataset recorded by a LeRobot teleoperator arrives: the column data
    is well-formed and only the declaration's spelling differs.
    """
    from strands_robots.dataset_recorder import DatasetRecorder

    def _record(joints: list[str], names: object) -> str:
        root = tmp_path / f"source-{len(joints)}-{abs(hash(repr(names))) % 10000}"
        recorder = DatasetRecorder.create(
            "local/source",
            fps=10,
            camera_keys=["cam"],
            camera_dims={"cam": (64, 64)},
            joint_names=list(joints),
            action_names=list(joints),
            root=str(root),
        )
        for step in range(4):
            observation: dict[str, object] = {joint: 0.1 * (index + 1) + step for index, joint in enumerate(joints)}
            observation["cam"] = np.full((64, 64, 3), 40 + step, dtype=np.uint8)
            action = {joint: 1.0 * (index + 1) + step for index, joint in enumerate(joints)}
            recorder.add_frame(observation, action, task="pick")
        recorder.save_episode()
        recorder.finalize()

        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for key in ("observation.state", "action"):
            info["features"][key]["names"] = names
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        return str(root)

    return _record


def _transform(source_root: str, tmp_path: Path):
    """Run the mock transform over one source and return (result, output_root)."""
    output_root = str(tmp_path / f"out-{Path(source_root).name}")
    result = create_transform("mock").transform(
        TransformSpec(source_root=source_root, output_root=output_root, variants_per_episode=1, seed=7)
    )
    return result, output_root


def _column(root: str, repo_id: str, key: str) -> np.ndarray:
    """Stack one column of episode 0 from a dataset on disk."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    info = dataset.meta.episodes[0]
    start, stop = int(info["dataset_from_index"]), int(info["dataset_to_index"])
    return np.stack([np.asarray(dataset[index][key]).ravel() for index in range(start, stop)])


class TestTheDeclarationSpellingsAreReadAsWritten:
    """``_column_names`` yields one name per column for every spelling LeRobot writes.

    Imported per-cell rather than at module scope: a module-level import of the
    reader would abort collection of the whole file on a tree without it, so the
    behavioural classes below would never be graded at all.
    """

    @staticmethod
    def _readers():
        from strands_robots.transforms.base import _column_names, _column_width

        return _column_names, _column_width

    @pytest.mark.parametrize(
        ("label", "declared", "expected"),
        [
            ("flat list", list(_ARM), list(_ARM)),
            ("group mapping (teleop_keyboard)", {"motors": list(_ARM)}, list(_ARM)),
            (
                "group mapping, two groups (bimanual)",
                {"left": _BIMANUAL[:6], "right": _BIMANUAL[6:]},
                list(_BIMANUAL),
            ),
            ("index mapping (teleop_gamepad)", {joint: i for i, joint in enumerate(_ARM)}, list(_ARM)),
            # The index is what orders the columns, so a declaration written out
            # of insertion order still names the right column.
            ("index mapping, written out of order", {"gripper": 1, "shoulder_pan": 0}, ["shoulder_pan", "gripper"]),
            ("no declaration", None, []),
        ],
    )
    def test_every_component_is_named_in_column_order(self, label, declared, expected):
        column_names, _ = self._readers()
        assert column_names({"shape": [len(expected)], "names": declared}) == expected, label

    def test_a_group_label_is_not_itself_a_column_name(self):
        """The regression in one line: the mapping's key is a group, not a column."""
        column_names, _ = self._readers()
        names = column_names({"shape": [6], "names": {"motors": list(_ARM)}})
        assert "motors" not in names
        assert len(names) == 6

    def test_the_width_comes_from_a_one_dimensional_shape_only(self):
        _, column_width = self._readers()
        assert column_width({"shape": [6]}) == 6
        # An image feature is not named per component, so it has no width here.
        assert column_width({"shape": [3, 64, 64]}) is None
        assert column_width({}) is None


class TestAGroupedDeclarationRoundTripsTheWholeVector:
    """The pass-through carries every component of a mapping-declared column."""

    @pytest.mark.parametrize(
        ("label", "joints", "declared"),
        [
            ("teleop_keyboard", _ARM, {"motors": list(_ARM)}),
            ("bimanual", _BIMANUAL, {"left": _BIMANUAL[:6], "right": _BIMANUAL[6:]}),
        ],
    )
    def test_the_output_trajectory_is_the_source_trajectory(self, label, joints, declared, source_declaring, tmp_path):
        source_root = source_declaring(joints, declared)
        result, output_root = _transform(source_root, tmp_path)
        assert result.status == "success", result.message

        for key in ("action", "observation.state"):
            source_column = _column(source_root, "local/source", key)
            output_column = _column(output_root, "local/augmented", key)
            assert output_column.shape == source_column.shape, (
                f"{label}: {key} lost columns - source {source_column.shape} -> output {output_column.shape}"
            )
            assert np.array_equal(output_column, source_column), f"{label}: {key} is not the source trajectory"
            assert source_column.shape[1] == len(joints), f"{label}: the fixture did not record {len(joints)} columns"

    def test_the_output_names_the_components_not_the_groups(self, source_declaring, tmp_path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        source_root = source_declaring(_ARM, {"motors": list(_ARM)})
        result, output_root = _transform(source_root, tmp_path)
        assert result.status == "success", result.message
        output = LeRobotDataset(repo_id="local/augmented", root=output_root)
        assert dict(output.meta.features)["action"]["names"] == list(_ARM)


class TestAFlatDeclarationIsUnaffected:
    """Over-reach control: the spelling every shipped recording uses is untouched."""

    def test_a_flat_list_source_still_round_trips(self, source_declaring, tmp_path):
        source_root = source_declaring(_ARM, list(_ARM))
        result, output_root = _transform(source_root, tmp_path)
        assert result.status == "success", result.message
        for key in ("action", "observation.state"):
            assert np.array_equal(
                _column(output_root, "local/augmented", key), _column(source_root, "local/source", key)
            ), key


class TestADeclarationThatCannotDescribeTheVectorIsRefused:
    """A name count that disagrees with the column width is refused, not narrowed."""

    @pytest.mark.parametrize(
        ("label", "declared", "declared_count"),
        [
            # Short: the trailing components would be dropped from the output.
            ("names short of the width", list(_ARM)[:5], 5),
            # Long: the extra column is one no frame supplies, so the recorder
            # would write 0.0 - a travel-to-zero command on a position actuator.
            ("names past the width", [*_ARM, "phantom_joint"], 7),
        ],
    )
    def test_the_refusal_names_both_counts_and_writes_nothing(
        self, label, declared, declared_count, source_declaring, tmp_path
    ):
        source_root = source_declaring(_ARM, declared)
        result, output_root = _transform(source_root, tmp_path)

        assert result.status == "error", f"{label}: a source whose names cannot describe its vector was accepted"
        assert f"{declared_count} name(s)" in result.message, result.message
        assert f"{len(_ARM)}-column" in result.message, result.message
        # Refused at open(), so no partial dataset is left behind to be trained on.
        assert not Path(output_root).exists(), f"{label}: a refused source still wrote an output dataset"

    def test_a_mapping_whose_components_do_not_cover_the_vector_is_refused(self, source_declaring, tmp_path):
        """The mapping spelling gets the same rule, not an exemption for being nested."""
        source_root = source_declaring(_ARM, {"motors": list(_ARM)[:4]})
        result, _ = _transform(source_root, tmp_path)
        assert result.status == "error", result.message
        assert "4 name(s)" in result.message and "6-column" in result.message, result.message
