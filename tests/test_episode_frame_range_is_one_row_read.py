"""An episode's frame range is read from that episode's own metadata row.

Three readers in this package answer the same question - which slice of the
global frame index does episode ``N`` occupy - and each walks a compatibility
ladder over the shapes LeRobot has used for it:

=================================================  ===========================
reader                                             leading rung
=================================================  ===========================
``transforms/base.py::_episode_frame_range``       ``dataset_from_index``
``tools/episode_judge.py`` (image block decode)    ``episode_data_index``
``dataset_recorder.py::load_lerobot_episode``      ``episode_data_index``
=================================================  ===========================

``load_lerobot_episode`` was the only one of the three with **no**
``dataset_from_index`` rung at all: it asked for ``episode_data_index`` and, on
any dataset without it, recomputed the range by accumulating ``length`` over
every *preceding* episode row.

``episode_data_index`` is not a shape any supported LeRobot exposes.
``pyproject.toml`` declares ``lerobot[feetech,dataset]>=0.6.1,<0.7.0``, and the
string occurs **0 times** in ``LeRobotDataset`` on 0.6.2 *and* on 0.5.1 (below
the floor), with ``hasattr(ds, "episode_data_index")`` False on a real instance
of each. So the accumulation was not a fallback - it was the only rung an
accepted index could reach, while the row it was recomputing from already
stated the answer:

    meta.episodes[1] -> {..., 'dataset_from_index': 4, 'dataset_to_index': 11,
                         'length': 7, 'episode_index': 1, ...}

Measured on a real 300-episode dataset, resolving one episode's range:

=========  ==============  ==============  ========  ==============
episode    accumulation    row read        ratio     row fetches
=========  ==============  ==============  ========  ==============
0          0.21 ms         0.095 ms        2x        1 vs 1
50         2.91 ms         0.060 ms        49x       51 vs 1
150        8.02 ms         0.053 ms        151x      151 vs 1
299        14.63 ms        0.047 ms        314x      300 vs 1
=========  ==============  ==============  ========  ==============

Both spellings return the same numbers, so this was never a wrong range - it
was a linear scan standing in for a constant-time read, behind a broad
``except Exception`` whose last resort decodes the dataset frame by frame.

Two docstring claims rested on that rung and were false because of it:

* ``load_lerobot_episode`` justified the ordering of its index guard by saying
  an accepted index "reaches the O(1) ``episode_data_index`` lookup" - naming a
  rung no supported LeRobot provides;
* ``_episode_frame_range`` claimed "Same ladder as ``load_lerobot_episode``",
  and the function it named had neither that leading rung nor its order.

Adding the rung is what makes both claims true, which is why the fix is one
change rather than a code fix plus two prose corrections.

The cells below are two layers. The stand-in layer counts metadata row reads,
so it grades the ladder with no LeRobot installed - which is what CI runs. The
``importorskip`` layer drives a real ``LeRobotDataset`` so the stand-in cannot
drift into agreeing with a shape LeRobot does not write.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import sys
import tomllib
import types
from typing import Any

import pytest

from strands_robots import dataset_recorder as dr
from strands_robots.transforms import base as transforms_base

# Distinct lengths, so *which* episode was resolved is visible in the returned
# range rather than inferred from a count.
_LENGTHS = (4, 7, 3)
_STARTS = (0, 4, 11)


class _CountingEpisodes:
    """Episode metadata rows that record every index a reader asks for.

    The row shape is the one LeRobot 0.6 writes, measured from a real dataset:
    ``dataset_from_index`` / ``dataset_to_index`` / ``length`` /
    ``episode_index``. Rows are plain dicts so a reader can use ``in``,
    ``[...]`` and ``.get`` against them exactly as it does against a real
    ``datasets.Dataset`` row.
    """

    def __init__(self, *, rows: list[dict[str, int]]) -> None:
        self._rows = rows
        self.reads: list[int] = []

    def __getitem__(self, index: int) -> dict[str, int]:
        self.reads.append(int(index))
        return self._rows[int(index)]

    def __len__(self) -> int:
        return len(self._rows)


def _rows_0_6() -> list[dict[str, int]]:
    """Rows as LeRobot 0.6 records them: the range stated on each row."""
    return [
        {
            "episode_index": i,
            "dataset_from_index": start,
            "dataset_to_index": start + n,
            "length": n,
        }
        for i, (start, n) in enumerate(zip(_STARTS, _LENGTHS, strict=True))
    ]


def _rows_length_only() -> list[dict[str, int]]:
    """Rows carrying only ``length`` - the shape the accumulation rung exists for."""
    return [{"episode_index": i, "length": n} for i, n in enumerate(_LENGTHS)]


class _Cell:
    """A tensor-like scalar, as an ``episode_data_index`` entry would be."""

    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _FakeDataset:
    """A ``LeRobotDataset`` stand-in whose metadata reads are observable."""

    def __init__(self, rows: list[dict[str, int]], *, tensor_index: bool = False) -> None:
        self.episodes = _CountingEpisodes(rows=rows)
        self.meta = types.SimpleNamespace(episodes=self.episodes, total_episodes=len(rows))
        self.frame_reads: list[int] = []
        if tensor_index:
            self.episode_data_index = {
                "from": [_Cell(s) for s in _STARTS],
                "to": [_Cell(s + n) for s, n in zip(_STARTS, _LENGTHS, strict=True)],
            }

    def __len__(self) -> int:
        return sum(_LENGTHS)

    def __getitem__(self, index: int) -> dict[str, int]:
        # Reached only by the loader's last-resort frame scan. Recorded so a
        # cell can assert an accepted index never got there.
        self.frame_reads.append(int(index))
        return {"episode_index": 0}


@pytest.fixture
def install_fake(monkeypatch):
    """Return a callable that installs a stand-in ``LeRobotDataset`` module."""

    def _install(dataset: _FakeDataset) -> _FakeDataset:
        module = types.ModuleType("lerobot.datasets.lerobot_dataset")
        module.LeRobotDataset = lambda repo_id=None, root=None: dataset  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
        monkeypatch.setitem(sys.modules, "lerobot.datasets", types.ModuleType("lerobot.datasets"))
        monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)
        return dataset

    return _install


def _transform_range(dataset: _FakeDataset, episode: int) -> tuple[int, int]:
    """Drive the transform reader with the same stand-in, via its own method.

    ``_episode_frame_range`` reads nothing but ``self.ds``, so a namespace
    carrying that one attribute exercises the shipped ladder without building a
    ``_SourceDataset`` (whose constructor reads a schema this question does not
    involve).
    """
    holder = types.SimpleNamespace(ds=dataset)
    return transforms_base._SourceDataset._episode_frame_range(holder, episode)  # type: ignore[arg-type]


# ── the regression: one row read, whatever the index ────────────────


class TestTheRangeIsReadFromTheEpisodesOwnRow:
    """The loader reads the requested episode's row and no other."""

    @pytest.mark.parametrize("episode", [0, 1, 2])
    def test_only_the_requested_episodes_row_is_read(self, install_fake, episode):
        dataset = install_fake(_FakeDataset(_rows_0_6()))
        dr.load_lerobot_episode("local/probe", episode=episode, root="/probe")
        assert dataset.episodes.reads == [episode], (
            f"resolving episode {episode} read metadata rows {dataset.episodes.reads}; "
            "the range is stated on that episode's own row, so one read answers it"
        )

    def test_the_row_reads_do_not_grow_with_the_index(self, install_fake):
        counts = []
        for episode in range(len(_LENGTHS)):
            dataset = install_fake(_FakeDataset(_rows_0_6()))
            dr.load_lerobot_episode("local/probe", episode=episode, root="/probe")
            counts.append(len(dataset.episodes.reads))
        assert len(set(counts)) == 1, (
            f"metadata row reads per episode were {counts}; a constant-time read does not "
            "grow with the index, and a linear one is what the accumulation rung was doing"
        )


# ── what the change must not disturb ────────────────────────────────


class TestTheResolvedRangeIsUnchanged:
    """Every rung still resolves the range it always did.

    These hold before and after the change: the point of adding a leading rung
    is that the answer does not move, only the number of reads it costs.
    """

    @pytest.mark.parametrize("episode", [0, 1, 2])
    def test_the_recorded_range_is_returned(self, install_fake, episode):
        install_fake(_FakeDataset(_rows_0_6()))
        _, start, length = dr.load_lerobot_episode("local/probe", episode=episode, root="/probe")
        assert (start, length) == (_STARTS[episode], _LENGTHS[episode])

    def test_a_pre_0_6_dataset_still_resolves_through_the_tensor_index(self, install_fake):
        dataset = install_fake(_FakeDataset(_rows_length_only(), tensor_index=True))
        _, start, length = dr.load_lerobot_episode("local/probe", episode=2, root="/probe")
        assert (start, length) == (_STARTS[2], _LENGTHS[2])
        assert dataset.frame_reads == [], "the tensor rung answered, so no frame scan was needed"

    def test_a_dataset_with_neither_still_accumulates_lengths(self, install_fake):
        dataset = install_fake(_FakeDataset(_rows_length_only()))
        _, start, length = dr.load_lerobot_episode("local/probe", episode=2, root="/probe")
        assert (start, length) == (_STARTS[2], _LENGTHS[2])
        assert dataset.frame_reads == [], "the accumulation answered, so no frame scan was needed"

    def test_the_range_columns_are_the_authority_not_the_length_column(self, install_fake):
        """A row states its range twice over; the range columns win.

        ``length`` and ``dataset_to_index - dataset_from_index`` agree on every
        row LeRobot writes (measured: 0 disagreements across a recorded
        dataset), so a reader taking either is indistinguishable in practice.
        The slice a caller then indexes is ``[from, to)``, which makes the range
        columns the authority and ``length`` a derived convenience - pinned
        here so a reader cannot quietly switch to the second source.
        """
        rows = _rows_0_6()
        rows[1]["length"] = _LENGTHS[1] + 5
        install_fake(_FakeDataset(rows))
        _, start, length = dr.load_lerobot_episode("local/probe", episode=1, root="/probe")
        assert (start, length) == (_STARTS[1], _LENGTHS[1]), (
            "the row's length column disagreed with its range columns; the range is what "
            "the frame slice is taken from, so it is the field a reader must resolve"
        )

    def test_the_frame_scan_is_not_reached_for_a_0_6_dataset(self, install_fake):
        dataset = install_fake(_FakeDataset(_rows_0_6()))
        dr.load_lerobot_episode("local/probe", episode=2, root="/probe")
        assert dataset.frame_reads == [], (
            "the frame scan decodes the dataset one frame at a time; an episode whose row "
            "states its range must never reach it"
        )


# ── the ladder is shared, not merely similar ────────────────────────

_LADDER_KEYS = ("dataset_from_index", "episode_data_index")


def _readers_of_the_frame_range() -> dict[str, str]:
    """Every function in the package that resolves an episode frame range.

    Derived rather than listed: a reader is any function whose body mentions
    one of the shapes LeRobot has used to record the range. That keeps a reader
    added later under the same rule instead of inheriting an exemption by being
    absent from a tuple.
    """
    root = pathlib.Path(transforms_base.__file__).parent.parent
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if not any(key in text for key in _LADDER_KEYS):
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            body = ast.unparse(node)
            if any(key in body for key in _LADDER_KEYS):
                found[f"{path.relative_to(root)}::{node.name}"] = body
    return found


class TestBothReadersUseTheSameLadder:
    """No reader of the range depends only on the rung LeRobot dropped."""

    def test_the_reader_set_is_not_empty(self):
        readers = _readers_of_the_frame_range()
        assert len(readers) >= 3, (
            f"found only {sorted(readers)}; three readers resolve this range, so a scan "
            "returning fewer has stopped seeing them and grades nothing"
        )

    def test_every_reader_reads_the_recorded_row(self):
        offenders = sorted(
            where for where, body in _readers_of_the_frame_range().items() if "dataset_from_index" not in body
        )
        assert offenders == [], (
            f"{offenders} resolve an episode's frame range without reading dataset_from_index. "
            "No LeRobot in the declared range exposes episode_data_index, so a reader without "
            "that rung falls through to a linear recomputation of a number the row already states"
        )

    def test_the_transform_readers_parity_claim_holds(self):
        """The transform reader says it shares this ladder; assert it does."""
        loader = inspect.getsource(dr.load_lerobot_episode)
        reader = inspect.getsource(transforms_base._SourceDataset._episode_frame_range)
        claim = inspect.getdoc(transforms_base._SourceDataset._episode_frame_range) or ""
        assert "load_lerobot_episode" in claim, (
            "the parity claim naming load_lerobot_episode has gone from the docstring, so "
            "this cell no longer grades a claim the tree makes"
        )
        order = [tuple(sorted(_LADDER_KEYS, key=source.index)) for source in (loader, reader)]
        assert order[0] == order[1], (
            f"the loader tries {order[0]} and the transform reader {order[1]}; the docstring "
            "claims the same rungs in the same order, so the two must agree"
        )


# ── premises: why the dropped rung cannot be relied on ──────────────


class TestPremises:
    """Facts about LeRobot that make the leading rung the right one.

    These hold on both trees - they are properties of the dependency, not of
    the change - and they are what the regression cells above rest on.
    """

    def test_the_declared_lerobot_range_is_the_0_6_series(self):
        pyproject = pathlib.Path(transforms_base.__file__).parent.parent.parent / "pyproject.toml"
        declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        extras = declared["project"]["optional-dependencies"]
        pins = [dep for group in extras.values() for dep in group if dep.startswith("lerobot[")]
        assert pins, "no lerobot pin found, so the supported range cannot be read"
        assert all(">=0.6" in pin for pin in pins), (
            f"lerobot pins are {pins}; the leading rung is chosen because every supported "
            "version records the range on the episode row"
        )

    def test_no_supported_lerobot_exposes_the_tensor_index(self):
        pytest.importorskip("lerobot")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        source = inspect.getsource(LeRobotDataset)
        assert "episode_data_index" not in source, (
            "this LeRobot exposes episode_data_index again; the fallback rung is reachable, "
            "and the leading rung's justification needs re-reading rather than deleting"
        )


# ── fidelity: the stand-in describes what LeRobot writes ────────────


class TestAgainstARealDataset:
    """Drive the shipped readers against a genuine on-disk LeRobotDataset."""

    @pytest.fixture
    def recorded(self, tmp_path):
        pytest.importorskip("lerobot.datasets.lerobot_dataset")
        import numpy as np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = tmp_path / "dataset"
        features = {
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
            "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        }
        writer = LeRobotDataset.create(
            repo_id="local/probe", fps=10, root=str(root), features=features, robot_type="probe"
        )
        for episode, length in enumerate(_LENGTHS):
            for frame in range(length):
                sample = np.array([episode, frame], dtype=np.float32)
                writer.add_frame({"observation.state": sample, "action": sample, "task": "probe"})
            writer.save_episode()
        writer.finalize()
        return root

    def test_the_row_shape_the_stand_in_models_is_the_one_lerobot_writes(self, recorded):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id="local/probe", root=str(recorded))
        assert not hasattr(dataset, "episode_data_index")
        row: dict[str, Any] = dataset.meta.episodes[1]
        assert {"dataset_from_index", "dataset_to_index", "length"} <= set(row)
        assert (row["dataset_from_index"], row["dataset_to_index"]) == (_STARTS[1], _STARTS[1] + _LENGTHS[1])

    @pytest.mark.parametrize("episode", [0, 1, 2])
    def test_both_readers_resolve_the_recorded_range(self, recorded, episode):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id="local/probe", root=str(recorded))
        expected = (_STARTS[episode], _STARTS[episode] + _LENGTHS[episode])
        assert _transform_range(dataset, episode) == expected
        _, start, length = dr.load_lerobot_episode("local/probe", episode=episode, root=str(recorded))
        assert (start, start + length) == expected, (
            "the loader and the transform reader answer the same question about the same "
            "dataset, so a range either resolves in both or in neither"
        )
