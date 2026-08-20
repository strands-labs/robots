"""The episode index a replay resolves is a non-negative whole number.

``episode`` selects *which* recorded trajectory is sent to the actuators, so an
index the dataset cannot be indexed by is not a slow replay - it is the wrong
motion. Three public surfaces resolve one: ``PolicyRunner.replay``,
``SimEngine.replay_episode`` (which forwards to it) and the
``load_lerobot_episode`` loader they share.

Pre-fix the only guard was ``if episode < 0`` inside the loader, which gave a
verdict to three classes of value it could not honor. Measured with the fake
dataset below (three episodes of lengths 10/20/30, so *which* episode was
resolved is visible in the returned frame range rather than inferred):

===============  ==================================================  ==========
``episode=``     pre-fix outcome                                     resolved
===============  ==================================================  ==========
``True``         SUCCESS start=10 length=20                          episode 1
``False``        SUCCESS start=0 length=10                           episode 0
``np.True_``     SUCCESS start=10 length=20                          episode 1
``2.5``          ValueError "Episode 2.5 has no frames"              -
``nan``          ValueError "Episode nan has no frames"              -
``inf``          ValueError "Episode inf out of range (0-2)"         -
``'0'``          TypeError from the ``<`` comparison                 -
``[0]``          TypeError from the ``<`` comparison                 -
``None``         TypeError from the ``<`` comparison                 -
===============  ==================================================  ==========

Three separate defects in one parameter:

* a **bool resolved an episode** - ``True < 0`` is False and the episode table
  is then indexed with it, so ``episode=True`` replayed episode 1 under
  ``status="success"``. No caller passing a flag meant "the second episode";
* a non-integral or non-finite index was **blamed on the dataset** ("has no
  frames"), naming the data rather than the index, and only after a
  full-length boundary scan;
* a str/list/None raised **TypeError**, which is not the ``ValueError`` the
  loader documents as its refusal channel, nor the structured envelope
  ``replay`` documents as its own.

The fix applies ``non_negative_whole_number_error`` - the shared rule whose own
docstring already names ``replay_episode``, the same quantity on the
neighbouring teleop surface - so the refusal is identical across both spellings
rather than merely equivalent in verdict.

The structural sweep below is what keeps that true as surfaces are added, so
its *discovery* has to be as wide as the quantity. It used to admit a surface
only when a parameter was literally named ``episode``, which is one spelling of
two: an index also arrives inside a **collection** - an entry of the
``episodes`` list ``record_deterministic_verdicts`` records, a key of the
``human_labels`` mapping ``measure_agreement`` calibrates against. Both apply
the shared rule, and both were invisible to the sweep, so
``test_the_surface_set_is_exactly_the_pinned_one`` reported a clean tree while
grading 9 of 11 surfaces and ``test_every_surface_validates_or_forwards`` never
ran on either. Deleting both guards left the whole file passing (203 passed),
while the behaviour they hold degraded exactly as the table above:
``{"episode": True}`` recorded a verdict against **episode 1** and left episode
0 unlabeled, and ``measure_agreement({True: ...})`` reported
``quality_agreement 1.0`` calibrated against episode 1 - the number that
decides whether a judge is trusted to filter training data.

Discovery is therefore derived from each surface's own ``Args:`` entry: a
parameter named ``episode``, or one whose documented description says it
carries an episode index. That keeps the rule a property of the tree rather
than a list to maintain, and it holds the collection spellings to the same
contract - ``filter_episodes``, which *returns* indices rather than resolving
one, stays out on the same rule.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
import re
import sys
import types

import numpy as np
import pytest

from strands_robots import episode_labels
from strands_robots.utils import non_negative_whole_number_error

# ── the fake dataset ────────────────────────────────────────────────
#
# Three episodes of distinct lengths so the returned ``(start, length)``
# identifies which episode was resolved. ``episode_data_index`` is a list per
# key, exactly as a real ``LeRobotDataset`` exposes it - which is why a bool
# indexed it successfully.

_EPISODE_LENGTHS = (10, 20, 30)
_STARTS = {0: 0, 10: 1, 30: 2}  # frame start -> episode number


class _Cell:
    """Stands in for the tensor cell whose ``.item()`` the loader reads."""

    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _Meta:
    total_episodes = len(_EPISODE_LENGTHS)
    episodes = [{"length": n} for n in _EPISODE_LENGTHS]


class _FakeDataset:
    """Minimal LeRobotDataset stand-in; records that it was constructed."""

    constructions: list[dict[str, object]] = []
    scans: list[int] = []
    fps = 30

    def __init__(self, repo_id=None, root=None):
        type(self).constructions.append({"repo_id": repo_id, "root": root})
        self.meta = _Meta()
        self.episode_data_index = {
            "from": [_Cell(0), _Cell(10), _Cell(30)],
            "to": [_Cell(10), _Cell(30), _Cell(60)],
        }

    def __len__(self) -> int:
        return sum(_EPISODE_LENGTHS)

    def __getitem__(self, idx):
        # Reached only by the loader's last-resort boundary scan. Recorded so a
        # test can assert an accepted index used the O(1) index instead.
        type(self).scans.append(idx)
        if idx < 10:
            ep = 0
        elif idx < 30:
            ep = 1
        else:
            ep = 2
        return {"episode_index": ep}


def superseded_non_negative_test(episode: object) -> bool:
    """The guard this change replaces: ``if episode < 0: raise``.

    Reproduced as a function so the premise class below *measures* what that
    comparison did with each probe value. Written inline as
    ``assert (True < 0) is False`` the same claim is decided when it is typed:
    it restates an outcome instead of measuring one, which is the shape
    ``tests/test_no_vacuous_comparisons.py`` refuses across the repository.
    Taking the value as a parameter is what makes the comparison a measurement.

    Args:
        episode: The value a caller supplied as an episode index.

    Returns:
        Whether the superseded comparison refused ``episode``.

    Raises:
        TypeError: If ``episode`` cannot be ordered against an int at all,
            which is itself one of the three outcomes the premise records.
    """
    return bool(episode < 0)  # type: ignore[operator]


@pytest.fixture
def fake_lerobot(monkeypatch):
    """Install a fake ``lerobot`` so the loader runs with no hub round-trip."""
    _FakeDataset.constructions = []
    _FakeDataset.scans = []
    ld = types.ModuleType("lerobot.datasets.lerobot_dataset")
    ld.LeRobotDataset = _FakeDataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", types.ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", ld)
    return _FakeDataset


# ── the probe set ───────────────────────────────────────────────────

UNUSABLE = [
    pytest.param(True, id="True"),
    pytest.param(False, id="False"),
    pytest.param(np.True_, id="np.True_"),
    pytest.param(2.5, id="2.5"),
    pytest.param(-1, id="-1"),
    pytest.param(-1.0, id="-1.0"),
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="inf"),
    pytest.param(-math.inf, id="-inf"),
    pytest.param("0", id="str"),
    pytest.param([0], id="list"),
    pytest.param(None, id="None"),
    pytest.param({"episode": 0}, id="dict"),
    pytest.param(np.float64(1.5), id="np.float64(1.5)"),
]

ACCEPTED = [
    pytest.param(0, 0, id="0"),
    pytest.param(1, 1, id="1"),
    pytest.param(2, 2, id="2"),
    pytest.param(2.0, 2, id="2.0"),
    pytest.param(np.int64(1), 1, id="np.int64(1)"),
    pytest.param(np.float64(2.0), 2, id="np.float64(2.0)"),
]


# The three outcomes the superseded ``episode < 0`` produced, split by what that
# comparison *did* rather than by what the value looks like. Referenced by probe
# id so the values are never spelled twice, and covered exhaustively by
# ``TestThePremise.test_the_premise_covers_every_probe_value``.
_UNUSABLE_BY_ID = {param.id: param.values[0] for param in UNUSABLE}

_PREMISE_GROUPS = {
    "passed": ("True", "False", "np.True_", "2.5", "nan", "inf", "np.float64(1.5)"),
    "refused": ("-1", "-1.0", "-inf"),
    "unorderable": ("str", "list", "None", "dict"),
}


def _premise_values(group: str) -> list[object]:
    """The probe values in ``group``, resolved through their ids."""
    return [_UNUSABLE_BY_ID[pid] for pid in _PREMISE_GROUPS[group]]


# ── the premise, measured rather than asserted ──────────────────────


class TestThePremise:
    """Why a bare ``< 0`` test could not refuse these values.

    Measured against the language and NumPy themselves, so the mechanism is a
    measurement rather than a claim about one: every probe value is pushed
    through :func:`superseded_non_negative_test`, the comparison this change
    replaces, and grouped by what that comparison actually did with it. The
    grouping is exhaustive over ``UNUSABLE`` -- asserted below, so a probe value
    added later cannot silently escape the premise.
    """

    @pytest.mark.parametrize("value", _premise_values("passed"), ids=_PREMISE_GROUPS["passed"])
    def test_the_superseded_comparison_did_not_refuse_these(self, value):
        assert superseded_non_negative_test(value) is False

    @pytest.mark.parametrize("value", _premise_values("refused"), ids=_PREMISE_GROUPS["refused"])
    def test_the_superseded_comparison_did_refuse_these(self, value):
        # These the old comparison caught, which is why the shared rule reports
        # them with the same message rather than a new one.
        assert superseded_non_negative_test(value) is True

    @pytest.mark.parametrize("value", _premise_values("unorderable"), ids=_PREMISE_GROUPS["unorderable"])
    def test_the_superseded_comparison_raised_instead_of_refusing(self, value):
        with pytest.raises(TypeError):
            superseded_non_negative_test(value)

    def test_the_premise_covers_every_probe_value(self):
        """Guard: no ``UNUSABLE`` entry escapes the premise, and none is claimed twice."""
        grouped = [pid for ids in _PREMISE_GROUPS.values() for pid in ids]
        assert len(grouped) == len(set(grouped)), "a probe value is claimed by two groups"
        assert sorted(grouped) == sorted(_UNUSABLE_BY_ID)

    def test_a_bool_then_indexes_a_list_as_an_int(self):
        # The mechanism: ``episode_data_index["from"][True]`` is element 1.
        table = [_Cell(0), _Cell(10), _Cell(30)]
        assert table[True].item() == 10
        assert table[False].item() == 0

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_shared_rule_refuses_every_probe(self, value):
        assert non_negative_whole_number_error(value, "episode", "ctx") is not None

    @pytest.mark.parametrize("value,expected", ACCEPTED)
    def test_the_shared_rule_accepts_every_usable_index(self, value, expected):
        assert non_negative_whole_number_error(value, "episode", "ctx") is None
        assert int(value) == expected


# ── the loader ──────────────────────────────────────────────────────


class TestLoadLerobotEpisodeRefusesAnUnusableIndex:
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_raises_value_error_naming_the_parameter(self, value, fake_lerobot):
        from strands_robots.dataset_recorder import load_lerobot_episode

        with pytest.raises(ValueError) as excinfo:
            load_lerobot_episode("fake/repo", value)
        msg = str(excinfo.value)
        assert "load_lerobot_episode" in msg
        assert "episode" in msg

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_refusal_is_the_shared_rule_verbatim(self, value, fake_lerobot):
        from strands_robots.dataset_recorder import load_lerobot_episode

        expected = non_negative_whole_number_error(value, "episode", "load_lerobot_episode")
        assert expected is not None
        with pytest.raises(ValueError) as excinfo:
            load_lerobot_episode("fake/repo", value)
        assert str(excinfo.value) == expected

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_no_dataset_is_constructed(self, value, fake_lerobot):
        """The refusal lands before the hub download, not after it."""
        from strands_robots.dataset_recorder import load_lerobot_episode

        with pytest.raises(ValueError):
            load_lerobot_episode("fake/repo", value)
        assert fake_lerobot.constructions == []
        assert fake_lerobot.scans == []

    def test_a_bool_no_longer_resolves_an_episode(self, fake_lerobot):
        """The load-bearing row: ``True`` resolved episode 1 pre-fix."""
        from strands_robots.dataset_recorder import load_lerobot_episode

        with pytest.raises(ValueError):
            load_lerobot_episode("fake/repo", True)

    def test_the_dataset_is_no_longer_blamed_for_a_non_integral_index(self, fake_lerobot):
        from strands_robots.dataset_recorder import load_lerobot_episode

        with pytest.raises(ValueError) as excinfo:
            load_lerobot_episode("fake/repo", 2.5)
        msg = str(excinfo.value)
        assert "has no frames" not in msg
        assert "out of range" not in msg

    @pytest.mark.parametrize("value", ["0", [0], None])
    def test_a_non_numeric_index_no_longer_raises_type_error(self, value, fake_lerobot):
        """The loader documents ``ValueError`` as its refusal channel."""
        from strands_robots.dataset_recorder import load_lerobot_episode

        with pytest.raises(ValueError):
            load_lerobot_episode("fake/repo", value)


class TestLoadLerobotEpisodeAcceptedDomain:
    @pytest.mark.parametrize("value,expected_episode", ACCEPTED)
    def test_an_accepted_index_resolves_that_episode(self, value, expected_episode, fake_lerobot):
        from strands_robots.dataset_recorder import load_lerobot_episode

        _, start, length = load_lerobot_episode("fake/repo", value)
        assert _STARTS[start] == expected_episode
        assert length == _EPISODE_LENGTHS[expected_episode]

    def test_an_integral_float_uses_the_index_not_the_full_scan(self, fake_lerobot):
        """``2.0`` fell through to an O(len(dataset)) boundary scan pre-fix.

        The guard coerces with ``int()`` once it has round-tripped the value,
        so an accepted index reaches ``episode_data_index`` directly.
        """
        from strands_robots.dataset_recorder import load_lerobot_episode

        _, start, length = load_lerobot_episode("fake/repo", 2.0)
        assert (start, length) == (30, 30)
        assert fake_lerobot.scans == []

    def test_an_out_of_range_whole_number_is_still_a_range_refusal(self, fake_lerobot):
        """Range is the dataset's business; the domain guard does not take it."""
        from strands_robots.dataset_recorder import load_lerobot_episode

        with pytest.raises(ValueError, match="out of range"):
            load_lerobot_episode("fake/repo", 99)


# ── the replay facade ───────────────────────────────────────────────


def _runner():
    from strands_robots.simulation.policy_runner import PolicyRunner
    from tests.simulation.test_policy_runner import FakeSim

    sim = FakeSim()
    return PolicyRunner(sim), sim


class TestReplayRefusesAnUnusableIndex:
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_returns_the_error_envelope(self, value, fake_lerobot):
        runner, _ = _runner()
        result = runner.replay(repo_id="fake/repo", episode=value)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "replay" in text
        assert "episode" in text

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_refusal_is_the_shared_rule_verbatim(self, value, fake_lerobot):
        runner, _ = _runner()
        expected = non_negative_whole_number_error(value, "episode", "replay")
        assert expected is not None
        result = runner.replay(repo_id="fake/repo", episode=value)
        assert result["content"][0]["text"] == expected

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_never_raises(self, value, fake_lerobot):
        """``replay`` documents the status dict as its only failure channel."""
        runner, _ = _runner()
        assert runner.replay(repo_id="fake/repo", episode=value)["status"] == "error"

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_no_dataset_is_downloaded_and_no_action_is_sent(self, value, fake_lerobot):
        runner, sim = _runner()
        runner.replay(repo_id="fake/repo", episode=value)
        assert fake_lerobot.constructions == []
        assert getattr(sim, "sent_actions", []) == []

    def test_a_bool_no_longer_replays_the_second_episode(self, fake_lerobot):
        runner, sim = _runner()
        result = runner.replay(repo_id="fake/repo", episode=True)
        assert result["status"] == "error"
        assert getattr(sim, "sent_actions", []) == []


class TestTheLoaderAndTheFacadeAgree:
    """One value, one verdict, whichever surface the caller reached."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_both_refuse_every_unusable_index(self, value, fake_lerobot):
        from strands_robots.dataset_recorder import load_lerobot_episode

        runner, _ = _runner()
        assert runner.replay(repo_id="fake/repo", episode=value)["status"] == "error"
        with pytest.raises(ValueError):
            load_lerobot_episode("fake/repo", value)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_two_messages_differ_only_in_the_context(self, value, fake_lerobot):
        facade = non_negative_whole_number_error(value, "episode", "replay")
        loader = non_negative_whole_number_error(value, "episode", "load_lerobot_episode")
        assert facade is not None and loader is not None
        assert facade.replace("replay:", "", 1) == loader.replace("load_lerobot_episode:", "", 1)


# ── the guard is the shared rule, not a second copy of it ───────────


class TestTheRefusalAddsNothingLocal:
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_neither_surface_narrows_or_widens_the_shared_domain(self, value, fake_lerobot):
        """No carve-out: the accepted set is exactly the shared rule's."""
        from strands_robots.dataset_recorder import load_lerobot_episode

        runner, _ = _runner()
        shared_refuses = non_negative_whole_number_error(value, "episode", "replay") is not None
        assert shared_refuses is True
        assert runner.replay(repo_id="fake/repo", episode=value)["status"] == "error"
        with pytest.raises(ValueError):
            load_lerobot_episode("fake/repo", value)

    def test_the_old_bare_comparison_message_is_gone(self):
        source = pathlib.Path(
            inspect.getsourcefile(__import__("strands_robots.dataset_recorder", fromlist=["x"]))  # type: ignore[arg-type]
        ).read_text()
        assert "Episode index must be non-negative, got" not in source


# ── structural sweep ────────────────────────────────────────────────

_REPLAY_EPISODE_SURFACES = {
    ("strands_robots/simulation/policy_runner.py", "replay"),
    ("strands_robots/simulation/base.py", "replay_episode"),
    ("strands_robots/dataset_recorder.py", "load_lerobot_episode"),
    # The episode-label surfaces resolve the same quantity: which recorded
    # episode a verdict/annotation/read is about. Each applies the shared rule
    # itself (the judge tools return the refusal as a structured error dict).
    ("strands_robots/episode_labels.py", "deterministic_verdict"),
    ("strands_robots/episode_labels.py", "annotate_episode"),
    ("strands_robots/tools/episode_judge.py", "load_episode"),
    ("strands_robots/tools/episode_judge.py", "sample_frames"),
    ("strands_robots/tools/episode_judge.py", "read_predicate_verdict"),
    ("strands_robots/tools/episode_judge.py", "write_label"),
    # The same quantity arriving inside a collection: an entry of the recorded
    # ``episodes`` list, and a key of the human holdout mapping. A spelling is
    # not a different contract, so these are graded like the bare parameter.
    ("strands_robots/episode_labels.py", "record_deterministic_verdicts"),
    ("strands_robots/episode_labels.py", "measure_agreement"),
}

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# A parameter carries an episode index when its own ``Args:`` entry says so.
# Derived from the docstring rather than from a list of parameter names, so a
# collection spelling (``episodes``, ``human_labels``) is discovered on the
# same rule as the bare ``episode``.
_CARRIES_AN_INDEX = re.compile(r"episode\W*index|index of the episode|``episode``")
_ARG_ENTRY = re.compile(r"^\s{4,}([*\w]+):\s*(.*)$")


def _arg_descriptions(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """Map each documented parameter to its ``Args:`` description.

    Continuation lines are folded into the entry they belong to, so a
    description that wraps is matched as one string.
    """
    lines = (ast.get_docstring(node) or "").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Args:")
    except StopIteration:
        return {}
    described: dict[str, str] = {}
    current: str | None = None
    entry_indent: int | None = None
    for line in lines[start + 1 :]:
        if line.strip() in ("Returns:", "Raises:", "Yields:", "Example:", "Examples:", "Note:"):
            break
        match = _ARG_ENTRY.match(line)
        indent = len(line) - len(line.lstrip())
        if match and (entry_indent is None or indent == entry_indent):
            entry_indent = indent
            current = match.group(1).lstrip("*")
            described[current] = match.group(2)
        elif current and line.strip():
            described[current] += " " + line.strip()
    return described


def _public_episode_surfaces(source: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every public function that resolves an episode index, in either spelling."""
    tree = ast.parse(source)
    found: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        if any(a.arg == "episode" for a in args):
            found.append(node)
            continue
        described = _arg_descriptions(node)
        if any(a.arg in described and _CARRIES_AN_INDEX.search(described[a.arg]) for a in args):
            found.append(node)
    return found


def _validates_or_forwards(node: ast.AST) -> bool:
    """True when the surface applies the shared rule or forwards verbatim."""
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        src = ast.unparse(call)
        if "non_negative_whole_number_error" in src and "episode" in src:
            return True
        # Forwarding verbatim to a surface that does validate.
        if ("replay" in src or "load_lerobot_episode" in src) and "episode=episode" in src:
            return True
    return False


class TestNoEpisodeIndexSurfaceDrifts:
    def test_the_surface_set_is_exactly_the_pinned_one(self):
        """Non-vacuity: a new surface joins the sweep or fails this test."""
        discovered = set()
        for path in sorted((_REPO_ROOT / "strands_robots").rglob("*.py")):
            for node in _public_episode_surfaces(path.read_text()):
                discovered.add((str(path.relative_to(_REPO_ROOT)), node.name))
        assert discovered == _REPLAY_EPISODE_SURFACES

    @pytest.mark.parametrize("rel_path,func_name", sorted(_REPLAY_EPISODE_SURFACES))
    def test_every_surface_validates_or_forwards(self, rel_path, func_name):
        source = (_REPO_ROOT / rel_path).read_text()
        node = next(n for n in _public_episode_surfaces(source) if n.name == func_name)
        assert _validates_or_forwards(node), f"{rel_path}::{func_name} neither validates nor forwards `episode`"

    def test_the_sweep_fails_on_a_planted_defect(self):
        """Meta-test: the sweep is not vacuously true."""
        planted = ast.parse("def replay(self, repo_id, episode=0):\n    return load_lerobot_episode(repo_id, 0)\n")
        node = planted.body[0]
        assert not _validates_or_forwards(node)

    def test_discovery_finds_an_index_that_arrives_inside_a_collection(self):
        """A collection-valued spelling is the same surface, so it is discovered.

        The narrow rule admitted a surface only when a parameter was literally
        named ``episode``. A planted surface taking a list of per-episode dicts
        - the ``record_deterministic_verdicts`` shape - was therefore invisible,
        and an unguarded one passed the sweep.
        """
        planted = ast.parse(
            "def record(root, verdicts):\n"
            '    """Record verdicts.\n'
            "\n"
            "    Args:\n"
            "        root: Dataset root.\n"
            "        verdicts: Per-episode dicts, each carrying an episode index.\n"
            '    """\n'
            "    return None\n"
        )
        discovered = _public_episode_surfaces(ast.unparse(planted))
        assert [n.name for n in discovered] == ["record"]
        assert not _validates_or_forwards(discovered[0]), "an unguarded collection surface must not pass"

    def test_discovery_leaves_a_surface_that_only_returns_indices_out(self):
        """``filter_episodes`` selects episodes; it resolves no caller index.

        The boundary is the same derived rule, not an exemption: no parameter
        of it is documented as carrying an index, so requiring it to apply the
        shared rule would demand a guard on a value it never receives.
        """
        source = (_REPO_ROOT / "strands_robots/episode_labels.py").read_text()
        names = {node.name for node in _public_episode_surfaces(source)}
        assert "filter_episodes" not in names
        assert {"record_deterministic_verdicts", "measure_agreement"} <= names


class TestTheCollectionSpellingsRefuseAnUnusableIndex:
    """The two collection-valued surfaces honour the shared domain end to end.

    The sweep above is structural; these drive the real functions, because what
    the sweep protects is the behaviour: an index outside the domain must be
    refused rather than resolve a different episode. ``True`` is the case worth
    naming - it is the value a misplaced flag takes, ``int(True)`` is 1, and
    every episode table is indexable by 1.
    """

    @staticmethod
    def _labeled_dataset(tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        episode_labels.record_deterministic_verdicts(
            root, [{"episode": 0, "success": True}, {"episode": 1, "success": True}], benchmark="b"
        )
        episode_labels.annotate_episode(root, 0, quality="low", model="m")
        episode_labels.annotate_episode(root, 1, quality="high", model="m")
        return root

    @pytest.mark.parametrize("bad", [True, False, 2.5, -1, "0", None, [0]])
    def test_a_recorded_verdict_index_outside_the_domain_is_refused(self, tmp_path, bad):
        root = tmp_path / "ds"
        root.mkdir()
        with pytest.raises(ValueError, match="non-negative whole number"):
            episode_labels.record_deterministic_verdicts(root, [{"episode": bad, "success": True}])
        assert not episode_labels.labels_path(root).exists(), "a refused verdict must write no sidecar"

    @pytest.mark.parametrize("bad", [True, False, 2.5, -1, "0", None])
    def test_a_holdout_episode_key_outside_the_domain_is_refused(self, tmp_path, bad):
        root = self._labeled_dataset(tmp_path)
        with pytest.raises(ValueError, match="non-negative whole number"):
            episode_labels.measure_agreement(root, {bad: {"quality": "high"}})

    def test_both_refusals_name_the_episode_index(self, tmp_path):
        """The refusal names the quantity, so the caller knows what to fix.

        ``measure_agreement`` used to report only ``human_labels key``, naming
        the container slot rather than the episode index it holds - which is
        also why the sweep's recogniser, which requires the shared rule to be
        applied to the episode, could not see the guard that was there.
        """
        root = self._labeled_dataset(tmp_path)
        with pytest.raises(ValueError) as recorded:
            episode_labels.record_deterministic_verdicts(root, [{"episode": True, "success": True}])
        with pytest.raises(ValueError) as calibrated:
            episode_labels.measure_agreement(root, {True: {"quality": "high"}})
        for message in (str(recorded.value), str(calibrated.value)):
            assert "episode" in message, message
            assert "True" in message, message

    def test_an_index_inside_the_domain_still_resolves_that_episode(self, tmp_path):
        """Control: the guard refuses the domain, not the callers."""
        root = self._labeled_dataset(tmp_path)
        assert episode_labels.deterministic_verdict(root, 1)["success"] is True
        agreement = episode_labels.measure_agreement(root, {1: {"quality": "high"}})
        assert agreement["episodes_compared"] == 1
        assert agreement["quality_agreement"] == 1.0


class TestTheTeleopSpellingStaysOutOfScope:
    """``replay_episode`` on the lerobot CLI tool is deliberately untouched.

    It is the same quantity and already carries the same shared rule, applied
    from ``lerobot_teleoperate``'s validation table. ``build_lerobot_command``
    is the CLI string builder its caller validates for, so it is not a second
    unguarded surface - it is downstream of one. Pinned so the boundary is a
    stated scope rather than an omission.
    """

    def test_the_teleop_tool_applies_the_same_shared_rule(self):
        source = (_REPO_ROOT / "strands_robots/tools/lerobot_teleoperate.py").read_text()
        assert '("replay_episode", non_negative_whole_number_error)' in source
