"""``STRANDS_MESH`` has one vocabulary owner, and it reports a value it ignores.

The variable is read by two resolvers that own opposite halves of its
vocabulary: :func:`strands_robots.robot._mesh_env_opt_in` reads the affirmative
spellings, and :func:`strands_robots.mesh.core.mesh_disabled_by_env` reads the
kill-switch ones. Neither can report a typo alone, because each half legitimately
treats the other's spellings as "not mine" -- so a third value that belongs to
neither was silently ignored on a switch documented to override an explicit
``mesh=True``.

Measured before the fix, with ``mesh=True`` passed explicitly:

    STRANDS_MESH='false'      -> None (gate refused)
    STRANDS_MESH='off'        -> Mesh CONSTRUCTED (gate passed), SILENT
    STRANDS_MESH='disabled'   -> Mesh CONSTRUCTED (gate passed), SILENT

``off`` is a spelling this package teaches elsewhere -- ``STRANDS_ROBOT_MESH_DC``
accepts ``("off", "0", "false", "no")`` -- so an operator can arrive at it
without inventing anything.

These tests pin the report and the delegation, not a wider vocabulary: whether
``off`` should *mean* ``false`` is a behaviour change on a safety switch and is
deliberately out of scope, so the recognized-spelling cases below are controls
asserting the verdicts did not move.
"""

from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path

import pytest

from strands_robots import _mesh_switch
from strands_robots._mesh_switch import (
    AFFIRMATIVE,
    MESH_ENV_VAR,
    NEGATIVE,
    mesh_env_request,
)
from strands_robots.mesh import core as mesh_core
from strands_robots.mesh.core import mesh_disabled_by_env
from strands_robots.robot import _mesh_env_opt_in

_OWNER_LOGGER = "strands_robots._mesh_switch"

#: Values that belong to neither half. ``off``/``on`` are spelled by other env
#: vars in this package; ``ture`` is a plain transposition of ``true``.
UNRECOGNIZED = ("off", "on", "ture", "disabled")


@pytest.fixture(autouse=True)
def _forget_reported_values():
    """Clear the once-per-value report ledger around each test.

    The ledger is module state by design (the report must not repeat per call),
    so a test that asserts a warning fires would otherwise depend on whether an
    earlier test already spent that value.
    """
    saved = set(_mesh_switch._UNKNOWN_WARNED)
    _mesh_switch._UNKNOWN_WARNED.clear()
    yield
    _mesh_switch._UNKNOWN_WARNED.clear()
    _mesh_switch._UNKNOWN_WARNED.update(saved)


class TestAnUnrecognizedValueIsReported:
    """The defect: a value belonging to neither half fell through in silence."""

    @pytest.mark.parametrize("raw", UNRECOGNIZED)
    def test_an_unrecognized_value_warns(self, monkeypatch, caplog, raw):
        monkeypatch.setenv(MESH_ENV_VAR, raw)
        with caplog.at_level(logging.WARNING, logger=_OWNER_LOGGER):
            assert mesh_env_request() is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f"{raw!r} produced {len(warnings)} warnings"
        message = warnings[0].getMessage()
        assert raw in message, message

    @pytest.mark.parametrize("raw", UNRECOGNIZED)
    def test_the_report_names_the_spellings_that_work(self, monkeypatch, caplog, raw):
        """A report that does not say what to write instead leaves the operator stuck."""
        monkeypatch.setenv(MESH_ENV_VAR, raw)
        with caplog.at_level(logging.WARNING, logger=_OWNER_LOGGER):
            mesh_env_request()
        message = caplog.records[0].getMessage()
        for spelling in AFFIRMATIVE + NEGATIVE:
            assert spelling in message, f"{spelling!r} missing from: {message}"

    def test_neither_reader_stays_silent_on_a_typo(self, monkeypatch, caplog):
        """Reached through the real resolvers, not the owner directly.

        Either reader consulting the owner is enough to surface the typo, which
        is the property that was missing: both used to answer from their own
        half and neither had the standing to complain.
        """
        monkeypatch.setenv(MESH_ENV_VAR, "off")
        with caplog.at_level(logging.WARNING, logger=_OWNER_LOGGER):
            assert _mesh_env_opt_in() is False
            assert mesh_disabled_by_env() is False
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


class TestTheReportIsOncePerDistinctValue:
    """Every construction site of a Mesh consults the kill switch."""

    def test_repeated_calls_report_once(self, monkeypatch, caplog):
        monkeypatch.setenv(MESH_ENV_VAR, "off")
        with caplog.at_level(logging.WARNING, logger=_OWNER_LOGGER):
            for _ in range(25):
                mesh_env_request()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f"{len(warnings)} lines for one typo"

    def test_a_second_distinct_typo_is_still_news(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger=_OWNER_LOGGER):
            monkeypatch.setenv(MESH_ENV_VAR, "off")
            mesh_env_request()
            monkeypatch.setenv(MESH_ENV_VAR, "ture")
            mesh_env_request()
        reported = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(reported) == 2, reported
        assert any("off" in m for m in reported)
        assert any("ture" in m for m in reported)


class TestRecognizedSpellingsDidNotMove:
    """Controls. The fix reports; it does not widen or narrow the vocabulary."""

    @pytest.mark.parametrize("raw", AFFIRMATIVE)
    def test_an_affirmative_spelling_opts_in_silently(self, monkeypatch, caplog, raw):
        monkeypatch.setenv(MESH_ENV_VAR, raw)
        with caplog.at_level(logging.WARNING, logger=_OWNER_LOGGER):
            assert mesh_env_request() is True
            assert _mesh_env_opt_in() is True
            assert mesh_disabled_by_env() is False
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    @pytest.mark.parametrize("raw", NEGATIVE)
    def test_a_negative_spelling_kills_silently(self, monkeypatch, caplog, raw):
        monkeypatch.setenv(MESH_ENV_VAR, raw)
        with caplog.at_level(logging.WARNING, logger=_OWNER_LOGGER):
            assert mesh_env_request() is False
            assert _mesh_env_opt_in() is False
            assert mesh_disabled_by_env() is True
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_unset_and_empty_say_nothing_without_a_report(self, monkeypatch, caplog, raw):
        """An empty value is not a typo -- it is the default, and must stay quiet."""
        monkeypatch.setenv(MESH_ENV_VAR, raw)
        with caplog.at_level(logging.WARNING, logger=_OWNER_LOGGER):
            assert mesh_env_request() is None
            assert _mesh_env_opt_in() is False
            assert mesh_disabled_by_env() is False
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_an_absent_variable_says_nothing_without_a_report(self, monkeypatch, caplog):
        monkeypatch.delenv(MESH_ENV_VAR, raising=False)
        with caplog.at_level(logging.WARNING, logger=_OWNER_LOGGER):
            assert mesh_env_request() is None
            assert _mesh_env_opt_in() is False
            assert mesh_disabled_by_env() is False
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    @pytest.mark.parametrize("raw", ["FALSE", "  false  ", "  FALSE  ", "TRUE", " 1 "])
    def test_case_and_whitespace_are_still_normalised(self, monkeypatch, caplog, raw):
        monkeypatch.setenv(MESH_ENV_VAR, raw)
        with caplog.at_level(logging.WARNING, logger=_OWNER_LOGGER):
            assert mesh_env_request() is (raw.strip().lower() in AFFIRMATIVE)
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_the_tristate_is_three_outcomes_not_two(self, monkeypatch):
        """``None`` is why one resolver can serve two readers.

        Collapsing "said nothing" into either bool would make one reader wrong:
        an unset variable is neither an opt-in nor a kill.
        """
        seen = set()
        for raw in AFFIRMATIVE + NEGATIVE + UNRECOGNIZED:
            monkeypatch.setenv(MESH_ENV_VAR, raw)
            seen.add(mesh_env_request())
        assert seen == {True, False, None}


class TestTheVocabularyCannotRESplit:
    """A guard, because the silence was caused by the split rather than by a bug.

    Re-adding an inline spelling at either reader restores the exact condition
    that made the typo unreportable, and it would do so without failing any
    behavioural test above.
    """

    @pytest.mark.parametrize("resolver", [_mesh_env_opt_in, mesh_disabled_by_env], ids=lambda f: f.__name__)
    def test_a_reader_does_not_read_the_variable_itself(self, resolver):
        tree = ast.parse(inspect.getsource(resolver).lstrip())
        reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and any(isinstance(arg, ast.Constant) and arg.value == MESH_ENV_VAR for arg in node.args)
        ]
        assert reads == [], (
            f"{resolver.__name__} reads {MESH_ENV_VAR} itself instead of delegating "
            "to _mesh_switch.mesh_env_request; that is the split that made an "
            "unrecognized value unreportable"
        )

    def test_core_keeps_no_second_copy_of_the_negative_half(self):
        """A re-export with no caller is the split waiting to be re-established.

        ``mesh.core`` used to spell the kill-switch values itself. Now that its
        predicate delegates, any surviving module-level copy there is dead code
        that a later edit could quietly start reading again.
        """
        source = Path(mesh_core.__file__).read_text(encoding="utf-8")
        assert "_MESH_KILL_SWITCH_VALUES" not in source

    def test_the_two_halves_do_not_overlap(self):
        assert set(AFFIRMATIVE).isdisjoint(NEGATIVE)

    def test_the_owner_imports_nothing_from_the_package(self):
        """``robot`` reaches the mesh package only lazily; the owner must not undo that.

        A resolver that pulled in ``strands_robots.mesh`` would make the
        Zenoh-backed session and core import eagerly at ``Robot`` import time.
        """
        source = Path(_mesh_switch.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        offenders = {m for m in imported if m.startswith("strands_robots")}
        assert offenders == set(), offenders


class TestTheseTestsAreNotVacuous:
    def test_the_unrecognized_sample_really_is_unrecognized(self):
        vocabulary = set(AFFIRMATIVE) | set(NEGATIVE)
        assert vocabulary.isdisjoint(UNRECOGNIZED)

    def test_off_is_a_spelling_the_package_uses_elsewhere(self):
        """The reason ``off`` is the headline case rather than an invented one."""
        source = Path(Path(_mesh_switch.__file__).parent / "tools" / "robot_mesh.py").read_text(encoding="utf-8")
        assert '"off"' in source

    def test_the_readers_disagree_about_which_half_they_own(self, monkeypatch):
        """The structural fact the guard above protects.

        Each reader answers False for the other half's spellings, which is
        correct and is exactly why neither could report a third value.
        """
        monkeypatch.setenv(MESH_ENV_VAR, "false")
        assert _mesh_env_opt_in() is False
        assert mesh_disabled_by_env() is True

        monkeypatch.setenv(MESH_ENV_VAR, "true")
        assert _mesh_env_opt_in() is True
        assert mesh_disabled_by_env() is False
