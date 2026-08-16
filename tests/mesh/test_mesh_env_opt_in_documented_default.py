"""A configuration table must not print a ``STRANDS_MESH`` default that opts in.

Mesh is opt-in: ``strands_robots.robot._mesh_env_opt_in`` turns it on only for
``true``/``1``/``yes``, so a bare ``Robot()`` is quiet and never spins up
Zenoh, ACL or e-stop machinery. ``tests/mesh/test_mesh_wiring.py`` pins that
behaviour.

A table that prints an opt-in spelling in its Default column therefore tells a
reader mesh is already running, and hides the one spelling that actually
enables it - the reader's only documented knob ("set it to ``false``") is a
no-op for the state they are actually in.

These tests grade the shipped tables against the resolver itself rather than
against a hand-copied list of spellings, so widening the accepted spellings
later cannot silently invalidate the guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from strands_robots.robot import _mesh_env_opt_in

_ENV = "STRANDS_MESH"
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The guard is only meaningful while it still reaches the shipped tables. If a
# rename or a reformat drops them all, fail loudly instead of reporting clean.
_MINIMUM_DOCUMENTED_ROWS = 3


def _documented_rows() -> list[tuple[str, int, str, str]]:
    """Return every markdown table row whose first cell is exactly ``STRANDS_MESH``.

    Returns:
        A list of ``(relative_path, line_number, description_cell, default_cell)``
        tuples. Rows that bundle several variables into one cell are skipped:
        this guard grades the single-variable rows a reader copies a value from.
    """
    docs = [_REPO_ROOT / "README.md", *sorted((_REPO_ROOT / "docs").rglob("*.md"))]
    rows: list[tuple[str, int, str, str]] = []
    for doc in docs:
        if not doc.exists():
            continue
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith("|") or not stripped.endswith("|"):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if len(cells) < 3 or cells[0] != f"`{_ENV}`":
                continue
            rows.append((str(doc.relative_to(_REPO_ROOT)), lineno, cells[1], cells[-1]))
    return rows


class TestTheResolverOnlyEverOptsIn:
    """The oracle the documented defaults are graded against."""

    @pytest.mark.parametrize("raw", ["true", "TRUE", "True", "1", "yes", " yes "])
    def test_an_opt_in_spelling_turns_mesh_on(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_ENV, raw)
        assert _mesh_env_opt_in() is True

    @pytest.mark.parametrize("raw", ["", " ", "false", "0", "no", "off", "on", "enabled", "maybe"])
    def test_every_other_value_leaves_mesh_off(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_ENV, raw)
        assert _mesh_env_opt_in() is False

    def test_an_unset_env_leaves_mesh_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_ENV, raising=False)
        assert _mesh_env_opt_in() is False


class TestTheDocumentedDefaultLeavesMeshOff:
    """Grade every shipped configuration table against the resolver."""

    def test_the_guard_still_reaches_the_shipped_tables(self) -> None:
        rows = _documented_rows()
        assert len(rows) >= _MINIMUM_DOCUMENTED_ROWS, (
            f"expected at least {_MINIMUM_DOCUMENTED_ROWS} configuration rows keyed to "
            f"{_ENV}, found {len(rows)}: {rows}. The extractor no longer reaches the "
            "shipped tables, so a clean result below would prove nothing."
        )

    def test_no_table_prints_an_opt_in_spelling_as_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        offenders: list[str] = []
        for path, lineno, _description, default in _documented_rows():
            for token in re.findall(r"[A-Za-z0-9]+", default):
                monkeypatch.setenv(_ENV, token)
                if _mesh_env_opt_in():
                    offenders.append(
                        f"{path}:{lineno} prints {token!r} as the default for {_ENV}, but "
                        f"{_ENV}={token} opts IN. A bare Robot() has mesh OFF, so this row "
                        "tells a reader mesh is already running and hides the opt-in."
                    )
        assert not offenders, "documented default contradicts the resolver:\n" + "\n".join(offenders)

    def test_every_table_still_names_a_spelling_that_opts_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Correcting the default must not remove the only way to turn mesh on."""
        silent: list[str] = []
        for path, lineno, description, default in _documented_rows():
            opts_in = False
            for token in re.findall(r"`([^`]+)`", f"{description} {default}"):
                monkeypatch.setenv(_ENV, token)
                opts_in = opts_in or _mesh_env_opt_in()
            if not opts_in:
                silent.append(
                    f"{path}:{lineno} documents {_ENV} without naming any value that "
                    "turns mesh on, so the opt-in is undiscoverable from this row."
                )
        assert not silent, "\n".join(silent)
