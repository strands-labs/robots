"""The wire-payload diagnostic tables must describe the diagnostic that ships.

``STRANDS_GROOT_WIRE_LOG`` is read as the directory the wire-payload dumps land
in: :func:`strands_robots.policies.groot.policy._wire_log_dir` returns the raw
value and :meth:`Gr00tPolicy._maybe_dump_wire_payload` passes it straight to
``os.makedirs``. Its two nearest neighbours in the same diagnostic family,
``STRANDS_LIBERO_ACTION_LOG`` and ``STRANDS_LIBERO_STATE_LOG``, are plain
on/off flags, so the family reads as if one convention covered all three.

A table that offers a bare flag token as the value therefore does not merely
mis-describe the knob: following it dumps pickle archives into a directory of
that name under whatever the process CWD happens to be, and the reader is never
told the value is theirs to choose.

These tests grade the shipped tables through the resolvers themselves, so
widening what a resolver accepts later cannot silently invalidate the guard.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from strands_robots.mesh.audit import audit_log_path
from strands_robots.policies.groot.policy import Gr00tPolicy, _wire_log_dir

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Environment variables whose value IS a filesystem destination, mapped to the
# shipped resolver that turns the raw value into that destination. Only
# side-effect-free resolvers belong here: grading must never create a directory
# on the machine running the suite (which is why ``STRANDS_ASSETS_DIR`` is
# absent - ``get_assets_dir`` mkdirs what it resolves).
_PATH_VALUED: dict[str, Callable[[], object]] = {
    "STRANDS_GROOT_WIRE_LOG": _wire_log_dir,
    "STRANDS_MESH_AUDIT_DIR": audit_log_path,
}

# The guard is only meaningful while it still reaches the shipped tables. If a
# rename or a reformat drops them all, fail loudly instead of reporting clean.
_MINIMUM_DOCUMENTED_ROWS = 2

_BACKTICKED = re.compile(r"`([^`]+)`")


def _documented_rows() -> list[tuple[str, str, int, str]]:
    """Return every markdown table row that documents a path-valued variable.

    A variable is matched when its backticked name appears in the row's first
    cell, so a row that bundles a companion variable into the same cell (the
    README documents this variable alongside its ``_MAX_CALLS`` companion) is
    graded rather than skipped.

    Returns:
        A list of ``(env_var, relative_path, line_number, description_cell)``
        tuples.
    """
    docs = [_REPO_ROOT / "README.md", *sorted((_REPO_ROOT / "docs").rglob("*.md"))]
    rows: list[tuple[str, str, int, str]] = []
    for doc in docs:
        if not doc.exists():
            continue
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith("|") or not stripped.endswith("|"):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if len(cells) < 3:
                continue
            for env in _PATH_VALUED:
                if f"`{env}`" in cells[0]:
                    rows.append((env, str(doc.relative_to(_REPO_ROOT)), lineno, cells[1]))
    return rows


def _advertised_values(description: str) -> list[str]:
    """Return the backticked tokens a description cell offers as a value to export.

    Two kinds of backticked token in a description are not values for the
    variable and are skipped:

    * an ALL-CAPS token names a companion environment variable
      (``STRANDS_GROOT_WIRE_LOG_MAX_CALLS``);
    * a token with an extension and no separator names an artifact written
      inside the directory (``mesh_audit.jsonl``), not the directory itself.

    Everything else is something a reader would plausibly export.
    """
    values = []
    for tok in _BACKTICKED.findall(description):
        # An all-uppercase token must contain a letter to be a variable name:
        # ``"1".upper() == "1"`` would otherwise skip the numeric literals
        # this guard exists to catch.
        if any(char.isalpha() for char in tok) and tok == tok.upper():
            continue
        if "/" not in tok and "." in tok:
            continue
        values.append(tok)
    return values


def _is_a_destination(value: str) -> bool:
    """Return whether a documented value names a place the reader chose.

    An absolute path or a home-relative path does; a bare token does not, since
    it lands under whatever the process CWD happens to be.
    """
    return Path(value).is_absolute() or value.startswith("~")


class TestTheResolverTakesTheValueAsTheDestination:
    """The oracle the documented values are graded against."""

    @pytest.mark.parametrize("env", sorted(_PATH_VALUED))
    def test_the_raw_value_is_used_as_the_path(self, env: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whatever is exported reaches the filesystem verbatim.

        This is what makes a bare flag token harmful rather than merely wrong:
        there is no flag branch to fall into, so ``1`` names a directory.
        """
        monkeypatch.setenv(env, "/tmp/strands-robots-doc-guard")
        assert "/tmp/strands-robots-doc-guard" in str(_PATH_VALUED[env]())

    def test_a_flag_token_resolves_to_a_relative_directory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``STRANDS_GROOT_WIRE_LOG=1`` is a directory named ``1``, not an on-switch."""
        monkeypatch.setenv("STRANDS_GROOT_WIRE_LOG", "1")
        resolved = _wire_log_dir()
        assert resolved == "1"
        assert not Path(str(resolved)).is_absolute()


class TestEveryDocumentedValueIsADestination:
    """The tables must not offer a value that is not a place."""

    def test_the_guard_reaches_the_shipped_tables(self) -> None:
        """A clean sweep over zero rows would prove nothing."""
        rows = _documented_rows()
        assert len(rows) >= _MINIMUM_DOCUMENTED_ROWS, (
            f"expected at least {_MINIMUM_DOCUMENTED_ROWS} documented rows for "
            f"{sorted(_PATH_VALUED)}, found {len(rows)}"
        )

    def test_no_row_offers_a_value_that_is_not_a_path(self) -> None:
        """Every value a row advertises must name a directory.

        A relative token would send the dumps to a directory of that name under
        whatever the process CWD happens to be, which is never what a reader
        copying a configuration table is asking for.
        """
        offenders = [
            f"{path}:{lineno}: {env} is documented with value {value!r}, which "
            f"{_PATH_VALUED[env].__name__}() resolves to a CWD-relative path; "
            f"document the directory the dumps should land in instead"
            for env, path, lineno, description in _documented_rows()
            for value in _advertised_values(description)
            if not _is_a_destination(value)
        ]
        assert not offenders, "\n".join(offenders)

    def test_the_wire_log_row_still_shows_a_value_a_reader_can_export(self) -> None:
        """Correcting the convention must not remove the worked example.

        Deleting the value rather than fixing it would clear the check above
        while leaving the reader with nothing to copy.
        """
        rows = [row for row in _documented_rows() if row[0] == "STRANDS_GROOT_WIRE_LOG"]
        assert rows, "STRANDS_GROOT_WIRE_LOG is no longer documented anywhere"
        for env, path, lineno, description in rows:
            assert _advertised_values(description), (
                f"{path}:{lineno}: the {env} row no longer shows a directory a reader can export"
            )


class TestTheDiagnosticCoversTheInProcessPath:
    """The dumper is not a wire tap on the service transport.

    :meth:`Gr00tPolicy._maybe_dump_wire_payload` is called from the in-process
    inference path as well as the service path, and the resulting archive is a
    pickle of the observation and the action chunk rather than the frames a
    socket carried. Describing the diagnostic in terms of the transport would
    hide the half a LOCAL-versus-SERVICE comparison needs, since in-process
    inference opens no socket at all.
    """

    def _policy(self) -> Gr00tPolicy:
        """Build a policy without contacting a server or loading a checkpoint."""
        policy = Gr00tPolicy.__new__(Gr00tPolicy)
        policy._groot_version = "n1.7"
        policy.data_config_name = "libero_panda"
        return policy

    def test_the_in_process_path_writes_its_own_archive(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A ``local`` dump lands beside the ``service`` dumps it is compared to."""
        import pickle

        monkeypatch.setenv("STRANDS_GROOT_WIRE_LOG", str(tmp_path))
        policy = self._policy()

        policy._maybe_dump_wire_payload(
            "local",
            {"video": {"image": "frame"}, "state": {"joints": [0.0]}},
            {"action.x": [0.1]},
        )

        written = sorted(p.name for p in tmp_path.iterdir())
        assert written == ["local_call0000.pkl"]
        payload = pickle.loads((tmp_path / written[0]).read_bytes())
        assert payload["mode"] == "local"
        assert payload["observation"] == {"video": {"image": "frame"}, "state": {"joints": [0.0]}}
        assert payload["action_chunk"] == {"action.x": [0.1]}

    def test_both_paths_dump_into_one_directory_under_distinct_names(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The two archives coexist, which is what makes them diffable."""
        monkeypatch.setenv("STRANDS_GROOT_WIRE_LOG", str(tmp_path))
        policy = self._policy()

        policy._maybe_dump_wire_payload("local", {"state": {"joints": [0.0]}}, {"action.x": [0.1]})
        policy._maybe_dump_wire_payload("service", {"state.joints": [0.0]}, {"action.x": [0.1]})

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "local_call0000.pkl",
            "service_call0001.pkl",
        ]
