"""Tests for the CA pin rotation grace-period contract (issue #250).

Issue #250 (Option B) ships a documented rotation runbook for the Amazon
Root CA1 pin instead of a signed-manifest fetch. The operational guarantee
the runbook depends on is that the accepted-pin set is a collection, so a
rotation can keep both the old and the new pin valid during fleet uptake:

* The built-in pin tuple plus a staged second pin from
  ``STRANDS_MESH_CA_PINS`` both resolve as accepted (dual-pin overlap).
* Verification accepts a certificate matching either pin in the set.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import pathlib
import re

import pytest

from strands_robots.mesh.iot import provision


def test_resolve_ca_pins_accepts_both_builtin_and_staged_pin_for_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """During a rotation, the built-in pin and a staged STRANDS_MESH_CA_PINS pin coexist."""
    builtin = set(provision._AMAZON_ROOT_CA1_PINS)
    assert builtin, "expected at least one built-in pin"
    staged = "a" * 64  # stand-in for the next root CA1 pin staged out-of-band
    assert staged not in builtin

    monkeypatch.setenv("STRANDS_MESH_CA_PINS", staged)
    resolved = provision._resolve_ca_pins()

    assert builtin.issubset(resolved), "old pin must stay valid during overlap"
    assert staged in resolved, "newly staged pin must also be accepted"
    assert len(resolved) >= len(builtin) + 1


def test_resolve_ca_pins_returns_collection_not_scalar() -> None:
    """The accepted-pin surface is a set so the dual-pin grace period is expressible."""
    resolved = provision._resolve_ca_pins()
    assert isinstance(resolved, frozenset)


def test_verify_ca_bytes_accepts_either_pin_during_dual_pin_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cert matching the staged pin verifies even when it is not the built-in pin."""
    rogue = b"-----BEGIN CERTIFICATE-----\nstaged-root-bytes\n-----END CERTIFICATE-----\n"
    staged = hashlib.sha256(rogue).hexdigest()
    assert staged not in set(provision._AMAZON_ROOT_CA1_PINS)

    # Without the staged pin, the rogue cert is rejected.
    assert provision._verify_ca_bytes(rogue) is False

    # Once staged out-of-band, the same cert is accepted (grace-period overlap).
    monkeypatch.setenv("STRANDS_MESH_CA_PINS", staged)
    assert provision._verify_ca_bytes(rogue) is True


# --- The runbook itself (issue #250 acceptance criterion 1) --------------------
#
# The two artifacts above are half of what #250 shipped: the AGENTS.md learnings
# entry and this file's dual-pin assertions. Both of them cite a *third* - a
# rotation runbook published for operators - and a citation is not a document.
# These tests grade the runbook against the code it describes, so a procedure
# that exists only in a contributor file cannot pass for a published one.

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_README = _REPO_ROOT / "README.md"
_AGENTS = _REPO_ROOT / "AGENTS.md"

# AGENTS.md > "Operational Runbooks for Security Pins" names where the runbook
# lives. Reading the heading from the citation rather than restating it is what
# makes these tests grade the citation: a heading renamed on one side and not the
# other fails here instead of quietly becoming a dead pointer.
_CITATION_RE = re.compile(r'README\.md > "([^"]+)"')


def _cited_runbook_heading() -> str:
    """Return the README heading AGENTS.md cites as the pin rotation runbook."""
    found = _CITATION_RE.findall(_AGENTS.read_text(encoding="utf-8"))
    assert found, "premise: AGENTS.md no longer cites a README heading for the pin runbook"
    return found[0]


def _readme_section(heading: str) -> str:
    """Return the body of the README section titled *heading*, or ``''``."""
    lines = _README.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if re.fullmatch(rf"#{{1,4}} {re.escape(heading)}", line.strip()):
            depth = len(line) - len(line.lstrip("#"))
            body: list[str] = []
            for nxt in lines[i + 1 :]:
                if nxt.startswith("#") and (len(nxt) - len(nxt.lstrip("#"))) <= depth:
                    break
                body.append(nxt)
            return "\n".join(body)
    return ""


def _ca_pin_env_vars() -> set[str]:
    """Env vars the provisioner reads that decide which CA pins are accepted.

    Derived from ``provision.py``'s own ``os.getenv`` calls, so a third knob
    added to the pin path is graded against the runbook without editing a list.
    """
    tree = ast.parse(pathlib.Path(provision.__file__).read_text(encoding="utf-8"))
    names = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in {"os.getenv", "os.environ.get"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and "CA_PIN" in node.args[0].value
    }
    assert len(names) >= 2, f"premise: expected the add-a-pin and the disable knob, found {names}"
    return names


def _refuse_on_disk(tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture) -> tuple[str, str]:
    """Refuse an on-disk CA whose bytes match no accepted pin; return (error, logs)."""
    ca = tmp_path / "AmazonRootCA1.pem"
    ca.write_bytes(b"-----BEGIN CERTIFICATE-----\nrotated-root\n-----END CERTIFICATE-----\n")
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError) as excinfo:
            provision._ensure_ca(ca)
    return str(excinfo.value), "\n".join(r.getMessage() for r in caplog.records)


def _refuse_download(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Refuse a downloaded CA matching no accepted pin; return (error, its digest)."""
    body = b"-----BEGIN CERTIFICATE-----\nnext-root\n-----END CERTIFICATE-----\n"
    monkeypatch.setattr(provision, "_download_with_per_socket_timeout", lambda url, timeout_s, max_bytes: body)
    with pytest.raises(RuntimeError) as excinfo:
        provision._ensure_ca(tmp_path / "AmazonRootCA1.pem")
    return str(excinfo.value), hashlib.sha256(body).hexdigest()


class TestTheCitedRunbookIsPublished:
    """The runbook the pin's other two artifacts depend on exists and is reachable."""

    def test_the_readme_publishes_the_heading_agents_cites(self) -> None:
        """AGENTS.md names a README heading; README has to carry it."""
        heading = _cited_runbook_heading()
        assert _readme_section(heading), (
            f"AGENTS.md cites README.md > {heading!r} as where the pin rotation runbook "
            "lives, and README.md has no such heading - the rotation procedure is "
            "published nowhere an operator reads"
        )

    def test_the_runbook_states_the_ordered_grace_period(self) -> None:
        """The four steps are what make a rotation shippable without a flag-day deploy."""
        body = _readme_section(_cited_runbook_heading()).lower()
        for phrase, why in (
            ("out of band", "verifying the new certificate independently is step 1"),
            ("keeps the old one", "both pins stay valid during the overlap"),
            ("uptake", "the overlap is bounded by fleet uptake, not release cadence"),
            ("follow-up release", "the old pin is dropped afterwards, not at cutover"),
        ):
            assert phrase in body, f"runbook does not state: {why} (missing {phrase!r})"

    def test_the_runbook_names_the_recompute_command_and_every_pin_env_var(self) -> None:
        """A runbook that omits a knob deciding the accepted set is not a procedure."""
        body = _readme_section(_cited_runbook_heading())
        assert "hashlib.sha256" in body, "runbook omits the recompute one-liner"
        assert provision._AMAZON_ROOT_CA1_URL in body, "runbook omits the CA URL it pins"
        for var in sorted(_ca_pin_env_vars()):
            assert var in body, f"{var} decides which CA pins are accepted and the runbook never mentions it"

    def test_every_pin_refusal_names_the_runbook(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """On-call reads the failure, not AGENTS.md, so the failure names the procedure."""
        heading = _cited_runbook_heading()
        on_disk_error, on_disk_logs = _refuse_on_disk(tmp_path, caplog)
        download_error, _ = _refuse_download(tmp_path / "fresh", monkeypatch)
        for label, text in (
            ("the on-disk re-use refusal", on_disk_error),
            ("the on-disk re-use warning", on_disk_logs),
            ("the download refusal", download_error),
        ):
            assert heading in text, f"{label} names no rotation procedure: {text}"

    def test_the_refusals_and_the_citation_name_one_place(self) -> None:
        """One owner for the pointer, so three refusals cannot drift to three places."""
        from strands_robots.mesh.iot.provision import _CA_ROTATION_RUNBOOK

        heading = _cited_runbook_heading()
        assert heading in _CA_ROTATION_RUNBOOK, (
            f"the refusals point at {_CA_ROTATION_RUNBOOK!r}, which is not the heading AGENTS.md cites ({heading!r})"
        )


class TestTheRunbookDoesNotWeakenTheRefusal:
    """Pointing at a procedure must not become advice to accept the rejected bytes."""

    def test_a_rejected_download_digest_is_not_offered_as_something_to_stage(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mismatch path cannot know a rotation from a MITM, so it advises neither."""
        error, digest = _refuse_download(tmp_path, monkeypatch)
        assert digest in error, "the refusal should still report the digest it got, for diagnosis"
        assert "STRANDS_MESH_CA_PINS" not in error, (
            "naming the add-a-pin knob in a message that just called these bytes rogue "
            f"reads as 'stage what you got': {error}"
        )

    def test_the_runbook_keeps_the_break_glass_out_of_the_rotation_steps(self) -> None:
        """Disabling the check is a different action from widening the accepted set."""
        body = _readme_section(_cited_runbook_heading())
        assert "STRANDS_MESH_DISABLE_CA_PIN" in body
        assert "not** part of this procedure" in body or "not part of this procedure" in body, (
            "the runbook must say the break-glass is not the response to a rotation"
        )

    def test_the_existing_refusal_wording_is_preserved(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Operators and forks match on these phrases; the pointer is appended, not swapped."""
        on_disk_error, on_disk_logs = _refuse_on_disk(tmp_path, caplog)
        download_error, _ = _refuse_download(tmp_path / "fresh2", monkeypatch)
        assert "failed pin check" in on_disk_error
        assert "accepted pins:" in on_disk_error
        assert "does NOT match pinned SHA-256" in on_disk_logs
        assert "refusing to write rogue CA" in download_error

    def test_a_staged_pin_still_admits_the_certificate_it_covers(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The emergency step the runbook documents has to actually work."""
        body = b"-----BEGIN CERTIFICATE-----\nnext-root\n-----END CERTIFICATE-----\n"
        monkeypatch.setattr(provision, "_download_with_per_socket_timeout", lambda url, timeout_s, max_bytes: body)
        monkeypatch.setenv("STRANDS_MESH_CA_PINS", hashlib.sha256(body).hexdigest())
        ca = tmp_path / "AmazonRootCA1.pem"
        provision._ensure_ca(ca)
        assert ca.read_bytes() == body
        assert set(provision._AMAZON_ROOT_CA1_PINS) <= provision._resolve_ca_pins()
