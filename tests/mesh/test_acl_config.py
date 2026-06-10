"""Tests for :mod:`strands_robots.mesh._acl_config`.

The ACL semantics validated here against a live Zenoh session live in
``test_zenoh_transport_security.py::TestACLEnforcement``. This file covers only
the static shape of the dict the builder emits and the JSON5-lite
loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.mesh import _acl_config as ac


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)


# --- default ACL --------------------------------------------------------


class TestDefaultACL:
    def test_enabled_is_true(self):
        # Without ``enabled: true`` Zenoh silently no-ops the ACL.
        assert ac.default_acl("strands")["enabled"] is True

    def test_default_permission_is_allow(self):
        # Permissive-by-design: documented in CHANGELOG section 8 and the
        # README "Default ACL -- permissive by design" section. Code now
        # matches docs (earlier mixed default_permission='deny' with two
        # ['**'] allow-rules -- effectively allow-any but confusing).
        assert ac.default_acl("strands")["default_permission"] == "allow"

    def test_default_acl_self_consistent(self):
        """Pin: default_permission matches the rule set's effective behaviour.

        This pins the prior fix for the code-vs-doc drift review feedback flagged
        5x across the prior fix/the prior fix/the prior fix/the prior fix/the prior fix. Mixing default_permission='deny' with
        ['**'] allow-rules looks like deny-by-default but is actually allow-any.
        """
        acl = ac.default_acl("strands")
        if acl["default_permission"] == "allow":
            # Permissive default: no rules needed (rules can only RESTRICT).
            assert acl["rules"] == [], "rules with default=allow can only deny -- inconsistent with permissive promise"
            assert acl["subjects"] == [], "subjects without rules are dead config"
            assert acl["policies"] == []
        else:
            assert acl["default_permission"] == "deny"
            # default=deny without explicit allow-rules == silent total outage.
            assert acl["rules"], "default_permission='deny' with empty rules is silent total deny-all"

    def test_acl_block_serialises_to_json(self):
        path, value = ac.acl_block("strands")
        assert path == "access_control"
        decoded = json.loads(value)
        assert decoded["enabled"] is True
        # Whatever the active default_permission, acl_block must round-trip it.
        assert decoded["default_permission"] in ("allow", "deny")


# --- ACL file loader ----------------------------------------------------


class TestACLFileLoader:
    def _good_acl_dict(self) -> dict:
        return {
            "enabled": True,
            "default_permission": "deny",
            "rules": [],
            "subjects": [{"id": "x", "interfaces": ["lo"], "cert_common_names": ["foo-*"]}],
            "policies": [],
        }

    def test_resolve_uses_default_when_unset(self):
        # When STRANDS_MESH_ACL_FILE is unset, resolve_acl returns default_acl()
        # which is permissive-by-design (default_permission=allow + empty rules).
        acl = ac.resolve_acl("strands")
        assert acl["enabled"] is True
        assert acl["default_permission"] == "allow"
        assert acl["subjects"] == []

    def test_resolve_loads_from_file(self, monkeypatch, tmp_path):
        path = tmp_path / "acl.json"
        path.write_text(json.dumps(self._good_acl_dict()))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))

        loaded = ac.resolve_acl("strands")
        assert loaded["enabled"] is True
        assert loaded["subjects"][0]["cert_common_names"] == ["foo-*"]

    def test_missing_file_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(tmp_path / "nope.json"))
        with pytest.raises(FileNotFoundError):
            ac.resolve_acl("strands")

    def test_oversize_file_rejected(self, monkeypatch, tmp_path):
        path = tmp_path / "huge.json"
        path.write_text("x" * (ac.ACL_FILE_MAX_BYTES + 1))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="refusing to load"):
            ac.resolve_acl("strands")

    def test_invalid_json_rejected(self, monkeypatch, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{this is not json")
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="not valid JSON5"):
            ac.resolve_acl("strands")

    def test_missing_required_field_rejected(self, monkeypatch, tmp_path):
        path = tmp_path / "incomplete.json"
        path.write_text(json.dumps({"enabled": True, "default_permission": "deny", "rules": []}))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="missing required field"):
            ac.resolve_acl("strands")

    def test_missing_enabled_rejected(self, monkeypatch, tmp_path):
        # Missing or false ``enabled`` silently disables the ACL in
        # Zenoh; the loader fails closed.
        no_enabled = self._good_acl_dict()
        del no_enabled["enabled"]
        path = tmp_path / "no_enabled.json"
        path.write_text(json.dumps(no_enabled))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="enabled: true"):
            ac.resolve_acl("strands")

    def test_explicit_enabled_false_rejected(self, monkeypatch, tmp_path):
        bad = self._good_acl_dict()
        bad["enabled"] = False
        path = tmp_path / "disabled.json"
        path.write_text(json.dumps(bad))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="enabled: true"):
            ac.resolve_acl("strands")

    def test_invalid_default_permission_rejected(self, monkeypatch, tmp_path):
        bad = self._good_acl_dict()
        bad["default_permission"] = "maybe"
        path = tmp_path / "weird.json"
        path.write_text(json.dumps(bad))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="must be 'allow' or 'deny'"):
            ac.resolve_acl("strands")

    def test_default_allow_logs_warning(self, monkeypatch, tmp_path, caplog):
        # Per review thread _acl_config.py:199, the warning is scoped to
        # ``allow + non-empty rules`` (the actual blacklist anti-pattern).
        # ``allow + empty rules`` is the documented permissive-default
        # shape and does NOT warn.
        bad = self._good_acl_dict()
        bad["default_permission"] = "allow"
        bad["rules"] = [
            {
                "id": "blacklisted",
                "key_exprs": ["strands/secret/**"],
                "messages": ["put"],
                "flows": ["ingress"],
                "permission": "deny",
            }
        ]
        path = tmp_path / "blacklist.json"
        path.write_text(json.dumps(bad))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        # B-08 / F-14: an allow+rules blacklist ACL now hard-refuses unless
        # the operator acknowledges the posture. Ack so we can still observe
        # the (first-line) warning this test pins.
        monkeypatch.setenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", "1")
        ac._clear_acl_cache_for_test()
        with caplog.at_level("WARNING"):
            ac.resolve_acl("strands")
        assert any("blacklist" in rec.message for rec in caplog.records)

    def test_default_allow_with_rules_refuses_without_ack(self, monkeypatch, tmp_path):
        """B-08 / F-14: allow+rules ACL refuses to load without explicit ack."""
        bad = self._good_acl_dict()
        bad["default_permission"] = "allow"
        bad["rules"] = [
            {
                "id": "blacklisted",
                "key_exprs": ["strands/secret/**"],
                "messages": ["put"],
                "flows": ["ingress"],
                "permission": "deny",
            }
        ]
        path = tmp_path / "blacklist.json"
        path.write_text(json.dumps(bad))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        monkeypatch.delenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", raising=False)
        ac._clear_acl_cache_for_test()
        with pytest.raises(ac.PermissiveACLError):
            ac.resolve_acl("strands")


# --- JSON5 preprocessor tests ------------------------------------------


# --- JSON5 parsing tests ----------------------------------------------


class TestJSON5EndToEnd:
    """End-to-end tests delegating to the json5 PyPI dep (post-the prior fix swap).

    The hand-rolled JSON5-lite preprocessor was replaced with json5.loads;
    these tests verify the shipped operator template parses cleanly and
    that the ACL loader's fail-closed contract holds on malformed input.
    """

    def test_example_file_loads_and_parses(self):
        # Load the canonical template `examples/mesh_acl_example.json5`
        # via the public _load_acl_file path (exercises the full
        # json5.loads -> _validate_acl_shape pipeline).
        example_path = Path(__file__).resolve().parents[2] / "examples" / "mesh_acl_example.json5"
        assert example_path.is_file(), f"Example file not found at {example_path}"
        parsed = ac._load_acl_file(example_path)

        # Verify top-level structure
        assert isinstance(parsed, dict)
        assert parsed["enabled"] is True
        assert parsed["default_permission"] == "deny"

        # Verify rules were parsed (not just comments)
        rule_ids = {r["id"] for r in parsed["rules"]}
        assert "robot_publish_telemetry" in rule_ids
        assert "operator_publish_cmds" in rule_ids
        assert "any_subscribe" in rule_ids

        # Verify subjects parsed
        subject_ids = {s["id"] for s in parsed["subjects"]}
        assert "robot_peer" in subject_ids
        assert "operator_peer" in subject_ids

        # Verify nested arrays with trailing commas parsed correctly
        robot_rule = next(r for r in parsed["rules"] if r["id"] == "robot_publish_telemetry")
        assert "**/presence" in robot_rule["key_exprs"]
        assert "**/response/**" in robot_rule["key_exprs"]

        # Verify inline comments did not break cert_common_names
        robot_subj = next(s for s in parsed["subjects"] if s["id"] == "robot_peer")
        assert "robot-a" in robot_subj["cert_common_names"]
        assert "robot-b" in robot_subj["cert_common_names"]


class TestJSON5MalformedFailsLoudly:
    """Operator-friendly diagnostics on malformed input — the json5 dep
    swap closes the silent-truncation surface the hand-rolled preprocessor
    had on unterminated `/*` blocks.
    """

    def test_unterminated_block_comment_raises(self, tmp_path):
        path = tmp_path / "bad.json5"
        path.write_text("""{
            /* unterminated block comment...
            "enabled": true,
        }""")
        with pytest.raises(ValueError, match=r"is not valid JSON5"):
            ac._load_acl_file(path)

    def test_unterminated_string_raises(self, tmp_path):
        path = tmp_path / "bad.json5"
        path.write_text("""{
            "enabled": "unterminated string,
        }""")
        with pytest.raises(ValueError, match=r"is not valid JSON5"):
            ac._load_acl_file(path)

    def test_object_in_array_with_unquoted_keys_parses(self, tmp_path):
        """The hand-rolled preprocessor missed the `[` lookback context;
        json5 handles this natively. Pin to the canonical operator shape.
        """
        path = tmp_path / "ok.json5"
        path.write_text("""{
            enabled: true,
            default_permission: "deny",
            rules: [
                { id: "r1", key_exprs: ["**"], messages: ["put"], flows: ["ingress"], permission: "allow" },
            ],
            subjects: [
                { id: "s1", cert_common_names: ["op-1"] },
            ],
            policies: [
                { rules: ["r1"], subjects: ["s1"] },
            ],
        }""")
        parsed = ac._load_acl_file(path)
        assert parsed["rules"][0]["id"] == "r1"

    def test_single_quoted_strings_parse(self, tmp_path):
        """JSON5 single-quoted strings work natively via json5.loads."""
        path = tmp_path / "ok.json5"
        path.write_text("""{
            \'enabled\': true,
            \'default_permission\': \'deny\',
            rules: [
                { id: \'r1\', key_exprs: [\'**\'], messages: [\'put\'], flows: [\'ingress\'], permission: \'allow\' },
            ],
            subjects: [{ id: \'s1\', cert_common_names: [\'op-1\'] }],
            policies: [{ rules: [\'r1\'], subjects: [\'s1\'] }],
        }""")
        parsed = ac._load_acl_file(path)
        assert parsed["enabled"] is True
        assert parsed["subjects"][0]["cert_common_names"] == ["op-1"]


class TestDefaultACLPermissiveShape:
    """Pin the post-the prior permissive-allow shape of the default ACL.

    Operators not supplying STRANDS_MESH_ACL_FILE get a permissive default
    by design: any peer that survived the mTLS handshake can publish and
    subscribe on any key (CHANGELOG section 8, README "Default ACL --
    permissive by design").

    The the prior fix fix delivers this with default_permission='allow' + empty rules
    (the simplest config matching the documented behaviour). Earlier mixes
    of default_permission='deny' + ['**'] allow-rules were flagged 5x in
    review for code-vs-doc drift; this class pins the correction.
    """

    def test_enabled_is_true(self):
        # Without ``enabled: true`` Zenoh silently no-ops the entire ACL.
        acl = ac.default_acl("strands")
        assert acl["enabled"] is True

    def test_default_permission_is_allow(self):
        acl = ac.default_acl("strands")
        assert acl["default_permission"] == "allow"

    def test_rule_set_is_empty(self):
        # default_permission='allow' + empty rules == permissive. Adding
        # ['**'] rules with this default is dead config.
        acl = ac.default_acl("strands")
        assert acl["rules"] == []

    def test_subject_set_is_empty(self):
        # Subjects only matter when rules constrain access. With an empty
        # rule set + permissive default, subjects are dead config.
        acl = ac.default_acl("strands")
        assert acl["subjects"] == []

    def test_policy_set_is_empty(self):
        acl = ac.default_acl("strands")
        assert acl["policies"] == []

    def test_load_acl_file_round_trip_with_example(self, tmp_path):
        # Another review concern: "the loader never being tested against the
        # shipped example." Load ``examples/mesh_acl_example.json5`` and
        # verify ``_load_acl_file`` parses it without raising.
        example_src = Path(__file__).resolve().parents[2] / "examples" / "mesh_acl_example.json5"
        assert example_src.is_file(), f"Example file not found at {example_src}"

        # Copy to tmp_path so we can use _load_acl_file (which requires a Path)
        tmp_example = tmp_path / "mesh_acl_example.json5"
        tmp_example.write_text(example_src.read_text(encoding="utf-8"), encoding="utf-8")

        # Should not raise
        loaded = ac._load_acl_file(tmp_example)
        assert loaded["enabled"] is True
        assert loaded["default_permission"] == "deny"
        assert len(loaded["rules"]) > 0
        assert len(loaded["subjects"]) > 0
        assert len(loaded["policies"]) > 0


class TestIsDefaultACLInUse:
    """operators forgetting STRANDS_MESH_ACL_FILE should
    get a runtime signal. is_default_acl_in_use() is the predicate the
    session-open WARNING calls."""

    def test_unset_returns_true(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)
        from strands_robots.mesh import _acl_config as ac

        assert ac.is_default_acl_in_use() is True

    def test_empty_returns_true(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", "")
        from strands_robots.mesh import _acl_config as ac

        assert ac.is_default_acl_in_use() is True

    def test_whitespace_only_returns_true(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", "   ")
        from strands_robots.mesh import _acl_config as ac

        assert ac.is_default_acl_in_use() is True

    def test_set_to_strict_path_returns_false(self, monkeypatch, tmp_path):
        """env-var pointing to a STRICT (non-permissive-shape)
        ACL file returns False -- the gate only fires on the
        permissive shape, regardless of source.
        """
        strict_file = tmp_path / "strict.json5"
        strict_file.write_text(
            '{"enabled": true, "default_permission": "deny", '
            '"rules": [{"id": "r", "permission": "allow", '
            '"flows": ["egress"], "messages": ["put"], '
            '"key_exprs": ["strands/op/cmd"]}], '
            '"subjects": [{"id": "r", "cert_common_names": ["op"]}], '
            '"policies": [{"rules": ["r"], "subjects": ["r"]}]}'
        )
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(strict_file))
        from strands_robots.mesh import _acl_config as ac

        assert ac.is_default_acl_in_use() is False

    def test_set_to_unloadable_path_fails_closed(self, monkeypatch):
        """env-var pointing to a NONEXISTENT path fails closed
        (returns True) so a typo does not silently lift the gate."""
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", "/etc/mesh-acl.json5")
        from strands_robots.mesh import _acl_config as ac

        # the prior implementation returned False; post-the prior fix the unloadable file fails
        # closed and triggers the gate.
        assert ac.is_default_acl_in_use() is True


class TestACLFileSymlinkAndTOCTOU:
    """Pin regressions for review feedback: ACL load must defeat symlink
    swap and TOCTOU on the size check.

    Mirrors the audit-log discipline (O_NOFOLLOW + bounded read).
    """

    def test_acl_load_refuses_symlink(self, tmp_path):
        """A symlink at STRANDS_MESH_ACL_FILE must be rejected, not followed."""
        import os as _os

        from strands_robots.mesh._acl_config import _load_acl_file

        target = tmp_path / "real.json5"
        target.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "default_permission": "deny",
                    "rules": [],
                    "subjects": [],
                    "policies": [],
                }
            )
        )
        link = tmp_path / "link.json5"
        _os.symlink(str(target), str(link))

        try:
            _load_acl_file(link)
        except ValueError as exc:
            assert "SYMLINK" in str(exc) or "symlink" in str(exc).lower(), exc
        else:
            raise AssertionError(
                "loader followed symlink instead of refusing -- ACL file gate must "
                "refuse symlinks the way audit.py does (O_NOFOLLOW + lstat reject)"
            )

    def test_acl_load_rejects_oversized_file(self, tmp_path):
        """Reading must be bounded at ACL_FILE_MAX_BYTES + 1 so an attacker
        who races content between stat() and read() cannot bypass the cap.
        """
        from strands_robots.mesh._acl_config import ACL_FILE_MAX_BYTES, _load_acl_file

        big = tmp_path / "big.json5"
        # Write more than the cap.
        big.write_text("x" * (ACL_FILE_MAX_BYTES + 1024))

        try:
            _load_acl_file(big)
        except ValueError as exc:
            assert "bytes" in str(exc) or "refusing" in str(exc).lower(), exc
        else:
            raise AssertionError("loader accepted oversized ACL file -- size cap not enforced")


class TestJSON5DepSwap:
    """The hand-rolled preprocessor was replaced with json5.loads. These
    pins ensure the new parser surfaces operator-friendly diagnostics on
    malformed input rather than silently truncating (the old behaviour).
    """

    def test_unterminated_block_comment_raises_clear_error(self, tmp_path):
        path = tmp_path / "acl.json5"
        path.write_text("{\n  /* unterminated block ...\n  enabled: true,\n}\n")
        with pytest.raises(ValueError, match=r"is not valid JSON5"):
            ac._load_acl_file(path)

    def test_no_legacy_preprocessor_symbols_exist(self):
        """The four hand-rolled preprocessor functions must be gone --
        catches a future revert that re-introduces the fragile parser.
        """
        for name in (
            "_strip_json5_comments",
            "_strip_trailing_commas",
            "_quote_unquoted_keys",
            "_convert_single_quoted_strings",
            "_json5_to_json",
        ):
            assert not hasattr(ac, name), f"{name} should have been removed in F3-A"

    def test_json5_pep_dependency_lazy_import(self):
        """json5 is imported lazily inside ``_parse_json5``.
        the prior implementation the import was at module top, which forced the dep
        on dev users running auth_mode=none who don't load an ACL
        file. Lazy-import means the dep is only paid when an ACL
        file actually needs to be parsed."""
        # Module no longer carries a top-level json5 attribute
        assert not hasattr(ac, "json5"), (
            "F15-E regression: json5 should be lazy-imported inside _parse_json5, not at module top-level"
        )
        # And calling _parse_json5 still works (proves the lazy import path)
        result = ac._parse_json5('{"foo": "bar"}', Path("/dev/null"))
        assert result == {"foo": "bar"}


# ---------------------------------------------------------------------
# the prior fix-1: bare except on permissive-ACL warning narrowed
# ---------------------------------------------------------------------


# === default_acl() shape regression test ===


class TestF16DefaultACLShapeIsLoadBearing:
    """the ``default_acl()`` shape is now load-
    bearing for the prior start-time gate -- if a future refactor
    accidentally bypasses the ``mtls + permissive default ACL``
    refusal, this regression test catches a default that becomes
    accidentally less permissive (which would silently break the
    first-run UX) OR more permissive in a non-obvious way (e.g. allow
    + non-empty rules) which would pass the prior is_truly_permissive_default
    check while shipping a different posture.
    """

    def test_default_acl_is_truly_permissive_shape(self):
        from strands_robots.mesh._acl_config import default_acl

        d = default_acl("strands")
        # Required fields present
        assert d["enabled"] is True
        assert d["default_permission"] == "allow"
        # Empty rules/subjects/policies -- the documented permissive-by-
        # design shape that the prior implementation's loader logic special-cases.
        assert d["rules"] == []
        assert d["subjects"] == []
        assert d["policies"] == []

    def test_is_default_acl_in_use_picks_up_default(self, monkeypatch):
        """When STRANDS_MESH_ACL_FILE is unset, is_default_acl_in_use()
        returns True so Mesh.start() can fire the prior gate."""
        from strands_robots.mesh._acl_config import is_default_acl_in_use

        monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)
        assert is_default_acl_in_use() is True

    def test_is_default_acl_in_use_with_strict_file_returns_false(self, monkeypatch, tmp_path):
        """shape-based check. A strict file returns False;
        previous semantics (env-var-presence -> False) replaced.
        """
        from strands_robots.mesh._acl_config import is_default_acl_in_use

        strict = tmp_path / "strict.json5"
        strict.write_text(
            '{"enabled": true, "default_permission": "deny", '
            '"rules": [{"id": "r", "permission": "allow", '
            '"flows": ["egress"], "messages": ["put"], '
            '"key_exprs": ["strands/op/cmd"]}], '
            '"subjects": [{"id": "r", "cert_common_names": ["op"]}], '
            '"policies": [{"rules": ["r"], "subjects": ["r"]}]}'
        )
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(strict))
        assert is_default_acl_in_use() is False
