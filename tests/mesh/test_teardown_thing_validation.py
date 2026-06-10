"""Regression test: teardown_thing validates thing_name before any AWS/FS call.

Addresses review thread on provision.py:320 (PR #228 R3) -- _validate_thing_name
was applied to provision_robot and provision_operator but NOT to teardown_thing,
leaving a path-traversal vector via ``DEFAULT_CERT_DIR / f"{thing_name}.pem"``.
"""

import pytest


class TestTeardownThingValidation:
    """teardown_thing must reject unsafe thing_name values."""

    def test_path_traversal_rejected(self):
        """thing_name containing '../' must raise ValueError before any I/O."""
        from strands_robots.mesh.iot.provision import teardown_thing

        with pytest.raises(ValueError, match="invalid characters"):
            teardown_thing("../../etc/passwd")

    def test_dots_rejected(self):
        """thing_name containing '.' must raise ValueError."""
        from strands_robots.mesh.iot.provision import teardown_thing

        with pytest.raises(ValueError, match="invalid characters"):
            teardown_thing("robot.v2")

    def test_colons_rejected(self):
        """thing_name containing ':' must raise ValueError."""
        from strands_robots.mesh.iot.provision import teardown_thing

        with pytest.raises(ValueError, match="invalid characters"):
            teardown_thing("robot:alpha")

    def test_empty_rejected(self):
        """Empty thing_name must raise ValueError."""
        from strands_robots.mesh.iot.provision import teardown_thing

        with pytest.raises(ValueError, match="non-empty string"):
            teardown_thing("")

    def test_valid_name_passes_validation(self, monkeypatch):
        """Valid thing_name passes validation, reaches boto3 import."""
        from strands_robots.mesh.iot import provision

        # Mock _require_boto3 to avoid real AWS calls
        mock_called = []

        def fake_require_boto3():
            mock_called.append(True)
            raise ImportError("boto3 not available in test")

        monkeypatch.setattr(provision, "_require_boto3", fake_require_boto3)

        with pytest.raises(ImportError, match="boto3 not available"):
            provision.teardown_thing("valid-robot-name_123")

        assert mock_called, "_require_boto3 should be called after validation passes"


class TestTeardownThingCertDirParity:
    """teardown_thing must honour a custom cert_dir to match provision_robot.

    Regression for the asymmetry where provision_robot accepted ``cert_dir=``
    but teardown_thing was hardcoded to DEFAULT_CERT_DIR -- callers who
    provisioned with a custom cert_dir would silently leak .cert.pem and
    .private.key on disk forever after teardown.
    """

    def test_cert_dir_kwarg_unlinks_local_files(self, tmp_path, monkeypatch):
        """teardown_thing(thing, cert_dir=tmp) must unlink files under tmp,
        not under DEFAULT_CERT_DIR."""
        from strands_robots.mesh.iot import provision

        # Seed cert + key files in a custom dir
        custom_dir = tmp_path / "iot"
        custom_dir.mkdir()
        cert = custom_dir / "test-robot.cert.pem"
        key = custom_dir / "test-robot.private.key"
        cert.write_text("FAKE CERT")
        key.write_text("FAKE KEY")
        assert cert.exists() and key.exists()

        # Stub boto3 + the iot client so teardown reaches the unlink loop
        # without making real AWS calls.
        fake_iot = type("FakeIoT", (), {})()

        class _NotFound(Exception):
            pass

        fake_iot.exceptions = type(
            "FakeExc",
            (),
            {"ResourceNotFoundException": _NotFound},
        )()
        fake_iot.list_thing_principals = lambda **kw: {"principals": []}
        fake_iot.delete_thing = lambda **kw: None

        fake_boto3 = type("FakeBoto3", (), {})()
        fake_boto3.client = lambda *a, **kw: fake_iot
        monkeypatch.setattr(provision, "_require_boto3", lambda: fake_boto3)

        provision.teardown_thing("test-robot", cert_dir=custom_dir)

        # Both files should now be gone under the custom dir.
        assert not cert.exists(), "cert.pem leaked under custom cert_dir"
        assert not key.exists(), "private.key leaked under custom cert_dir"

    def test_no_public_key_suffix_attempted(self, tmp_path, monkeypatch):
        """``.public.key`` was dead code (``_create_cert`` never writes it).
        teardown_thing should not attempt to unlink it."""
        from strands_robots.mesh.iot import provision

        custom_dir = tmp_path / "iot"
        custom_dir.mkdir()

        attempted: list[str] = []
        original_unlink = type(custom_dir).unlink

        def _track_unlink(self, *a, **kw):
            attempted.append(self.name)
            return original_unlink(self, *a, **kw)

        monkeypatch.setattr(type(custom_dir), "unlink", _track_unlink)

        fake_iot = type("FakeIoT", (), {})()

        class _NotFound(Exception):
            pass

        fake_iot.exceptions = type(
            "FakeExc",
            (),
            {"ResourceNotFoundException": _NotFound},
        )()
        fake_iot.list_thing_principals = lambda **kw: {"principals": []}
        fake_iot.delete_thing = lambda **kw: None
        fake_boto3 = type("FakeBoto3", (), {})()
        fake_boto3.client = lambda *a, **kw: fake_iot
        monkeypatch.setattr(provision, "_require_boto3", lambda: fake_boto3)

        # Pre-seed only the suffixes _create_cert actually writes.
        (custom_dir / "robot.cert.pem").write_text("c")
        (custom_dir / "robot.private.key").write_text("k")

        provision.teardown_thing("robot", cert_dir=custom_dir)

        # We should never have attempted a `.public.key` suffix.
        assert not any(name.endswith(".public.key") for name in attempted), (
            f"unexpected .public.key unlink attempt in {attempted}"
        )


class TestTeardownThingDocstringShape:
    """Pin: teardown_thing's docstring must render with consistent indentation.

    Regression marker for two related defects in the same docstring:

    1. R7 docstring-typo bug (a literal `n` glyph between the body and the
       `Note:` section -- the failure mode of an editor inserting `n` instead
       of `\n`).
    2. R7-fix indentation bug: the original repair left the body at 8 spaces,
       the `Note:` heading at 4, and the Note body at 12 -- which `cleandoc`
       resolved to a body indented as if it were a literal blockquote with the
       Note body double-indented underneath. Sphinx / pdoc / IDE help renderers
       treat that as a malformed Google-style docstring.

    The pin tests assert on *post-cleandoc structure*, not adjacent substrings,
    because the original R7-fix pin asserted only on substring presence and
    missed the whole indentation regression. A pin test must reject the same
    failure mode it was added to prevent.
    """

    def test_no_stray_n_literal_in_docstring(self):
        """The docstring must not contain a bare `n    Note:` artefact."""
        from strands_robots.mesh.iot.provision import teardown_thing

        ds = teardown_thing.__doc__
        assert ds is not None, "teardown_thing.__doc__ went missing"
        # The R7 typo manifested as a literal 'n    Note:' on a line by itself
        # (where '\n    Note:' was intended). Pin the absence of that artefact.
        assert "n    Note:" not in ds, "stray 'n' before Note: section -- R7 docstring typo regression"
        # And keep the Note section itself, since the original R7 fix was
        # adding the cert_dir trust note.
        assert "Note:" in ds, "Note: section must remain"
        assert "trusted operator input" in ds, "cert_dir trust note (R7) must remain"

    def test_cleandoc_renders_consistent_indentation(self):
        """After ``inspect.cleandoc``, body, ``Note:`` heading, and Note body
        must use the Google-style indent ladder: body and heading at 0, Note
        body at 4. The R7-fix indentation bug rendered body at 4 (blockquote)
        and Note body at 8 (double-indented).
        """
        import inspect

        from strands_robots.mesh.iot.provision import teardown_thing

        cleaned = inspect.cleandoc(teardown_thing.__doc__ or "")
        lines = cleaned.split("\n")

        # Summary line at column 0.
        assert lines[0].startswith("Detach + delete"), "summary line missing"
        assert not lines[0].startswith(" "), "summary line must be at column 0"

        # Body paragraph ("Cleans up the cert files") must be at column 0,
        # not indented as a literal blockquote.
        body_line = next(ln for ln in lines if ln.lstrip().startswith("Cleans up the cert files"))
        assert body_line == body_line.lstrip(), (
            f"docstring body must render at column 0 after cleandoc; "
            f"got {len(body_line) - len(body_line.lstrip())} leading spaces"
        )

        # `Note:` heading at column 0.
        note_heading = next(ln for ln in lines if ln.rstrip() == "Note:")
        assert note_heading == "Note:", f"`Note:` heading must render at column 0; got {note_heading!r}"

        # Note body at exactly 4 spaces (one indent level under heading).
        note_body_line = next(ln for ln in lines if ln.lstrip().startswith("``cert_dir`` is treated"))
        leading = len(note_body_line) - len(note_body_line.lstrip())
        assert leading == 4, f"Note body must render at 4 spaces (Google-style); got {leading}"
