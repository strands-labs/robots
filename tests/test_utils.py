"""Tests for strands_robots.utils - require_optional lazy import helper."""

import numpy as np
import pytest

from strands_robots.utils import (
    coerce_pose_vector,
    process_rss_mb,
    require_optional,
    require_optionals,
)


class TestRequireOptional:
    """Tests for the require_optional lazy import utility."""

    def test_imports_stdlib_module(self):
        """Should successfully import a stdlib module."""
        mod = require_optional("json")
        assert hasattr(mod, "dumps")

    def test_caches_module(self):
        """Second call should return the cached module (same object)."""
        mod1 = require_optional("json")
        mod2 = require_optional("json")
        assert mod1 is mod2

    def test_missing_module_raises_import_error(self):
        """Non-existent module should raise ImportError."""
        with pytest.raises(ImportError):
            require_optional("nonexistent_module_xyz_12345")

    def test_error_message_includes_module_name(self):
        with pytest.raises(ImportError, match="nonexistent_module_xyz"):
            require_optional("nonexistent_module_xyz")

    def test_error_message_includes_purpose(self):
        with pytest.raises(ImportError, match="for testing"):
            require_optional("nonexistent_xyz", purpose="testing")

    def test_error_message_includes_pip_install(self):
        with pytest.raises(ImportError, match="pip install my-package"):
            require_optional("nonexistent_xyz", pip_install="my-package")

    def test_error_message_includes_extra(self):
        with pytest.raises(ImportError, match="strands-robots\\[my-extra\\]"):
            require_optional("nonexistent_xyz", extra="my-extra")

    def test_error_message_default_pip_install(self):
        """When pip_install is not set, should use module_name."""
        with pytest.raises(ImportError, match="pip install nonexistent_xyz"):
            require_optional("nonexistent_xyz")

    def test_dotted_module(self):
        """Should handle dotted module names like os.path."""
        mod = require_optional("os.path")
        assert hasattr(mod, "join")

    def test_system_install_replaces_the_pip_block(self):
        """A system-provided module must be refused with its own remedy, no pip line.

        Some modules arrive with a system package and are absent from PyPI, so any
        ``pip install`` line is an instruction the caller can follow to no effect.
        ``system_install`` is the way to say that, and it has to leave the message
        with no pip command at all.
        """
        with pytest.raises(ImportError) as excinfo:
            require_optional(
                "nonexistent_xyz",
                purpose="testing",
                system_install="Source a distro:\n  source /opt/ros/jazzy/setup.bash",
            )
        message = str(excinfo.value)
        assert "'nonexistent_xyz' is required for testing" in message
        assert "source /opt/ros/jazzy/setup.bash" in message
        assert "pip install" not in message
        assert "Install with:" not in message

    def test_system_install_wins_over_a_pip_remedy(self):
        """Neither ``extra`` nor ``pip_install`` may leak back into the message.

        A caller migrating a site is likely to leave the old arguments in place;
        emitting both would put the dead-end command back in front of the reader
        next to the one that works.
        """
        with pytest.raises(ImportError) as excinfo:
            require_optional(
                "nonexistent_xyz",
                pip_install="not-a-real-distribution",
                extra="my-extra",
                system_install="Source a distro first.",
            )
        message = str(excinfo.value)
        assert message.endswith("Source a distro first.")
        assert "not-a-real-distribution" not in message
        assert "my-extra" not in message

    def test_pip_block_is_unchanged_without_system_install(self):
        """Omitting ``system_install`` must leave the pip remedy exactly as it was."""
        with pytest.raises(ImportError) as excinfo:
            require_optional("nonexistent_xyz", purpose="testing", extra="my-extra", pip_install="my-package")
        assert str(excinfo.value) == (
            "'nonexistent_xyz' is required for testing\n"
            "Install with:\n"
            "  pip install 'strands-robots[my-extra]'\n"
            "  pip install my-package"
        )


# safe_join / get_search_paths tests (added for PR #84 follow-up)


class TestSafeJoin:
    """Tests for the centralised path-traversal guard."""

    def test_joins_clean_paths(self, tmp_path):
        from strands_robots.utils import safe_join

        result = safe_join(tmp_path, "robot/model.xml")
        assert result == tmp_path / "robot" / "model.xml"

    def test_rejects_traversal(self, tmp_path):
        from strands_robots.utils import safe_join

        with pytest.raises(ValueError, match="Path traversal blocked"):
            safe_join(tmp_path, "../etc/passwd")

    def test_rejects_absolute_escape(self, tmp_path):
        from strands_robots.utils import safe_join

        with pytest.raises(ValueError, match="Path traversal blocked"):
            safe_join(tmp_path, "robot/../../etc/passwd")

    def test_same_path_is_allowed(self, tmp_path):
        from strands_robots.utils import safe_join

        # Empty / dot path resolves to base itself - must not raise
        result = safe_join(tmp_path, ".")
        assert result == tmp_path

    def test_rejects_symlink_escape_when_resolving(self, tmp_path):
        from strands_robots.utils import safe_join

        # A symlink whose target lies outside *base* stays lexically under base
        # yet resolves outside it; with resolve_symlinks it must be rejected.
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret").write_text("x")
        base = tmp_path / "base"
        base.mkdir()
        (base / "link").symlink_to(outside)
        with pytest.raises(ValueError, match="via symlink"):
            safe_join(base, "link/secret", resolve_symlinks=True)

    def test_symlink_escape_allowed_by_default(self, tmp_path):
        from strands_robots.utils import safe_join

        # Default (lexical) mode returns the managed path even when a component
        # is a symlink: the asset cache intentionally symlinks robot dirs to
        # installed packages outside the cache, and must not be rejected.
        outside = tmp_path / "outside"
        outside.mkdir()
        base = tmp_path / "base"
        base.mkdir()
        (base / "link").symlink_to(outside)
        assert safe_join(base, "link/model.xml") == base / "link" / "model.xml"

    def test_allows_symlink_within_base_when_resolving(self, tmp_path):
        from strands_robots.utils import safe_join

        # A symlink that resolves back inside *base* must still be permitted so
        # the symlink hardening does not over-block legitimate asset layouts.
        base = tmp_path / "base"
        (base / "real").mkdir(parents=True)
        (base / "alias").symlink_to(base / "real")
        result = safe_join(base, "alias/model.xml", resolve_symlinks=True)
        assert result == base / "alias" / "model.xml"


class TestGetSearchPaths:
    """Tests for the centralised search-path resolver."""

    def test_returns_assets_dir_and_cwd_assets(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        from strands_robots.utils import get_search_paths

        paths = get_search_paths()
        assert tmp_path in paths
        assert (tmp_path / "assets") in paths

    def test_returns_unique_paths(self, tmp_path, monkeypatch):
        # When CWD is already the assets dir, we shouldn't list the same path twice
        # (deduping is explicit in the implementation).
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
        from strands_robots.utils import get_search_paths

        paths = get_search_paths()
        assert len(paths) == len(set(paths))


class TestRequireOptionals:
    """Tests for require_optionals - aggregate multi-dep gate.

    Unlike require_optional in a loop (which raises on the FIRST missing dep and
    hides the rest), require_optionals reports EVERY missing dep in one error so
    a partially-provisioned environment is fixed in a single install.
    """

    def test_all_present_returns_none(self):
        """When every module is importable the gate passes silently."""
        assert require_optionals(["json", "os.path"]) is None

    def test_lists_every_missing_module(self):
        """A single error must name ALL missing deps, not just the first."""
        with pytest.raises(ImportError) as exc:
            require_optionals(["nope_aaa_xyz", "nope_bbb_xyz", "nope_ccc_xyz"])
        msg = str(exc.value)
        assert "nope_aaa_xyz" in msg
        assert "nope_bbb_xyz" in msg
        assert "nope_ccc_xyz" in msg

    def test_reports_only_the_missing_ones(self):
        """Present deps are not named; only the absent ones surface."""
        with pytest.raises(ImportError) as exc:
            require_optionals(["json", "nope_present_mix_xyz"])
        msg = str(exc.value)
        assert "nope_present_mix_xyz" in msg
        assert "json" not in msg

    def test_pip_install_line_lists_all_missing(self):
        """The bare ``pip install`` hint concatenates every missing module."""
        with pytest.raises(ImportError, match=r"pip install nope_d1_xyz nope_d2_xyz"):
            require_optionals(["nope_d1_xyz", "nope_d2_xyz"])

    def test_error_includes_extra_and_purpose(self):
        """extra and purpose flow into the message like require_optional."""
        with pytest.raises(ImportError) as exc:
            require_optionals(["nope_extra_xyz"], extra="my-extra", purpose="testing")
        msg = str(exc.value)
        assert "strands-robots[my-extra]" in msg
        assert "for testing" in msg

    def test_singular_vs_plural_phrasing(self):
        """One missing dep reads 'is required'; many read 'are required'."""
        with pytest.raises(ImportError, match="is required"):
            require_optionals(["nope_single_xyz"])
        with pytest.raises(ImportError, match="are required"):
            require_optionals(["nope_two_a_xyz", "nope_two_b_xyz"])


# Path-resolution tests (get_base_dir / get_assets_dir / resolve_asset_path)


class TestPathResolution:
    """Tests for the path-resolution helpers.

    These functions decide where ALL strands-robots user data, assets, and
    model files land. They honour two env-var overrides whose contracts differ:
    ``STRANDS_BASE_DIR`` relocates the entire base dir, while
    ``STRANDS_ASSETS_DIR`` relocates ONLY the assets subdir and must never move
    the base dir (so user-level metadata stays in a predictable location).
    """

    def test_get_base_dir_honors_env_override(self, tmp_path, monkeypatch):
        from strands_robots.utils import get_base_dir

        target = tmp_path / "custom_base"
        monkeypatch.setenv("STRANDS_BASE_DIR", str(target))

        result = get_base_dir()
        assert result == target
        assert result.is_dir()  # created if needed

    def test_get_base_dir_default(self, tmp_path, monkeypatch):
        from strands_robots.utils import get_base_dir

        monkeypatch.delenv("STRANDS_BASE_DIR", raising=False)
        default = tmp_path / "sr_default"
        monkeypatch.setattr("strands_robots.utils.DEFAULT_BASE_DIR", default)

        result = get_base_dir()
        assert result == default
        assert result.is_dir()

    def test_get_assets_dir_honors_env_override(self, tmp_path, monkeypatch):
        from strands_robots.utils import get_assets_dir

        target = tmp_path / "custom_assets"
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(target))

        result = get_assets_dir()
        assert result == target
        assert result.is_dir()

    def test_get_assets_dir_default(self, tmp_path, monkeypatch):
        from strands_robots.utils import get_assets_dir

        monkeypatch.delenv("STRANDS_ASSETS_DIR", raising=False)
        default = tmp_path / "sr_default"
        monkeypatch.setattr("strands_robots.utils.DEFAULT_BASE_DIR", default)

        result = get_assets_dir()
        assert result == default / "assets"
        assert result.is_dir()

    def test_assets_env_does_not_move_base_dir(self, tmp_path, monkeypatch):
        """Documented contract: STRANDS_ASSETS_DIR moves ONLY the assets dir.

        The base dir must stay at its default even when assets are relocated, so
        user-level metadata (e.g. user_robots.json) lands predictably.
        """
        from strands_robots.utils import get_assets_dir, get_base_dir

        assets_target = tmp_path / "elsewhere_assets"
        base_default = tmp_path / "sr_default"
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets_target))
        monkeypatch.delenv("STRANDS_BASE_DIR", raising=False)
        monkeypatch.setattr("strands_robots.utils.DEFAULT_BASE_DIR", base_default)

        assert get_assets_dir() == assets_target
        # Base dir is unaffected by the assets override.
        assert get_base_dir() == base_default

    def test_resolve_asset_path_none_uses_default_name(self, tmp_path, monkeypatch):
        from strands_robots.utils import resolve_asset_path

        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))

        result = resolve_asset_path(None, default_name="g1")
        assert result == tmp_path / "g1"

    def test_resolve_asset_path_relative_under_assets(self, tmp_path, monkeypatch):
        from strands_robots.utils import resolve_asset_path

        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))

        result = resolve_asset_path("robot/model.xml")
        assert result == tmp_path / "robot" / "model.xml"

    def test_resolve_asset_path_absolute_returned_as_is(self, tmp_path, monkeypatch):
        from strands_robots.utils import resolve_asset_path

        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path / "assets"))
        absolute = tmp_path / "outside" / "model.xml"

        result = resolve_asset_path(absolute)
        assert result == absolute
        # An absolute input escapes the assets sandbox by design.
        assert not str(result).startswith(str(tmp_path / "assets"))

    def test_resolve_asset_path_expands_tilde(self, tmp_path, monkeypatch):
        from strands_robots.utils import resolve_asset_path

        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path / "assets"))
        monkeypatch.setenv("HOME", str(tmp_path / "home"))

        result = resolve_asset_path("~/models/policy.pt")
        # ~ expands to an absolute home path, so it is returned as-is, not
        # joined under the assets dir.
        assert result == tmp_path / "home" / "models" / "policy.pt"
        assert not str(result).startswith(str(tmp_path / "assets"))


class TestProcessRssMb:
    """Tests for process_rss_mb resident-memory telemetry helper.

    Covers all three resolution paths: psutil (preferred), the
    resource.getrusage fallback (with Linux/macOS unit normalisation), and the
    None result when neither source is available.
    """

    def test_psutil_path_returns_positive_float(self):
        """With psutil available, returns this process's current RSS in MB."""
        rss = process_rss_mb()
        assert isinstance(rss, float)
        assert rss > 0.0

    def test_resource_fallback_linux_normalises_kilobytes(self, monkeypatch):
        """psutil absent + Linux: ru_maxrss is kilobytes, divided by 1024 -> MB."""
        import sys
        import types

        monkeypatch.setitem(sys.modules, "psutil", None)
        fake_resource = types.ModuleType("resource")
        fake_resource.RUSAGE_SELF = 0
        fake_resource.getrusage = lambda _who: types.SimpleNamespace(ru_maxrss=2048)
        monkeypatch.setitem(sys.modules, "resource", fake_resource)
        monkeypatch.setattr(sys, "platform", "linux")

        assert process_rss_mb() == pytest.approx(2.0)

    def test_resource_fallback_macos_normalises_bytes(self, monkeypatch):
        """psutil absent + macOS: ru_maxrss is bytes, divided by 1024**2 -> MB."""
        import sys
        import types

        monkeypatch.setitem(sys.modules, "psutil", None)
        fake_resource = types.ModuleType("resource")
        fake_resource.RUSAGE_SELF = 0
        fake_resource.getrusage = lambda _who: types.SimpleNamespace(ru_maxrss=2 * 1024 * 1024)
        monkeypatch.setitem(sys.modules, "resource", fake_resource)
        monkeypatch.setattr(sys, "platform", "darwin")

        assert process_rss_mb() == pytest.approx(2.0)

    def test_returns_none_when_no_source_available(self, monkeypatch):
        """Neither psutil nor resource importable: returns None, not a misleading 0."""
        import sys

        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setitem(sys.modules, "resource", None)

        assert process_rss_mb() is None


class TestPoseVectorDomain:
    """Accepted domain of the shared pose-vector guard.

    Promoted out of the MuJoCo facade so the scene-construction calls and the
    motion primitives (which live in a module the facade imports) cannot hold
    different opinions about the same ``[x, y, z]``.
    """

    def test_an_omitted_vector_is_not_an_error(self):
        assert coerce_pose_vector("m", "position", None, 3) == (None, None)

    def test_numpy_components_are_normalized_to_plain_floats(self):
        values, error = coerce_pose_vector("m", "position", np.array([0.1, 0.2, 0.3]), 3)
        assert error is None
        assert all(type(v) is float for v in values)

    @pytest.mark.parametrize(
        "bad", [0.5, np.float64(1.0), [0.1, 0.2], [0.1, "b", 0.3], [0.1, float("nan"), 0.3], [True, 0.2, 0.3]]
    )
    def test_a_vector_that_cannot_be_honored_returns_a_message(self, bad):
        values, error = coerce_pose_vector("m", "position", bad, 3)
        assert values is None
        assert error and error.startswith("m: 'position'")

    @pytest.mark.parametrize("text", ["box", "cube", "0.1,0.2,0.3", "123", b"abc"])
    def test_a_string_is_refused_on_its_type_not_its_length(self, text):
        """A string names its own type in the refusal, whatever its length.

        Every one of these was already refused, so this pins WHICH question the
        caller is sent to fix. A string carries a length, so before the type guard
        the verdict was picked by that length: at ``expected_len`` 3, ``"box"``
        drew ``elements must be numbers``, ``"cube"`` drew a wrong element *count*
        of 4, and ``"0.1,0.2,0.3"`` drew a count of 11 - one mistake reported three
        ways, two of them describing the string's characters as though they were
        pose components. ``add_camera(target="cube")`` is the call that found it,
        a camera aimed at a named body being what the parameter looks like it takes.
        """
        values, error = coerce_pose_vector("add_camera", "target", text, 3)
        assert values is None
        assert error is not None
        assert error.startswith("add_camera: 'target' must be a list/tuple of 3 numbers")
        assert type(text).__name__ in error
        # The character count must not appear as a component count. "123" would
        # make this vacuous by containing its own digits, so it is checked on the
        # phrase the length gate produces rather than on the number.
        assert "-element vector" not in error
        assert "elements must be numbers" not in error


class TestLerobotVersion:
    """``lerobot_version`` degrades instead of raising.

    Promoted out of the streaming-dataset reader so it and
    :mod:`strands_robots.dataset_recorder` - which both name the installed
    version in an error message - cannot report it differently.
    """

    def test_an_unresolvable_distribution_reads_as_unknown(self, monkeypatch):
        # A source checkout without dist-info cannot resolve the version. The
        # value only ever enriches an error message, so a PackageNotFoundError
        # must not escape and mask the failure being reported.
        import importlib.metadata as md

        from strands_robots.utils import lerobot_version

        def _raise(_name):
            raise md.PackageNotFoundError("lerobot")

        monkeypatch.setattr(md, "version", _raise)

        assert lerobot_version() == "unknown"

    def test_a_resolvable_distribution_reads_as_its_version(self, monkeypatch):
        import importlib.metadata as md

        from strands_robots.utils import lerobot_version

        monkeypatch.setattr(md, "version", lambda _name: "0.6.1")

        assert lerobot_version() == "0.6.1"

    def test_an_unimportable_metadata_module_reads_as_unknown(self, monkeypatch):
        """The other documented failure - the import itself - must also degrade.

        ``importlib.metadata`` is stdlib, so this handler is the defensive half.
        A handler that cannot run is not a defence though: naming
        ``PackageNotFoundError`` beside ``ImportError`` bound it as a local the
        failing import never reaches, so evaluating the handler raised
        ``UnboundLocalError`` - from a function documented never to raise, and
        exactly on the branch the second name appeared to cover.
        """
        import builtins

        from strands_robots.utils import lerobot_version

        real_import = builtins.__import__

        def _no_metadata(name, *args, **kwargs):
            if name == "importlib.metadata":
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_metadata)

        assert lerobot_version() == "unknown"

    def test_an_unresolvable_distribution_is_already_an_import_error(self):
        """The single handler rests on a stdlib hierarchy, so assert it.

        ``version`` raises ``PackageNotFoundError`` and the one-name handler
        catches it as an ``ImportError``. Were a future CPython to stop deriving
        it from ``ModuleNotFoundError``, that handler would silently stop
        catching it, so the premise is pinned here rather than left to prose.
        """
        import importlib.metadata as md

        assert issubclass(md.PackageNotFoundError, ImportError)
