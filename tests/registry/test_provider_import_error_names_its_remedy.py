"""A provider whose optional dependency is missing reports its own remedy.

Every policy provider except one defers its heavy import and reports a missing
dependency through :func:`~strands_robots.utils.require_optional` /
:func:`~strands_robots.utils.require_optionals`, which name the extra that ships
it. ``lerobot_local`` imports ``torch`` at module level, so the failure happens
while importing the provider's module - before any provider machinery runs - and
the bare ``ModuleNotFoundError: No module named 'torch'`` that escaped named
neither the provider the caller asked for nor the way to fix it.

:func:`~strands_robots.registry.policies.import_policy_class` is the single
funnel every provider class is imported through, so the translation lives there
and the remedy no longer depends on WHERE a provider imports its dependency.

Two further contracts are pinned here because both were reported as defects and
neither existed:

* No provider is ever SUBSTITUTED for another when its dependency is missing.
  A silent swap to ``mock`` was reported; the factory has never done it, and
  this pins that it cannot start.
* A provider whose module exists but whose dependency is missing is not
  misreported as an unknown provider. The auto-discovery branch swallowed the
  ``ImportError`` and raised ``Unknown policy provider``, sending a caller who
  had the name right to go and check the name.

The absent dependency is emulated by making the import the funnel performs raise
the exact ``ModuleNotFoundError`` the real import system raises (``name`` set),
so the production path runs unchanged and the tests need no minimal install.
"""

import contextlib
import json
import pathlib
import tomllib
from typing import Any

import pytest

import strands_robots.registry.policies as policies_mod
from strands_robots.policies import create_policy

#: Extra that ships ``lerobot_local``'s dependency, as declared in policies.json.
_LEROBOT_EXTRA = "lerobot"

#: Providers whose module needs an optional dependency, so a substitution could hide there.
_SUBSTITUTION_CANDIDATES = ("lerobot_local", "groot", "cosmos3", "wbc")


def _absent(module: str) -> Any:
    """Build an import_module stand-in that reports ``module`` as not installed.

    Args:
        module: Top-level module name to report absent.

    Returns:
        A callable raising the same ``ModuleNotFoundError`` (with ``name`` set)
        that the import system raises for a package that is not installed.
    """

    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError(f"No module named {module!r}", name=module)

    return _import


def _repo_root() -> pathlib.Path:
    """Return the repository root, derived from the package under test.

    ``policies.py`` sits at ``<root>/strands_robots/registry/``, so the root is
    two parents up from the package directory. Derived from the module rather
    than from a path literal so the guard follows the package.
    """
    root = pathlib.Path(policies_mod.__file__).resolve().parents[2]
    assert (root / "pyproject.toml").is_file(), f"repo root not resolved: {root}"
    return root


class TestAMissingDependencyReportsItsRemedy:
    """The translated error names the provider, the module and the fix."""

    def test_it_names_the_provider_the_missing_module_and_the_install_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(policies_mod.importlib, "import_module", _absent("torch"))
        with pytest.raises(ImportError) as excinfo:
            policies_mod.import_policy_class("lerobot_local")
        message = str(excinfo.value)
        assert "lerobot_local" in message, message
        assert "torch" in message, message
        assert f"strands-robots[{_LEROBOT_EXTRA}]" in message, message

    def test_the_original_import_error_is_kept_as_the_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(policies_mod.importlib, "import_module", _absent("torch"))
        with pytest.raises(ImportError) as excinfo:
            policies_mod.import_policy_class("lerobot_local")
        cause = excinfo.value.__cause__
        assert isinstance(cause, ModuleNotFoundError), cause
        assert getattr(cause, "name", None) == "torch"

    def test_create_policy_propagates_the_actionable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The report reaches the caller through the public entry point."""
        monkeypatch.setattr(policies_mod.importlib, "import_module", _absent("torch"))
        with pytest.raises(ImportError) as excinfo:
            create_policy("lerobot_local", pretrained_name_or_path="allenai/MolmoAct2-SO100_101")
        message = str(excinfo.value)
        assert f"strands-robots[{_LEROBOT_EXTRA}]" in message, message

    def test_a_provider_declaring_no_extra_still_names_the_missing_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a declared extra the module is still named, not swallowed."""
        monkeypatch.setattr(policies_mod.importlib, "import_module", _absent("pyzmq_stand_in"))
        with pytest.raises(ImportError) as excinfo:
            policies_mod.import_policy_class("groot")
        message = str(excinfo.value)
        assert "groot" in message, message
        assert "pyzmq_stand_in" in message, message


class TestNoProviderIsSubstituted:
    """A missing dependency never yields a different provider's policy.

    Reported as a silent swap to ``mock``; the factory has never done it. Pinned
    so it cannot begin to - a swapped provider whose action space happens to fit
    is indistinguishable from the requested one at the call site.
    """

    @pytest.mark.parametrize("provider", _SUBSTITUTION_CANDIDATES)
    def test_a_missing_dependency_raises_rather_than_substituting(
        self, provider: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(policies_mod.importlib, "import_module", _absent("torch"))
        with pytest.raises(ImportError) as excinfo:
            policies_mod.import_policy_class(provider)
        assert "mock" not in str(excinfo.value).lower()

    @pytest.mark.parametrize("provider", _SUBSTITUTION_CANDIDATES)
    def test_the_funnel_never_falls_back_to_the_mock_provider(
        self, provider: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing the funnel returns on the failure path is a policy at all."""
        monkeypatch.setattr(policies_mod.importlib, "import_module", _absent("torch"))
        returned: list[Any] = []
        # The refusal itself is asserted by the cases above; this one is only about
        # what the funnel RETURNS, so the ImportError is suppressed rather than
        # required -- a tree that stopped raising would still have to append nothing
        # for this to pass, which is the substitution being ruled out.
        with contextlib.suppress(ImportError):
            returned.append(policies_mod.import_policy_class(provider))
        assert returned == [], f"the funnel returned {returned!r} for {provider!r} instead of reporting"


class TestAnUnknownProviderIsStillUnknown:
    """The translation does not swallow the unknown-provider report.

    Over-reach control: the two failures are distinct and must stay distinct.
    """

    def test_an_unknown_name_is_a_value_error_naming_the_available_providers(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            policies_mod.import_policy_class("no_such_provider_at_all")
        message = str(excinfo.value)
        assert "no_such_provider_at_all" in message, message
        assert "Available" in message, message

    def test_an_existing_module_with_a_missing_dependency_is_not_reported_as_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The auto-discovery branch must not blame the provider name.

        A module that imports but cannot satisfy its dependency reports the
        dependency. Reported as an unknown provider it sent a caller whose name
        was correct to go and check the name.
        """
        monkeypatch.setattr(policies_mod.importlib, "import_module", _absent("some_missing_dep"))
        with pytest.raises(ImportError) as excinfo:
            policies_mod.import_policy_class("not_in_the_registry")
        message = str(excinfo.value)
        assert "some_missing_dep" in message, message
        assert "Unknown policy provider" not in message, message


class TestTheDeclaredExtrasAreReal:
    """Every extra a provider names must be installable.

    A remedy naming an extra that does not exist is a dead end wearing the
    clothes of an instruction: ``pip`` exits 0 for an unknown extra and
    installs nothing.
    """

    def test_every_provider_extra_is_declared_in_pyproject(self) -> None:
        registry = json.loads(
            (pathlib.Path(policies_mod.__file__).parent / "policies.json").read_text(encoding="utf-8")
        )
        pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
        declared = set(pyproject["project"]["optional-dependencies"])
        named = {name: cfg["extra"] for name, cfg in registry["providers"].items() if cfg.get("extra") is not None}
        assert named, "no provider declares an extra - this guard would pass vacuously"
        assert named.get("lerobot_local") == _LEROBOT_EXTRA
        undeclared = {p: e for p, e in named.items() if e not in declared}
        assert not undeclared, f"providers naming an undeclared extra: {undeclared}"


class TestAUsableProviderStillImports:
    """Over-reach control: the translation only fires on a failed import."""

    def test_a_provider_with_no_optional_dependency_imports_unchanged(self) -> None:
        cls = policies_mod.import_policy_class("mock")
        assert cls.__name__ == "MockPolicy"
