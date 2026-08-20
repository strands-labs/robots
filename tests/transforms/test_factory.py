"""Tests for the dataset-transform factory: registry, aliases, discovery."""

import importlib

import pytest

from strands_robots.transforms import (
    DatasetTransform,
    create_transform,
    import_transform_class,
    list_transforms,
    register_transform,
)
from strands_robots.transforms.cosmos_transfer import CosmosTransferTransform
from strands_robots.transforms.mock import MockTransform


class TestFactory:
    """The factory resolves the built-ins, runtime names and refusals."""

    def test_create_mock(self):
        """``mock`` resolves through the package's built-in registration."""
        t = create_transform("mock")
        assert isinstance(t, MockTransform)
        assert t.provider_name == "mock"

    def test_create_cosmos_transfer(self):
        """``cosmos_transfer`` constructs without any generation stack installed."""
        t = create_transform("cosmos_transfer")
        assert isinstance(t, CosmosTransferTransform)
        assert t.provider_name == "cosmos_transfer"

    def test_list_transforms_includes_builtins(self):
        registered = list_transforms()
        assert "mock" in registered
        assert "cosmos_transfer" in registered

    def test_kwargs_passthrough(self):
        """Constructor kwargs reach the transform class."""
        t = create_transform("mock", pixel_shift=10)
        assert isinstance(t, MockTransform)

    def test_runtime_register_and_alias(self):
        register_transform("custom_v2v", lambda: MockTransform, aliases=["cv"])
        assert isinstance(create_transform("custom_v2v"), MockTransform)
        assert isinstance(create_transform("cv"), MockTransform)
        assert "custom_v2v" in list_transforms()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="No transform registered"):
            create_transform("does_not_exist_xyz")

    def test_discovered_module_with_a_missing_dependency_surfaces_the_import_error(self, tmp_path, monkeypatch):
        """A provider module that EXISTS but whose backend dep is absent is not "unregistered".

        Collapsing that ``ModuleNotFoundError`` into the "No transform
        registered ... Available transforms" refusal sends the user to the
        wrong problem: the fix is installing the dependency, not spelling the
        provider differently. Only the absence of the provider module itself
        may fall through to the ValueError.
        """
        import strands_robots.transforms as transforms_pkg

        (tmp_path / "heavybackend.py").write_text(
            '"""Fake discovered transform whose backend SDK is not installed."""\n'
            "import a_backend_sdk_that_is_not_installed_xyz  # noqa: F401\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(transforms_pkg, "__path__", [*transforms_pkg.__path__, str(tmp_path)])
        importlib.invalidate_caches()
        with pytest.raises(ModuleNotFoundError, match="a_backend_sdk_that_is_not_installed_xyz"):
            import_transform_class("heavybackend")

    def test_auto_discovery_import(self):
        """``import_transform_class`` finds the module-per-provider layout."""
        assert import_transform_class("mock") is MockTransform
        assert import_transform_class("cosmos_transfer") is CosmosTransferTransform

    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DatasetTransform()  # type: ignore[abstract]
