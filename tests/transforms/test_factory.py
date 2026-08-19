"""Tests for the dataset-transform factory: registry, aliases, discovery."""

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

    def test_auto_discovery_import(self):
        """``import_transform_class`` finds the module-per-provider layout."""
        assert import_transform_class("mock") is MockTransform
        assert import_transform_class("cosmos_transfer") is CosmosTransferTransform

    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DatasetTransform()  # type: ignore[abstract]
