#!/usr/bin/env python3
"""Tests for Cosmos Transfer 2.5 integration."""

import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE = os.path.join(os.path.dirname(__file__), "..", "strands_robots")


class TestCosmosTransferSyntax:
    """Validate Cosmos Transfer module files parse correctly."""

    def _check_syntax(self, filepath):
        with open(filepath, "r") as f:
            source = f.read()
        ast.parse(source, filename=filepath)

    def test_cosmos_transfer_init_syntax(self):
        self._check_syntax(os.path.join(BASE, "cosmos_transfer", "__init__.py"))


class TestCosmosTransferModuleStructure:
    """Test Cosmos Transfer module exports and structure."""

    def test_import_module(self):
        from strands_robots.cosmos_transfer import __all__

        assert "CosmosTransferPipeline" in __all__
        assert "CosmosTransferConfig" in __all__
        assert "transfer_video" in __all__

    def test_lazy_import_no_gpu(self):
        """Cosmos Transfer module should import without GPU/torch installed."""
        from strands_robots import cosmos_transfer

        assert hasattr(cosmos_transfer, "__all__")
        assert len(cosmos_transfer.__all__) == 3


class TestCosmosTransferPipelineContract:
    """Verify CosmosTransferPipeline has the expected API."""

    def test_pipeline_class_exists(self):
        with open(os.path.join(BASE, "cosmos_transfer", "__init__.py")) as f:
            source = f.read()
        tree = ast.parse(source)

        class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        assert "CosmosTransferPipeline" in class_names
        assert "CosmosTransferConfig" in class_names

    def test_pipeline_has_transfer_video(self):
        """CosmosTransferPipeline must have transfer_video method."""
        with open(os.path.join(BASE, "cosmos_transfer", "__init__.py")) as f:
            source = f.read()
        tree = ast.parse(source)

        pipeline_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "CosmosTransferPipeline":
                pipeline_class = node
                break

        assert pipeline_class is not None

        method_names = {node.name for node in ast.walk(pipeline_class) if isinstance(node, ast.FunctionDef)}

        required = {"transfer_video", "generate_depth_control", "generate_edge_control"}
        missing = required - method_names
        assert not missing, f"CosmosTransferPipeline missing methods: {missing}"

    def test_config_has_control_types(self):
        """CosmosTransferConfig should support multiple control types."""
        with open(os.path.join(BASE, "cosmos_transfer", "__init__.py")) as f:
            source = f.read()

        control_types = ["depth", "edge", "seg", "blur"]
        for ct in control_types:
            assert ct in source, f"Missing control type: {ct}"

    def test_supports_multi_gpu(self):
        """Cosmos Transfer should support multi-GPU via torchrun."""
        with open(os.path.join(BASE, "cosmos_transfer", "__init__.py")) as f:
            source = f.read()

        assert "torchrun" in source or "multi_gpu" in source or "world_size" in source, "Missing multi-GPU support"

    def test_supports_autoregressive(self):
        """Cosmos Transfer should support autoregressive chunking for long videos."""
        with open(os.path.join(BASE, "cosmos_transfer", "__init__.py")) as f:
            source = f.read()

        assert (
            "autoregressive" in source.lower() or "chunk" in source.lower()
        ), "Missing autoregressive/chunking support"


class TestCosmosTransferDreamGenIntegration:
    """Test Cosmos Transfer is properly integrated into DreamGen pipeline."""

    def test_dreamgen_has_cosmos_transfer_model(self):
        """DreamGen config should accept 'cosmos_transfer' as video_model."""
        with open(os.path.join(BASE, "dreamgen", "__init__.py")) as f:
            source = f.read()

        assert "cosmos_transfer" in source, "DreamGen missing cosmos_transfer model option"

    def test_dreamgen_stage1_skip(self):
        """Stage 1 (fine-tuning) should be skipped for cosmos_transfer."""
        with open(os.path.join(BASE, "dreamgen", "__init__.py")) as f:
            source = f.read()

        assert "SKIPPING Stage 1" in source, "DreamGen missing Stage 1 skip for cosmos_transfer"
        assert '"status": "skipped"' in source, "DreamGen not returning skipped status"

    def test_dreamgen_stage2_cosmos_route(self):
        """Stage 2 (generation) should route to Cosmos Transfer pipeline."""
        with open(os.path.join(BASE, "dreamgen", "__init__.py")) as f:
            source = f.read()

        assert "_generate_via_cosmos_transfer" in source, "DreamGen missing _generate_via_cosmos_transfer method"
        assert "CosmosTransferPipeline" in source, "DreamGen Stage 2 not importing CosmosTransferPipeline"
        assert "CosmosTransferConfig" in source, "DreamGen Stage 2 not importing CosmosTransferConfig"

    def test_dreamgen_cosmos_transfer_calls_pipeline(self):
        """The cosmos transfer route should actually call pipeline.transfer_video()."""
        with open(os.path.join(BASE, "dreamgen", "__init__.py")) as f:
            source = f.read()

        assert (
            "pipeline.transfer_video" in source or "transfer_video(" in source
        ), "DreamGen cosmos route not calling transfer_video()"


class TestTopLevelExports:
    """Test Cosmos Transfer is exported from top-level package."""

    def test_init_has_cosmos_exports(self):
        with open(os.path.join(BASE, "__init__.py")) as f:
            source = f.read()

        assert "CosmosTransferPipeline" in source
        assert "CosmosTransferConfig" in source
