#!/usr/bin/env python3
"""Tests for Newton Physics backend integration."""

import ast
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE = os.path.join(os.path.dirname(__file__), "..", "strands_robots")


class TestNewtonSyntax:
    """Validate Newton module files parse correctly."""

    def _check_syntax(self, filepath):
        with open(filepath, "r") as f:
            source = f.read()
        ast.parse(source, filename=filepath)

    def test_newton_init_syntax(self):
        self._check_syntax(os.path.join(BASE, "newton", "__init__.py"))

    def test_newton_backend_syntax(self):
        self._check_syntax(os.path.join(BASE, "newton", "newton_backend.py"))


class TestNewtonModuleStructure:
    """Test Newton module exports and structure."""

    def test_import_newton_module(self):
        from strands_robots.newton import __all__

        assert "NewtonBackend" in __all__
        assert "NewtonConfig" in __all__
        assert "NewtonGymEnv" in __all__

    def test_newton_lazy_import_no_gpu(self):
        """Newton module should import without GPU/newton installed."""
        from strands_robots import newton

        assert hasattr(newton, "__all__")
        assert len(newton.__all__) == 6

    def test_newton_exports_match_all(self):
        from strands_robots.newton import __all__

        # Just verify __all__ entries are strings — actual import needs GPU deps
        for name in __all__:
            assert isinstance(name, str)
            assert len(name) > 0


class TestNewtonBackendContract:
    """Verify NewtonBackend has the same API contract as Simulation/IsaacSimBackend."""

    def test_backend_has_required_methods(self):
        """NewtonBackend must expose the same interface as other backends."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()
        tree = ast.parse(source)

        # Find the NewtonBackend class
        backend_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NewtonBackend":
                backend_class = node
                break

        assert backend_class is not None, "NewtonBackend class not found"

        # Check required method names (same as Simulation/IsaacSimBackend)
        method_names = {node.name for node in ast.walk(backend_class) if isinstance(node, ast.FunctionDef)}

        required_methods = {
            "create_world",
            "add_robot",
            "step",
            "render",
            "get_state",
            "destroy",
            "run_policy",
            "record_video",
        }

        missing = required_methods - method_names
        assert not missing, f"NewtonBackend missing required methods: {missing}"

    def test_backend_has_newton_specific_methods(self):
        """NewtonBackend should have parallel env + differentiable methods."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()
        tree = ast.parse(source)

        backend_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NewtonBackend":
                backend_class = node
                break

        method_names = {node.name for node in ast.walk(backend_class) if isinstance(node, ast.FunctionDef)}

        # Newton-specific capabilities
        newton_methods = {
            "replicate",  # Parallel env cloning
            "get_observation",  # Batched observation
            "reset",  # RL episode resets (FIX #8)
        }

        missing = newton_methods - method_names
        assert not missing, f"NewtonBackend missing Newton-specific methods: {missing}"

    def test_newton_config_dataclass(self):
        """NewtonConfig should have expected fields."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        # Check for key config fields
        assert "num_envs" in source, "NewtonConfig missing num_envs"
        assert "device" in source, "NewtonConfig missing device"
        assert "solver" in source, "NewtonConfig missing solver"

    def test_solver_names(self):
        """NewtonBackend should support known Newton solvers."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        # All 7 Newton solvers
        solvers = [
            "mujoco",
            "featherstone",
            "semi_implicit",
            "xpbd",
            "vbd",
            "style3d",
            "implicit_mpm",
        ]

        for solver in solvers:
            assert solver in source, f"Missing solver: {solver}"


class TestNewtonFactoryIntegration:
    """Test that factory.py correctly routes to Newton backend."""

    def test_factory_has_newton_route(self):
        """factory.py must have elif backend == 'newton' routing."""
        with open(os.path.join(BASE, "factory.py")) as f:
            source = f.read()

        assert (
            'backend == "newton"' in source or "backend == 'newton'" in source
        ), "factory.py missing Newton backend routing"
        assert "NewtonBackend" in source, "factory.py missing NewtonBackend import"
        assert "NewtonConfig" in source, "factory.py missing NewtonConfig import"

    def test_factory_newton_replicate(self):
        """factory.py should auto-replicate for num_envs > 1."""
        with open(os.path.join(BASE, "factory.py")) as f:
            source = f.read()

        assert "replicate" in source, "factory.py missing replicate() call for Newton"
        assert "num_envs" in source, "factory.py missing num_envs handling"

    def test_factory_docstring_has_newton_example(self):
        """Robot() docstring should include Newton usage example."""
        with open(os.path.join(BASE, "factory.py")) as f:
            source = f.read()

        assert 'backend="newton"' in source, "factory.py missing Newton example in docstring"


class TestTopLevelNewtonExports:
    """Test Newton is exported from top-level strands_robots package."""

    def test_init_has_newton_imports(self):
        """strands_robots/__init__.py should have Newton lazy imports."""
        with open(os.path.join(BASE, "__init__.py")) as f:
            source = f.read()

        assert "NewtonBackend" in source, "__init__.py missing NewtonBackend"
        assert "CosmosTransferPipeline" in source, "__init__.py missing CosmosTransferPipeline"


class TestRecordVideoCosmosHook:
    """Test that record_video supports cosmos_transfer flag across backends."""

    def test_simulation_record_video_has_cosmos_params(self):
        """Simulation.record_video should accept cosmos_transfer, cosmos_prompt, cosmos_control."""
        with open(os.path.join(BASE, "simulation.py")) as f:
            source = f.read()
        tree = ast.parse(source)

        # Find record_video method in the Simulation class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Simulation":
                for item in ast.walk(node):
                    if isinstance(item, ast.FunctionDef) and item.name == "record_video":
                        arg_names = [a.arg for a in item.args.args]
                        assert "cosmos_transfer" in arg_names, "Missing cosmos_transfer param"
                        assert "cosmos_prompt" in arg_names, "Missing cosmos_prompt param"
                        assert "cosmos_control" in arg_names, "Missing cosmos_control param"
                        return
        assert False, "record_video not found in Simulation class"

    def test_newton_record_video_has_cosmos_params(self):
        """NewtonBackend.record_video should accept cosmos_transfer, cosmos_prompt, cosmos_control."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NewtonBackend":
                for item in ast.walk(node):
                    if isinstance(item, ast.FunctionDef) and item.name == "record_video":
                        arg_names = [a.arg for a in item.args.args]
                        assert "cosmos_transfer" in arg_names, "Missing cosmos_transfer param"
                        assert "cosmos_prompt" in arg_names, "Missing cosmos_prompt param"
                        assert "cosmos_control" in arg_names, "Missing cosmos_control param"
                        return
        assert False, "record_video not found in NewtonBackend class"

    def test_simulation_cosmos_imports_in_record_video(self):
        """Simulation.record_video should import CosmosTransferPipeline when cosmos_transfer=True."""
        with open(os.path.join(BASE, "simulation.py")) as f:
            source = f.read()
        assert "CosmosTransferPipeline" in source, "simulation.py missing CosmosTransferPipeline import"
        assert "CosmosTransferConfig" in source, "simulation.py missing CosmosTransferConfig import"


class TestNewtonRecordVideoBugFixes:
    """Regression tests for Newton record_video bugs found by QA agent."""

    def test_record_video_uses_physics_dt_not_dt(self):
        """Bug fix: record_video must reference self._config.physics_dt, not self._config.dt.

        NewtonConfig has 'physics_dt' attribute, not 'dt'. Using 'dt' causes
        AttributeError at runtime when recording video.
        """
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        # Extract the record_video method body
        import re

        match = re.search(r"def record_video\(.*?\n(.*?)(?=\n    def |\nclass |\Z)", source, re.DOTALL)
        assert match, "record_video method not found"
        method_body = match.group(1)

        # Must NOT contain self._config.dt (only self._config.physics_dt)
        assert "self._config.dt" not in method_body.replace(
            "self._config.physics_dt", ""
        ), "record_video still uses self._config.dt instead of self._config.physics_dt"
        assert "self._config.physics_dt" in method_body, "record_video should use self._config.physics_dt"

    def test_record_video_passes_none_not_dict_to_step(self):
        """Bug fix: record_video must call self.step(None), not self.step({}).

        The step() method expects None or an array for actions. Passing {}
        (truthy empty dict) would fail in _apply_actions().
        """
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def record_video\(.*?\n(.*?)(?=\n    def |\nclass |\Z)", source, re.DOTALL)
        assert match, "record_video method not found"
        method_body = match.group(1)

        # Must NOT pass empty dict to step
        assert "self.step({})" not in method_body, "record_video must call self.step(None), not self.step({})"
        assert (
            "self.step(None)" in method_body or "self.step()" in method_body
        ), "record_video should call self.step(None) or self.step()"

    def test_record_video_checks_success_key_not_status(self):
        """Bug fix: render() returns {"image": ndarray} — record_video must
        check the image key, not a "status" key, to detect successful renders.
        """
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def record_video\(.*?\n(.*?)(?=\n    def |\nclass |\Z)", source, re.DOTALL)
        assert match, "record_video method not found"
        method_body = match.group(1)

        # Must check render result via .get("image"), not .get("status")
        assert '.get("image")' in method_body, "record_video should check render result via .get('image')"

    def test_record_video_uses_image_key_not_frame(self):
        """Bug fix: render() returns {"image": ndarray}, not {"frame": ndarray}.

        record_video must read render_result["image"] not render_result["frame"].
        """
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def record_video\(.*?\n(.*?)(?=\n    def |\nclass |\Z)", source, re.DOTALL)
        assert match, "record_video method not found"
        method_body = match.group(1)

        # Must use "image" key not "frame"
        assert (
            '"frame"' not in method_body
        ), "record_video should use render_result['image'], not render_result['frame']"
        assert '"image"' in method_body, "record_video should reference 'image' key from render result"

    def test_newton_backend_imports_os_and_tempfile(self):
        """Bug fix: newton_backend.py must import os and tempfile.

        record_video uses os.path.join, os.makedirs, os.path.getsize,
        tempfile.gettempdir() but these weren't imported.
        """
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        # Check top-level imports
        import re

        top_imports = re.findall(r"^import (\w+)", source, re.MULTILINE)
        assert "os" in top_imports, "newton_backend.py must import os at module level"
        assert "tempfile" in top_imports, "newton_backend.py must import tempfile at module level"

    def test_newton_config_has_physics_dt_not_dt(self):
        """Verify NewtonConfig uses 'physics_dt' as the attribute name."""
        from strands_robots.newton.newton_backend import NewtonConfig

        config = NewtonConfig()
        assert hasattr(config, "physics_dt"), "NewtonConfig must have physics_dt"
        assert not hasattr(config, "dt"), "NewtonConfig should NOT have a 'dt' attribute (use physics_dt)"

    def test_render_returns_success_and_image_keys(self):
        """Verify render() method returns dict with 'success' and 'image' keys."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()
        tree = ast.parse(source)

        # Find the render method in NewtonBackend class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NewtonBackend":
                for item in ast.walk(node):
                    if isinstance(node, ast.ClassDef):
                        for child in node.body:
                            if isinstance(child, ast.FunctionDef) and child.name == "render":
                                # Get the source of the render method
                                render_lines = ast.get_source_segment(source, child)
                                assert render_lines is not None, "Could not extract render source"
                                assert '"success"' in render_lines, "render must return dict with 'success' key"
                                assert '"image"' in render_lines, "render must return dict with 'image' key"
                                return
        pytest.fail("render method not found in NewtonBackend class")

    def test_newton_cosmos_imports_in_record_video(self):
        """NewtonBackend.record_video should import CosmosTransferPipeline when cosmos_transfer=True."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()
        assert "CosmosTransferPipeline" in source, "newton_backend.py missing CosmosTransferPipeline import"
        assert "CosmosTransferConfig" in source, "newton_backend.py missing CosmosTransferConfig import"


class TestNewtonCriticalGapFixes:
    """Tests for the 6 critical API gap fixes identified in issue #29 analysis.

    These verify that the Newton backend correctly implements the upstream
    Newton simulation loop pattern:
        state.clear_forces()
        model.collide(state, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
    """

    def test_contacts_allocated_in_finalize(self):
        """FIX #1: _finalize_model() must allocate contacts via model.contacts()."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def _finalize_model\(.*?\n(.*?)(?=\n    # ---|\n    def )", source, re.DOTALL)
        assert match, "_finalize_model method not found"
        method_body = match.group(1)

        assert "self._contacts" in method_body, "_finalize_model must allocate self._contacts"
        assert (
            "model.contacts()" in method_body
            or "_model.contacts()" in method_body
            or "_collision_pipeline.contacts()" in method_body
        ), "_finalize_model must allocate contacts (via model or collision pipeline)"

    def test_solver_step_passes_contacts_not_none(self):
        """FIX #1: _solver_step() must pass self._contacts, NOT None."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def _solver_step\(.*?\n(.*?)(?=\n    def )", source, re.DOTALL)
        assert match, "_solver_step method not found"
        method_body = match.group(1)

        # Must pass self._contacts, not None
        assert "self._contacts" in method_body, "_solver_step must pass self._contacts to solver.step()"
        # The old pattern "None,   # contacts" should NOT be present
        lines = method_body.strip().split("\n")
        solver_step_lines = [line for line in lines if "self._solver.step(" in line or "self._contacts" in line]
        assert any(
            "self._contacts" in line for line in solver_step_lines
        ), "_solver_step must pass self._contacts (not None) to solver.step()"

    def test_solver_step_calls_clear_forces(self):
        """FIX #2: _solver_step() must call state_0.clear_forces() before stepping."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def _solver_step\(.*?\n(.*?)(?=\n    def )", source, re.DOTALL)
        assert match, "_solver_step method not found"
        method_body = match.group(1)

        assert "clear_forces" in method_body, "_solver_step must call clear_forces() to prevent force accumulation"

    def test_solver_step_calls_model_collide(self):
        """FIX #1+: _solver_step() must call model.collide() for collision detection."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def _solver_step\(.*?\n(.*?)(?=\n    def )", source, re.DOTALL)
        assert match, "_solver_step method not found"
        method_body = match.group(1)

        assert "collide" in method_body, "_solver_step must call model.collide() for collision detection"

    def test_finalize_calls_eval_fk(self):
        """FIX #3: _finalize_model() must call eval_fk() to init body transforms."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def _finalize_model\(.*?\n(.*?)(?=\n    # ---|\n    def )", source, re.DOTALL)
        assert match, "_finalize_model method not found"
        method_body = match.group(1)

        assert "eval_fk" in method_body, "_finalize_model must call eval_fk() to initialise body transforms"

    def test_finalize_registers_custom_attributes(self):
        """FIX #4: _finalize_model() must call register_custom_attributes() BEFORE finalize."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def _finalize_model\(.*?\n(.*?)(?=\n    # ---|\n    def )", source, re.DOTALL)
        assert match, "_finalize_model method not found"
        method_body = match.group(1)

        assert (
            "register_custom_attributes" in method_body
        ), "_finalize_model must call register_custom_attributes() before finalize()"

        # Verify order: register_custom_attributes BEFORE builder.finalize()
        reg_pos = method_body.find("register_custom_attributes")
        fin_pos = method_body.find("self._builder.finalize")
        assert reg_pos < fin_pos, "register_custom_attributes must be called BEFORE builder.finalize()"

    def test_replicate_ground_after_robots(self):
        """FIX #5: replicate() must add ground plane AFTER replicate(), not before."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def replicate\(.*?\n(.*?)(?=\n    # ---|\n    def )", source, re.DOTALL)
        assert match, "replicate method not found"
        method_body = match.group(1)

        # Find the primary replicate path (not the fallback)
        # Ground plane should come AFTER main_builder.replicate(), not before
        replicate_pos = method_body.find("main_builder.replicate(")
        ground_pos = method_body.find("add_ground_plane", replicate_pos if replicate_pos >= 0 else 0)

        if replicate_pos >= 0 and ground_pos >= 0:
            assert ground_pos > replicate_pos, "Ground plane must be added AFTER replicate(), not before"

    def test_backend_has_contacts_attribute(self):
        """FIX #1: NewtonBackend.__init__() must initialise self._contacts."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def __init__\(.*?\n(.*?)(?=\n    # ---|\n    def )", source, re.DOTALL)
        assert match, "__init__ method not found"
        init_body = match.group(1)

        assert "self._contacts" in init_body, "NewtonBackend.__init__() must initialise self._contacts"

    def test_backend_has_reset_method(self):
        """FIX #8: NewtonBackend must have a reset() method for RL episode resets."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()
        tree = ast.parse(source)

        backend_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NewtonBackend":
                backend_class = node
                break

        assert backend_class is not None, "NewtonBackend class not found"

        method_names = {node.name for node in ast.walk(backend_class) if isinstance(node, ast.FunctionDef)}

        assert "reset" in method_names, "NewtonBackend must have a reset() method"

    def test_default_joint_state_stored(self):
        """FIX #8: _finalize_model() must store default joint state for resets."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def _finalize_model\(.*?\n(.*?)(?=\n    # ---|\n    def )", source, re.DOTALL)
        assert match, "_finalize_model method not found"
        method_body = match.group(1)

        assert "self._default_joint_q" in method_body, "_finalize_model must store default joint positions for resets"
        assert "self._default_joint_qd" in method_body, "_finalize_model must store default joint velocities for resets"

    def test_destroy_cleans_up_contacts(self):
        """destroy() must clean up the contacts buffer."""
        with open(os.path.join(BASE, "newton", "newton_backend.py")) as f:
            source = f.read()

        import re

        match = re.search(r"def destroy\(.*?\n(.*?)(?=\n    # ---|\n    def )", source, re.DOTALL)
        assert match, "destroy method not found"
        method_body = match.group(1)

        # Contacts may be cleaned via setattr loop or direct assignment
        assert "_contacts" in method_body, "destroy() must clean up _contacts (directly or via setattr loop)"
