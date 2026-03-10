#!/usr/bin/env python3
"""Mock-based tests for strands_robots.tools.use_lerobot.

These tests DO NOT require lerobot to be installed — they mock the entire
lerobot module hierarchy to exercise all code paths in use_lerobot.py,
including _import_from_lerobot, _discover_modules, and the main use_lerobot
tool function (discovery mode, describe mode, call mode, error handling).

Targets: lines 66, 75-78, 92-164, 183-184, 196-197, 298-321, 328-329,
         340-341, 348, 354, 357-359, 370-371
"""

import importlib
import inspect
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

# Pre-mock strands @tool decorator so the module can import cleanly
_mock_strands = types.ModuleType("strands")
_mock_strands.tool = lambda f: f
sys.modules.setdefault("strands", _mock_strands)

# Pre-mock cv2 to prevent OpenCV 4.12 cv2.dnn.DictValue crash
import importlib.machinery as _im_fix  # noqa: E402

_mock_cv2 = MagicMock()
_mock_cv2.__spec__ = _im_fix.ModuleSpec("cv2", None)
_mock_cv2.dnn = MagicMock()
sys.modules.setdefault("cv2", _mock_cv2)
sys.modules.setdefault("cv2.dnn", _mock_cv2.dnn)

import strands_robots.tools.use_lerobot as _use_lerobot_mod  # noqa: E402
from strands_robots.tools.use_lerobot import (  # noqa: E402
    _describe_object,
    _discover_modules,
    _import_from_lerobot,
    use_lerobot,
)

# ── Helper: Build a fake lerobot module hierarchy ─────────────


@contextmanager
def fake_lerobot():
    """Create a full fake lerobot module hierarchy for testing.

    Provides:
    - lerobot (top-level with __path__)
    - lerobot.policies (package)
    - lerobot.policies.factory (module with get_policy_class)
    - lerobot.datasets (package)
    - lerobot.robots (package with config submodule)
    - lerobot.robots.config (with RobotConfig class)
    - lerobot.robots.so100 (with SO100Config class)
    - lerobot.teleoperators (package with config submodule)
    - lerobot.teleoperators.config (with TeleoperatorConfig class)
    - lerobot.teleoperators.keyboard (with KeyboardConfig class)
    - lerobot.utils.constants (with HF_LEROBOT_CALIBRATION)
    """
    import os
    import tempfile

    # Create a temp directory to serve as __path__ for iter_modules
    tmpdir = tempfile.mkdtemp()

    # Top-level lerobot module
    lerobot = types.ModuleType("lerobot")
    lerobot.__path__ = [tmpdir]
    lerobot.__file__ = os.path.join(tmpdir, "__init__.py")

    # --- Create top-level items for iter_modules discovery ---
    # Package dirs (ispkg=True, line 99)
    for d in ["policies", "datasets", "robots", "teleoperators", "utils"]:
        p = os.path.join(tmpdir, d)
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "__init__.py"), "w") as f:
            f.write("")

    # Private package to test line 97 (continue on _prefixed)
    priv = os.path.join(tmpdir, "_internal")
    os.makedirs(priv, exist_ok=True)
    with open(os.path.join(priv, "__init__.py"), "w") as f:
        f.write("")

    # 'tests' package to test line 97 (continue on 'tests')
    tests_dir = os.path.join(tmpdir, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    with open(os.path.join(tests_dir, "__init__.py"), "w") as f:
        f.write("")

    # Plain module (ispkg=False, line 101)
    with open(os.path.join(tmpdir, "common_utils.py"), "w") as f:
        f.write("")

    # --- policies package ---
    policies = types.ModuleType("lerobot.policies")
    policies.__path__ = [os.path.join(tmpdir, "policies")]

    factory = types.ModuleType("lerobot.policies.factory")

    def get_policy_class(name: str) -> type:
        """Get policy class by name."""
        return type(f"{name.upper()}Policy", (), {"name": name})

    factory.get_policy_class = get_policy_class

    # --- datasets package ---
    datasets = types.ModuleType("lerobot.datasets")
    datasets.__path__ = [os.path.join(tmpdir, "datasets")]

    lerobot_dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")

    class LeRobotDataset:
        """Mock LeRobot dataset."""

        def __init__(self, repo_id: str = ""):
            self.repo_id = repo_id

        @classmethod
        def create(cls, repo_id: str = "", fps: int = 30, **kwargs):
            """Create a new dataset."""
            return cls(repo_id=repo_id)

        def push_to_hub(self):
            pass

    lerobot_dataset_mod.LeRobotDataset = LeRobotDataset

    # --- robots package ---
    robots = types.ModuleType("lerobot.robots")
    robots_dir = os.path.join(tmpdir, "robots")
    robots.__path__ = [robots_dir]

    # Create subdirs that iter_modules will find
    # 'config' dir — should be skipped by line 131
    for skip_name in ["config", "robot", "utils"]:
        skip_dir = os.path.join(robots_dir, skip_name)
        os.makedirs(skip_dir, exist_ok=True)
        with open(os.path.join(skip_dir, "__init__.py"), "w") as f:
            f.write("")

    # 'so100' dir — should be iterated
    so100_dir = os.path.join(robots_dir, "so100")
    os.makedirs(so100_dir, exist_ok=True)
    with open(os.path.join(so100_dir, "__init__.py"), "w") as f:
        f.write("")

    robots_config = types.ModuleType("lerobot.robots.config")

    class RobotConfig:
        pass

    robots_config.RobotConfig = RobotConfig

    robots_so100 = types.ModuleType("lerobot.robots.so100")

    class SO100Config(RobotConfig):
        pass

    robots_so100.SO100Config = SO100Config
    robots_so100.NotAConfig = "just a string"  # Should be filtered out

    # --- teleoperators package ---
    teleoperators = types.ModuleType("lerobot.teleoperators")
    teleop_dir = os.path.join(tmpdir, "teleoperators")
    teleoperators.__path__ = [teleop_dir]

    # Skip dirs for teleoperators (line 151)
    for skip_name in ["config", "teleoperator", "utils"]:
        skip_dir = os.path.join(teleop_dir, skip_name)
        os.makedirs(skip_dir, exist_ok=True)
        with open(os.path.join(skip_dir, "__init__.py"), "w") as f:
            f.write("")

    # Keyboard dir — should be iterated
    kbd_dir = os.path.join(teleop_dir, "keyboard")
    os.makedirs(kbd_dir, exist_ok=True)
    with open(os.path.join(kbd_dir, "__init__.py"), "w") as f:
        f.write("")

    teleop_config = types.ModuleType("lerobot.teleoperators.config")

    class TeleoperatorConfig:
        pass

    teleop_config.TeleoperatorConfig = TeleoperatorConfig

    teleop_keyboard = types.ModuleType("lerobot.teleoperators.keyboard")

    class KeyboardConfig(TeleoperatorConfig):
        pass

    teleop_keyboard.KeyboardConfig = KeyboardConfig

    # --- utils package ---
    utils = types.ModuleType("lerobot.utils")
    utils.__path__ = [os.path.join(tmpdir, "utils")]

    constants = types.ModuleType("lerobot.utils.constants")
    constants.HF_LEROBOT_CALIBRATION = "/path/to/calibration"

    # Register all in sys.modules
    mods = {
        "lerobot": lerobot,
        "lerobot.policies": policies,
        "lerobot.policies.factory": factory,
        "lerobot.datasets": datasets,
        "lerobot.datasets.lerobot_dataset": lerobot_dataset_mod,
        "lerobot.robots": robots,
        "lerobot.robots.config": robots_config,
        "lerobot.robots.so100": robots_so100,
        "lerobot.teleoperators": teleoperators,
        "lerobot.teleoperators.config": teleop_config,
        "lerobot.teleoperators.keyboard": teleop_keyboard,
        "lerobot.utils": utils,
        "lerobot.utils.constants": constants,
    }

    originals = {}
    for name, mod in mods.items():
        originals[name] = sys.modules.get(name)
        sys.modules[name] = mod

    try:
        yield lerobot
    finally:
        for name in mods:
            if originals[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = originals[name]
        # Clean up tmpdir
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# _import_from_lerobot — Mock-based tests
# ============================================================================


class TestImportFromLerobotMocked:
    """Test _import_from_lerobot using fake lerobot modules."""

    def test_import_module_directly(self):
        """Import a module (first try succeeds)."""
        with fake_lerobot():
            mod = _import_from_lerobot("policies.factory")
            assert hasattr(mod, "get_policy_class")

    def test_import_with_lerobot_prefix(self):
        """If user passes 'lerobot.X', it still resolves."""
        with fake_lerobot():
            mod = _import_from_lerobot("lerobot.policies.factory")
            assert hasattr(mod, "get_policy_class")

    def test_import_class_via_attribute(self):
        """Import a class: module import fails, falls back to getattr (line 66)."""
        with fake_lerobot():
            cls = _import_from_lerobot("datasets.lerobot_dataset.LeRobotDataset")
            assert inspect.isclass(cls)

    def test_import_function_via_attribute(self):
        """Import a function from a module."""
        with fake_lerobot():
            func = _import_from_lerobot("policies.factory.get_policy_class")
            assert callable(func)

    def test_import_constant_value(self):
        """Import a constant value (non-callable attribute)."""
        with fake_lerobot():
            val = _import_from_lerobot("utils.constants.HF_LEROBOT_CALIBRATION")
            assert val == "/path/to/calibration"

    def test_import_deeply_nested_attribute(self):
        """Import a deeply nested attribute (module.Class.method via multi-level split)."""
        with fake_lerobot():
            name = _import_from_lerobot("policies.factory.get_policy_class.__name__")
            assert name == "get_policy_class"

    def test_import_fallback_to_segment_loop(self):
        """Test the segment loop when first two strategies fail."""
        with fake_lerobot():
            method = _import_from_lerobot("datasets.lerobot_dataset.LeRobotDataset.create")
            assert callable(method)

    def test_import_nonexistent_raises(self):
        """Non-existent paths raise ImportError."""
        with fake_lerobot():
            with pytest.raises(ImportError, match="Cannot resolve"):
                _import_from_lerobot("nonexistent.module.Nope")

    def test_import_single_segment(self):
        """Single-segment import ('policies')."""
        with fake_lerobot():
            mod = _import_from_lerobot("policies")
            assert mod is not None


# ============================================================================
# _discover_modules — Mock-based tests (lines 92-164)
# ============================================================================


class TestDiscoverModulesMocked:
    """Test _discover_modules with fake lerobot — covers lines 92-164."""

    def test_discover_finds_packages_and_modules(self):
        """Discovery walks lerobot.__path__ and finds packages and plain modules."""
        with fake_lerobot():
            result = _discover_modules()

            assert "error" not in result
            assert "packages" in result
            assert "modules" in result
            assert "key_apis" in result
            assert isinstance(result["packages"], list)
            assert isinstance(result["modules"], list)
            # Verify packages found (line 99)
            assert len(result["packages"]) > 0
            # Verify plain modules found (line 101)
            assert "common_utils" in result["modules"]

    def test_discover_key_apis(self):
        """Key APIs dict is populated."""
        with fake_lerobot():
            result = _discover_modules()
            assert len(result["key_apis"]) > 0
            assert "cameras.opencv.OpenCVCamera" in result["key_apis"]

    def test_discover_robot_configs(self):
        """Robot configs are discovered — covers lines 128-142."""
        with fake_lerobot():
            result = _discover_modules()
            assert "robot_configs" in result
            assert "robots.so100.SO100Config" in result["robot_configs"]

    def test_discover_teleop_configs(self):
        """Teleoperator configs are discovered — covers lines 148-162."""
        with fake_lerobot():
            result = _discover_modules()
            assert "teleop_configs" in result
            assert "teleoperators.keyboard.KeyboardConfig" in result["teleop_configs"]

    def test_discover_skips_private_and_tests(self):
        """Modules starting with _ or named 'tests' are skipped (line 97)."""
        with fake_lerobot():
            result = _discover_modules()
            all_names = result["packages"] + result["modules"]
            assert "_internal" not in all_names
            assert "tests" not in all_names

    def test_discover_no_lerobot(self):
        """When lerobot is not installed, returns error."""
        with patch.dict("sys.modules", {"lerobot": None}):
            result = _discover_modules()
            assert "error" in result
            assert "not installed" in result["error"]

    def test_discover_robots_exception(self):
        """Robot discovery gracefully handles exceptions."""
        with fake_lerobot():
            with patch.dict("sys.modules", {"lerobot.robots": None}):
                result = _discover_modules()
                assert "error" not in result

    def test_discover_teleoperators_exception(self):
        """Teleoperator discovery gracefully handles exceptions."""
        with fake_lerobot():
            with patch.dict("sys.modules", {"lerobot.teleoperators": None}):
                result = _discover_modules()
                assert "error" not in result

    def test_discover_robot_module_import_failure(self):
        """Individual robot submodule import failure is caught (lines 138-139).

        Patches importlib.import_module to raise for lerobot.robots.so100,
        exercising the except Exception: continue path.
        """
        with fake_lerobot():
            original_import = importlib.import_module

            def failing_robot_import(name, package=None):
                if name == "lerobot.robots.so100":
                    raise RuntimeError("Simulated import failure for so100")
                return original_import(name, package)

            with patch(
                "strands_robots.tools.use_lerobot.importlib.import_module",
                side_effect=failing_robot_import,
            ):
                result = _discover_modules()
                assert "error" not in result
                # robot_configs should exist but NOT contain so100
                assert "robot_configs" in result
                assert not any("so100" in rc for rc in result["robot_configs"])

    def test_discover_teleop_module_import_failure(self):
        """Individual teleop submodule import failure is caught (lines 158-159).

        Patches importlib.import_module to raise for lerobot.teleoperators.keyboard.
        """
        with fake_lerobot():
            original_import = importlib.import_module

            def failing_teleop_import(name, package=None):
                if name == "lerobot.teleoperators.keyboard":
                    raise RuntimeError("Simulated import failure for keyboard")
                return original_import(name, package)

            with patch(
                "strands_robots.tools.use_lerobot.importlib.import_module",
                side_effect=failing_teleop_import,
            ):
                result = _discover_modules()
                assert "error" not in result
                # teleop_configs should exist but NOT contain keyboard
                assert "teleop_configs" in result
                assert not any("keyboard" in tc for tc in result["teleop_configs"])


# ============================================================================
# _describe_object — Additional mock coverage (lines 183-184, 196-197)
# ============================================================================


class TestDescribeObjectAdditional:
    """Additional tests for describe paths that need mocking."""

    def test_describe_class_init_no_doc(self):
        """Class whose __init__.__doc__ is falsy."""

        class NoDocInit:
            def __init__(self):
                pass

        NoDocInit.__init__.__doc__ = None
        info = _describe_object(NoDocInit)
        assert "init_doc" not in info

    def test_describe_class_init_signature_raises_value_error(self):
        """Class whose __init__ raises ValueError on signature (lines 183-184)."""

        class BadSigClass:
            def __init__(self):
                """Some doc."""
                pass

        # Mock inspect.signature to raise ValueError for this class's __init__
        original_signature = inspect.signature

        def mock_signature(obj, **kwargs):
            if obj is BadSigClass.__init__:
                raise ValueError("no signature found")
            return original_signature(obj, **kwargs)

        with patch("strands_robots.tools.use_lerobot.inspect.signature", side_effect=mock_signature):
            info = _describe_object(BadSigClass)
            assert info["type"] == "type"
            # init_params should NOT be present because signature raised
            assert "init_params" not in info

    def test_describe_class_init_signature_raises_type_error(self):
        """Class whose __init__ raises TypeError on signature (lines 183-184)."""

        class BadSigClass2:
            def __init__(self):
                """Some doc."""
                pass

        original_signature = inspect.signature

        def mock_signature(obj, **kwargs):
            if obj is BadSigClass2.__init__:
                raise TypeError("not supported")
            return original_signature(obj, **kwargs)

        with patch("strands_robots.tools.use_lerobot.inspect.signature", side_effect=mock_signature):
            info = _describe_object(BadSigClass2)
            assert "init_params" not in info

    def test_describe_callable_signature_raises_value_error(self):
        """Callable whose signature raises ValueError (lines 196-197)."""

        def bad_callable(x):
            """Has doc."""
            pass

        original_signature = inspect.signature

        def mock_signature(obj, **kwargs):
            if obj is bad_callable:
                raise ValueError("no signature")
            return original_signature(obj, **kwargs)

        with patch("strands_robots.tools.use_lerobot.inspect.signature", side_effect=mock_signature):
            info = _describe_object(bad_callable)
            assert info["type"] == "function"
            # params should NOT be present
            assert "params" not in info
            # doc should still be present
            assert "doc" in info

    def test_describe_callable_signature_raises_type_error(self):
        """Callable whose signature raises TypeError (lines 196-197)."""

        def bad_callable2(x):
            pass

        bad_callable2.__doc__ = None

        original_signature = inspect.signature

        def mock_signature(obj, **kwargs):
            if obj is bad_callable2:
                raise TypeError("unsupported callable")
            return original_signature(obj, **kwargs)

        with patch("strands_robots.tools.use_lerobot.inspect.signature", side_effect=mock_signature):
            info = _describe_object(bad_callable2)
            assert "params" not in info
            assert "doc" not in info

    def test_describe_callable_no_doc(self):
        """Callable without a docstring."""

        def no_doc_func(x):
            pass

        no_doc_func.__doc__ = None
        info = _describe_object(no_doc_func)
        assert "doc" not in info

    def test_describe_callable_with_annotation(self):
        """Callable with type annotations."""

        def typed_func(a: int, b: str = "hello") -> bool:
            """Typed function."""
            pass

        info = _describe_object(typed_func)
        assert info["params"]["a"]["annotation"] is not None
        assert info["params"]["b"]["annotation"] is not None

    def test_describe_callable_no_annotation(self):
        """Callable without type annotations."""

        def untyped_func(a, b="hi"):
            pass

        info = _describe_object(untyped_func)
        assert info["params"]["a"]["annotation"] is None

    def test_describe_module_with_doc(self):
        """Module with docstring."""
        mod = types.ModuleType("test_mod")
        mod.__doc__ = "This is a test module."
        mod.public_attr = 42

        info = _describe_object(mod)
        assert info["type"] == "module"
        assert "public_names" in info
        assert "doc" in info
        assert "test module" in info["doc"]

    def test_describe_module_without_doc(self):
        """Module without docstring."""
        mod = types.ModuleType("test_mod_nodoc")
        mod.__doc__ = None

        info = _describe_object(mod)
        assert info["type"] == "module"
        assert "doc" not in info


# ============================================================================
# use_lerobot — Discovery mode with mocked lerobot (lines 298-321)
# ============================================================================


class TestUseLerobotDiscoveryMocked:
    """Test use_lerobot discovery mode with fake lerobot."""

    def test_discovery_success(self):
        """Full discovery returns packages, modules, key APIs."""
        with fake_lerobot():
            result = use_lerobot(module="__discovery__", method="list_modules")
            assert result["status"] == "success"
            text = result["content"][0]["text"]
            assert "LeRobot API Discovery" in text
            assert "Packages:" in text
            assert "Key APIs:" in text
            assert "Usage:" in text

    def test_discovery_shows_robot_configs(self):
        """Discovery output includes robot configs."""
        with fake_lerobot():
            result = use_lerobot(module="__discovery__")
            text = result["content"][0]["text"]
            assert "Robot Configs" in text
            assert "SO100Config" in text

    def test_discovery_shows_teleop_configs(self):
        """Discovery output includes teleop configs."""
        with fake_lerobot():
            result = use_lerobot(module="__discovery__")
            text = result["content"][0]["text"]
            assert "Teleop Configs" in text
            assert "KeyboardConfig" in text

    def test_discovery_no_lerobot_error(self):
        """Discovery with no lerobot returns error."""
        with patch.dict("sys.modules", {"lerobot": None}):
            result = use_lerobot(module="__discovery__")
            assert result["status"] == "error"
            assert "not installed" in result["content"][0]["text"]


# ============================================================================
# use_lerobot — Describe mode (lines 328-329)
# ============================================================================


class TestUseLerobotDescribeMocked:
    """Test use_lerobot describe mode with fake lerobot."""

    def test_describe_module(self):
        """Describe a module."""
        with fake_lerobot():
            result = use_lerobot(module="policies.factory", method="__describe__")
            assert result["status"] == "success"
            text = result["content"][0]["text"]
            assert "policies.factory" in text

    def test_describe_class(self):
        """Describe a class."""
        with fake_lerobot():
            result = use_lerobot(
                module="datasets.lerobot_dataset.LeRobotDataset",
                method="__describe__",
            )
            assert result["status"] == "success"

    def test_describe_function(self):
        """Describe a function."""
        with fake_lerobot():
            result = use_lerobot(
                module="policies.factory.get_policy_class",
                method="__describe__",
            )
            assert result["status"] == "success"


# ============================================================================
# use_lerobot — Method not found (lines 340-341)
# ============================================================================


class TestUseLerobotMethodNotFound:
    """Test method-not-found error path."""

    def test_method_not_found_shows_available(self):
        """Non-existent method returns error with available methods."""
        with fake_lerobot():
            result = use_lerobot(module="policies.factory", method="nonexistent_method_XYZ")
            assert result["status"] == "error"
            text = result["content"][0]["text"]
            assert "not found" in text
            assert "Available:" in text


# ============================================================================
# use_lerobot — Non-callable attribute (line 348)
# ============================================================================


class TestUseLerobotNonCallable:
    """Test accessing non-callable attributes."""

    def test_non_callable_returns_value(self):
        """Accessing a constant returns its value."""
        with fake_lerobot():
            result = use_lerobot(module="utils.constants", method="HF_LEROBOT_CALIBRATION")
            assert result["status"] == "success"
            text = result["content"][0]["text"]
            assert "=" in text
            assert "/path/to/calibration" in text


# ============================================================================
# use_lerobot — Successful call (lines 354, 357-359)
# ============================================================================


class TestUseLerobotCallMocked:
    """Test successful function calls with fake lerobot."""

    def test_call_function(self):
        """Call a function and get serialized result."""
        with fake_lerobot():
            result = use_lerobot(
                module="policies.factory",
                method="get_policy_class",
                parameters={"name": "act"},
            )
            assert result["status"] == "success"
            text = result["content"][0]["text"]
            assert "✅" in text
            assert "get_policy_class()" in text

    def test_call_with_label(self):
        """Call with a label (line 354)."""
        with fake_lerobot():
            result = use_lerobot(
                module="policies.factory",
                method="get_policy_class",
                parameters={"name": "diffusion"},
                label="Get diffusion policy",
            )
            assert result["status"] == "success"

    def test_call_no_params(self):
        """Call with no parameters (params=None → {})."""
        mock_module = MagicMock()
        mock_module.my_method = lambda: {"result": "ok"}
        mock_module.my_method.__name__ = "my_method"

        # Ensure module is in sys.modules for patch to work
        sys.modules.setdefault("strands_robots.tools.use_lerobot", _use_lerobot_mod)
        with patch.object(
            _use_lerobot_mod,
            "_import_from_lerobot",
            return_value=mock_module,
        ):
            result = use_lerobot(module="noargs_test", method="my_method", parameters=None)
            assert result["status"] == "success"

    def test_call_result_truncation(self):
        """Result is truncated to 3000 chars."""
        mock_module = MagicMock()
        mock_module.big_method = lambda: "x" * 5000
        mock_module.big_method.__name__ = "big_method"

        sys.modules.setdefault("strands_robots.tools.use_lerobot", _use_lerobot_mod)
        with patch.object(
            _use_lerobot_mod,
            "_import_from_lerobot",
            return_value=mock_module,
        ):
            result = use_lerobot(module="big_test", method="big_method")
            assert result["status"] == "success"


# ============================================================================
# use_lerobot — Error handling (lines 370-371)
# ============================================================================


class TestUseLerobotErrorsMocked:
    """Test error handling paths."""

    def test_import_error(self):
        """ImportError is caught and formatted."""
        result = use_lerobot(module="totally.fake.module", method="do_stuff")
        assert result["status"] == "error"
        assert "Import error" in result["content"][0]["text"]

    def test_type_error_with_signature(self):
        """TypeError shows expected signature."""
        with fake_lerobot():
            result = use_lerobot(
                module="policies.factory",
                method="get_policy_class",
                parameters={"wrong_param": "bad"},
            )
            assert result["status"] == "error"
            assert "TypeError" in result["content"][0]["text"]

    def test_type_error_without_signature(self):
        """TypeError when signature inspection also fails (line 370-371)."""
        import strands_robots.tools.use_lerobot as _mod

        orig = _mod._import_from_lerobot

        mock_target = MagicMock(side_effect=TypeError("bad call"))
        mock_target.__name__ = "mock_func"
        mock_module = MagicMock()
        mock_module.bad_method = mock_target

        _mod._import_from_lerobot = lambda p: mock_module
        try:
            with patch("inspect.signature", side_effect=ValueError("no sig")):
                result = _mod.use_lerobot(module="mock", method="bad_method")
                assert result["status"] == "error"
                assert "TypeError" in result["content"][0]["text"]
                assert "bad call" in result["content"][0]["text"]
        finally:
            _mod._import_from_lerobot = orig

    def test_generic_exception(self):
        """RuntimeError and other exceptions are caught."""
        import strands_robots.tools.use_lerobot as _mod

        orig = _mod._import_from_lerobot
        _mod._import_from_lerobot = lambda p: (_ for _ in ()).throw(RuntimeError("kaboom"))
        try:
            result = _mod.use_lerobot(module="anything", method="something")
            assert result["status"] == "error"
            assert "kaboom" in result["content"][0]["text"]
            assert "RuntimeError" in result["content"][0]["text"]
        finally:
            _mod._import_from_lerobot = orig

    def test_type_error_with_valid_signature(self):
        """TypeError where we CAN get the signature shows param info."""
        import strands_robots.tools.use_lerobot as _mod

        orig = _mod._import_from_lerobot

        def raising_func(**kwargs):
            raise TypeError("missing required argument: 'required_arg'")

        mock_module = MagicMock()
        mock_module.typed_method = raising_func

        _mod._import_from_lerobot = lambda p: mock_module
        try:
            result = _mod.use_lerobot(
                module="test_mod",
                method="typed_method",
                parameters={"wrong": "val"},
            )
            assert result["status"] == "error"
            assert "TypeError" in result["content"][0]["text"]
        finally:
            _mod._import_from_lerobot = orig


# ============================================================================
# Integration: Full workflow — discover → describe → call (all mocked)
# ============================================================================


class TestUseLerobotFullWorkflowMocked:
    """End-to-end workflow test with fake lerobot."""

    def test_full_workflow(self):
        """Discover → describe → call, all mocked."""
        with fake_lerobot():
            # Step 1: Discover
            disc = use_lerobot()
            assert disc["status"] == "success"
            assert "LeRobot API Discovery" in disc["content"][0]["text"]

            # Step 2: Describe the factory
            desc = use_lerobot(
                module="policies.factory.get_policy_class",
                method="__describe__",
            )
            assert desc["status"] == "success"

            # Step 3: Call it
            call = use_lerobot(
                module="policies.factory",
                method="get_policy_class",
                parameters={"name": "act"},
            )
            assert call["status"] == "success"

    def test_describe_then_access_constant(self):
        """Describe a constants module, then access a constant."""
        with fake_lerobot():
            desc = use_lerobot(module="utils.constants", method="__describe__")
            assert desc["status"] == "success"

            val = use_lerobot(module="utils.constants", method="HF_LEROBOT_CALIBRATION")
            assert val["status"] == "success"
            assert "/path/to/calibration" in val["content"][0]["text"]


# ============================================================================
# Edge cases and corner cases
# ============================================================================


class TestUseLerobotEdgeCases:
    """Edge cases for comprehensive coverage."""

    def test_empty_method_string(self):
        """Empty method string (method='')."""
        mock_func = MagicMock(return_value="result")
        mock_func.__name__ = "mock"

        sys.modules.setdefault("strands_robots.tools.use_lerobot", _use_lerobot_mod)
        with patch.object(
            _use_lerobot_mod,
            "_import_from_lerobot",
            return_value=mock_func,
        ):
            result = use_lerobot(module="callable_module", method="")
            assert result["status"] == "success"

    def test_discovery_no_robot_configs(self):
        """Discovery when robot/teleop discovery returns empty."""
        with fake_lerobot():
            empty_robots = types.ModuleType("lerobot.robots")
            empty_robots.__path__ = ["/nonexistent/path"]
            with patch.dict("sys.modules", {"lerobot.robots": empty_robots}):
                result = _discover_modules()
                assert "error" not in result

    def test_discovery_with_tests_module_skipped(self):
        """The 'tests' submodule should be skipped."""
        with fake_lerobot():
            result = _discover_modules()
            if "packages" in result:
                assert "tests" not in result["packages"]
            if "modules" in result:
                assert "tests" not in result["modules"]
