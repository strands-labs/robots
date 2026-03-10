#!/usr/bin/env python3
"""Tests for strands_robots.tools.use_lerobot — universal LeRobot access tool.

Tests that require a real lerobot installation are skipped when lerobot
is not available.  Error/edge cases use mocks and always run.
"""

import importlib
import inspect
import json
import sys
import types
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

from strands_robots.tools.use_lerobot import (  # noqa: E402
    _describe_object,
    _discover_modules,
    _import_from_lerobot,
    _serialize_result,
    use_lerobot,
)

# ── Check whether lerobot is available ────────────────────────
_lerobot_available = importlib.util.find_spec("lerobot") is not None
requires_lerobot = pytest.mark.skipif(not _lerobot_available, reason="lerobot is not installed")


# ── _import_from_lerobot ──────────────────────────────────────


@requires_lerobot
class TestImportFromLerobot:
    """Test the dynamic import resolver (requires lerobot)."""

    def test_import_module(self):
        """Import a real lerobot submodule."""
        mod = _import_from_lerobot("policies.factory")
        assert inspect.ismodule(mod)
        assert hasattr(mod, "get_policy_class")

    def test_import_function(self):
        """Import a function from a real lerobot module."""
        func = _import_from_lerobot("policies.factory.get_policy_class")
        assert callable(func)

    def test_import_class(self):
        """Import a class from lerobot."""
        cls = _import_from_lerobot("datasets.lerobot_dataset.LeRobotDataset")
        assert inspect.isclass(cls)

    def test_import_with_lerobot_prefix(self):
        """If user passes 'lerobot.X', still works."""
        mod = _import_from_lerobot("lerobot.policies.factory")
        assert inspect.ismodule(mod)

    def test_import_constant(self):
        """Import a constant value."""
        result = _import_from_lerobot("utils.constants.HF_LEROBOT_CALIBRATION")
        # Should be a Path or string constant
        assert result is not None

    def test_import_deeply_nested(self):
        """Multi-level attribute resolution (module.Class.attribute)."""
        # policies.factory module has get_policy_class function, which has __name__
        cls = _import_from_lerobot("policies.factory.get_policy_class.__name__")
        assert cls == "get_policy_class"

    def test_import_module_fallback_to_attribute(self):
        """When module import fails, tries parent + getattr."""
        # datasets.lerobot_dataset.LeRobotDataset — module import fails, falls back to getattr
        result = _import_from_lerobot("datasets.lerobot_dataset.LeRobotDataset")
        assert inspect.isclass(result)

    def test_import_single_segment(self):
        """Single-segment import (just 'policies')."""
        mod = _import_from_lerobot("policies")
        assert inspect.ismodule(mod)

    def test_import_robots_module(self):
        """Import lerobot.robots subpackage."""
        mod = _import_from_lerobot("robots")
        assert inspect.ismodule(mod)
        assert hasattr(mod, "__path__")

    def test_import_config_class(self):
        """Import a config class."""
        try:
            cls = _import_from_lerobot("robots.config.RobotConfig")
            assert inspect.isclass(cls)
        except ImportError:
            pytest.skip("RobotConfig not found in this lerobot version")


class TestImportFromLerobotErrors:
    """Test import error paths (no lerobot needed)."""

    def test_import_nonexistent_raises(self):
        """Non-existent module raises ImportError."""
        with pytest.raises(ImportError, match="Cannot resolve"):
            _import_from_lerobot("nonexistent.module.DoesNotExist")

    def test_import_nonexistent_attribute_raises(self):
        """Module exists but attribute doesn't → ImportError."""
        with pytest.raises(ImportError, match="Cannot resolve"):
            _import_from_lerobot("policies.factory.this_does_not_exist_XYZ")


# ── _discover_modules ─────────────────────────────────────────


@requires_lerobot
class TestDiscoverModules:
    """Test module discovery with real lerobot."""

    def test_discover_returns_packages(self):
        result = _discover_modules()
        assert "packages" in result
        assert len(result["packages"]) > 0
        assert "policies" in result["packages"]

    def test_discover_returns_key_apis(self):
        result = _discover_modules()
        assert "key_apis" in result
        assert len(result["key_apis"]) > 0
        assert any("factory" in k for k in result["key_apis"])

    def test_discover_modules_list(self):
        result = _discover_modules()
        assert "modules" in result
        assert isinstance(result["modules"], list)

    def test_discover_no_private_modules(self):
        """Should skip modules starting with _."""
        result = _discover_modules()
        for pkg in result["packages"]:
            assert not pkg.startswith("_")
        for mod in result["modules"]:
            assert not mod.startswith("_")


class TestDiscoverModulesNoLerobot:
    """Test module discovery edge cases (no lerobot needed)."""

    def test_discover_robot_configs(self):
        result = _discover_modules()
        # lerobot 0.4.4 has robot configs
        if "robot_configs" in result:
            assert isinstance(result["robot_configs"], list)

    def test_discover_teleop_configs(self):
        result = _discover_modules()
        if "teleop_configs" in result:
            assert isinstance(result["teleop_configs"], list)

    def test_discover_no_lerobot(self):
        """When lerobot is not installed."""
        with patch.dict("sys.modules", {"lerobot": None}):
            result = _discover_modules()
            assert "error" in result
            assert "not installed" in result["error"]


# ── _describe_object ──────────────────────────────────────────


class TestDescribeObject:
    """Test object introspection."""

    def test_describe_class(self):
        """Describe a class."""

        class MyClass:
            """My docstring."""

            def __init__(self, x, y=5):
                pass

            def method_a(self):
                pass

            def _private(self):
                pass

        info = _describe_object(MyClass)
        assert info["type"] == "type"
        assert "method_a" in info["methods"]
        assert "_private" not in info["methods"]
        assert "init_params" in info
        assert "x" in info["init_params"]

    def test_describe_function(self):
        """Describe a function with parameters."""

        def my_func(a: int, b: str = "hello") -> bool:
            """Does stuff."""
            pass

        info = _describe_object(my_func)
        assert info["type"] == "function"
        assert "params" in info
        assert info["params"]["a"]["default"] == "REQUIRED"
        assert info["params"]["b"]["default"] == "hello"
        assert "doc" in info

    def test_describe_module(self):
        import os

        info = _describe_object(os)
        assert info["type"] == "module"
        assert "public_names" in info
        assert len(info["public_names"]) <= 30

    def test_describe_plain_value(self):
        info = _describe_object(42)
        assert "value" in info
        assert "42" in info["value"]

    def test_describe_string_value(self):
        info = _describe_object("hello world")
        assert "value" in info

    def test_describe_class_with_classmethod(self):
        class Foo:
            @classmethod
            def bar(cls):
                pass

            @staticmethod
            def baz():
                pass

        info = _describe_object(Foo)
        assert "class_methods" in info
        assert "bar" in info["class_methods"]
        assert "baz" in info["class_methods"]

    def test_describe_class_init_doc(self):
        class Foo:
            def __init__(self):
                """Init docstring here."""
                pass

        info = _describe_object(Foo)
        assert "init_doc" in info
        assert "Init docstring here" in info["init_doc"]

    def test_describe_callable_without_signature(self):
        """Built-in functions may not have inspectable signatures."""
        info = _describe_object(print)
        assert info["type"] == "builtin_function_or_method"
        # Should still succeed without params

    @requires_lerobot
    def test_describe_real_lerobot_class(self):
        """Describe a real lerobot class."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        info = _describe_object(LeRobotDataset)
        assert info["type"] == "type"
        assert "methods" in info

    def test_describe_doc_truncation(self):
        """Docstrings should be truncated to 500 chars."""

        def long_doc():
            pass

        long_doc.__doc__ = "x" * 1000
        info = _describe_object(long_doc)
        assert len(info.get("doc", "")) <= 500

    def test_describe_inherited_init_doc(self):
        """Class with no explicit __init__ gets object.__init__ doc."""

        class Foo:
            pass

        info = _describe_object(Foo)
        # Python objects always have __init__.__doc__ from object
        assert "init_doc" in info
        assert "Initialize self" in info["init_doc"]


# ── _serialize_result ─────────────────────────────────────────


class TestSerializeResult:
    """Test result serialization."""

    def test_none(self):
        assert _serialize_result(None) == "None"

    def test_string(self):
        assert _serialize_result("hello") == "hello"

    def test_int(self):
        assert _serialize_result(42) == "42"

    def test_float(self):
        assert _serialize_result(3.14) == "3.14"

    def test_bool(self):
        assert _serialize_result(True) == "True"

    def test_list(self):
        result = _serialize_result([1, 2, 3])
        parsed = json.loads(result)
        assert len(parsed) == 3

    def test_tuple(self):
        result = _serialize_result((1, 2))
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_list_truncation(self):
        """Lists truncated to 50 items."""
        result = _serialize_result(list(range(100)))
        parsed = json.loads(result)
        assert len(parsed) == 50

    def test_list_item_truncation(self):
        """Items truncated to 200 chars."""
        long_item = "x" * 500
        result = _serialize_result([long_item])
        parsed = json.loads(result)
        assert len(parsed[0]) <= 200

    def test_dict(self):
        result = _serialize_result({"a": 1, "b": "hello"})
        parsed = json.loads(result)
        assert parsed["a"] == 1
        assert parsed["b"] == "hello"

    def test_dict_truncation(self):
        """Dicts truncated to 50 keys."""
        big_dict = {f"key_{i}": i for i in range(100)}
        result = _serialize_result(big_dict)
        parsed = json.loads(result)
        assert len(parsed) == 50

    def test_dict_non_serializable_value(self):
        """Non-JSON-serializable dict values become strings."""
        result = _serialize_result({"obj": object()})
        parsed = json.loads(result)
        assert isinstance(parsed["obj"], str)

    def test_object_described(self):
        """Unknown objects get described."""

        class Foo:
            def bar(self):
                pass

        result = _serialize_result(Foo())
        parsed = json.loads(result)
        assert parsed["type"] == "Foo"

    def test_dict_with_non_string_keys(self):
        """Dict keys get string-ified."""
        result = _serialize_result({1: "one", 2: "two"})
        parsed = json.loads(result)
        assert "1" in parsed


# ── use_lerobot (main tool function) ──────────────────────────


class TestUseLerobot:
    """Test the main tool function."""

    # ── Discovery mode (requires lerobot) ──

    @requires_lerobot
    def test_discovery_mode(self):
        """Default call returns discovery info."""
        result = use_lerobot()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "LeRobot API Discovery" in text
        assert "Packages:" in text

    @requires_lerobot
    def test_discovery_explicit(self):
        result = use_lerobot(module="__discovery__", method="list_modules")
        assert result["status"] == "success"
        assert "Key APIs" in result["content"][0]["text"]

    def test_discovery_no_lerobot(self):
        with patch.dict("sys.modules", {"lerobot": None}):
            result = use_lerobot(module="__discovery__")
            assert result["status"] == "error"
            assert "not installed" in result["content"][0]["text"]

    @requires_lerobot
    def test_discovery_shows_robots(self):
        result = use_lerobot(module="__discovery__")
        result["content"][0]["text"]
        # Should show robot configs if available
        assert "success" == result["status"]

    @requires_lerobot
    def test_discovery_shows_usage_examples(self):
        result = use_lerobot(module="__discovery__")
        text = result["content"][0]["text"]
        assert "Usage:" in text

    # ── Describe mode (requires lerobot) ──

    @requires_lerobot
    def test_describe_module(self):
        result = use_lerobot(module="policies.factory", method="__describe__")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "policies.factory" in text

    @requires_lerobot
    def test_describe_class(self):
        result = use_lerobot(module="datasets.lerobot_dataset.LeRobotDataset", method="__describe__")
        assert result["status"] == "success"

    @requires_lerobot
    def test_describe_function(self):
        result = use_lerobot(module="policies.factory.get_policy_class", method="__describe__")
        assert result["status"] == "success"

    # ── Call mode (requires lerobot) ──

    @requires_lerobot
    def test_call_get_policy_class(self):
        """Call a real lerobot function."""
        result = use_lerobot(
            module="policies.factory",
            method="get_policy_class",
            parameters={"name": "act"},
        )
        assert result["status"] == "success"
        assert "act" in result["content"][0]["text"].lower() or "Act" in result["content"][0]["text"]

    @requires_lerobot
    def test_call_non_callable_attribute(self):
        """Accessing a non-callable returns its value."""
        result = use_lerobot(module="utils.constants", method="HF_LEROBOT_CALIBRATION")
        assert result["status"] == "success"
        assert "=" in result["content"][0]["text"]

    @requires_lerobot
    def test_call_method_not_found(self):
        """Method that doesn't exist returns error with available list."""
        result = use_lerobot(module="policies.factory", method="this_does_not_exist_XYZ")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]
        assert "Available:" in result["content"][0]["text"]

    @requires_lerobot
    def test_call_with_label(self):
        """Label parameter doesn't break anything."""
        result = use_lerobot(
            module="policies.factory",
            method="get_policy_class",
            parameters={"name": "act"},
            label="Get ACT policy class",
        )
        assert result["status"] == "success"

    @requires_lerobot
    def test_call_no_params(self):
        """Calling with None parameters."""
        result = use_lerobot(module="policies.factory", method="__describe__")
        assert result["status"] == "success"

    # ── Error handling (no lerobot needed) ──

    def test_import_error(self):
        """Module that can't be imported."""
        result = use_lerobot(module="nonexistent.module.X", method="do_thing")
        assert result["status"] == "error"
        assert "Import error" in result["content"][0]["text"]

    @requires_lerobot
    def test_type_error_with_signature(self):
        """Wrong params shows expected signature."""
        result = use_lerobot(
            module="policies.factory",
            method="get_policy_class",
            parameters={"wrong_param": "bad"},
        )
        assert result["status"] == "error"
        assert "TypeError" in result["content"][0]["text"]

    def test_generic_exception(self):
        """Other exceptions are caught — uses module-level monkeypatch."""
        import strands_robots.tools.use_lerobot as _mod

        orig = _mod._import_from_lerobot
        _mod._import_from_lerobot = lambda p: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            result = _mod.use_lerobot(module="anything", method="something")
            assert result["status"] == "error"
            assert "boom" in result["content"][0]["text"]
        finally:
            _mod._import_from_lerobot = orig

    def test_type_error_no_signature(self):
        """TypeError when we can't get the signature — uses monkeypatch."""
        import strands_robots.tools.use_lerobot as _mod

        mock_target = MagicMock(side_effect=TypeError("bad args"))
        mock_target.__name__ = "mock_func"
        mock_obj = MagicMock()
        mock_obj.bad_method = mock_target

        orig = _mod._import_from_lerobot
        _mod._import_from_lerobot = lambda p: mock_obj
        try:
            with patch("inspect.signature", side_effect=ValueError("no sig")):
                result = _mod.use_lerobot(module="mock", method="bad_method")
                assert result["status"] == "error"
                assert "TypeError" in result["content"][0]["text"]
        finally:
            _mod._import_from_lerobot = orig


# ── Integration tests (real lerobot 0.4.4) ───────────────────


@requires_lerobot
class TestUseLeroboIntegration:
    """Integration tests using real lerobot installation."""

    def test_full_workflow_discover_describe_call(self):
        """Full workflow: discover → describe → call."""
        # Step 1: Discover
        disc = use_lerobot()
        assert disc["status"] == "success"

        # Step 2: Describe the factory
        desc = use_lerobot(module="policies.factory.get_policy_class", method="__describe__")
        assert desc["status"] == "success"

        # Step 3: Call it
        call = use_lerobot(
            module="policies.factory",
            method="get_policy_class",
            parameters={"name": "diffusion"},
        )
        assert call["status"] == "success"

    def test_list_available_policies(self):
        """Discover all policy types via lerobot."""
        result = use_lerobot(
            module="policies.factory",
            method="get_policy_class",
            parameters={"name": "act"},
        )
        assert result["status"] == "success"

    def test_describe_lerobot_dataset(self):
        """Describe the LeRobotDataset class to see methods."""
        result = use_lerobot(
            module="datasets.lerobot_dataset.LeRobotDataset",
            method="__describe__",
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        # Should list methods
        assert "methods" in text.lower() or "create" in text.lower()
