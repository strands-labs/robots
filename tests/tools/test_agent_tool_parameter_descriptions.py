"""Every parameter an agent tool exposes must describe itself to the model.

A ``@tool`` function's input schema is derived from its docstring by
``docstring_parser``. When an entry for a parameter is not found in the
``Args:`` section, the decorator substitutes the placeholder
``"Parameter <name>"`` - a string that names the parameter and says nothing
about it. The model driving the tool reads that schema and nothing else, so a
parameter with a placeholder description is undiscoverable: it cannot be
learned that ``hf_repo`` is required for one action, that ``remove_volumes``
discards downloaded checkpoints, or which two values ``dagger_input_device``
accepts.

Three spellings produce a placeholder, and the second two look like
documentation in the source, which is what makes the loss silent:

* the entry is absent;
* the entry sits under a section header other than ``Args:``, where the parser
  discards it (``TestTheMechanism`` pins this, and it also drops the prose from
  the tool description);
* one entry names several parameters at once (``a / b: ...``), which the parser
  reads as a single parameter literally named ``"a / b"``, matching none of
  them.

The two assertions below are the consumer-side and the producer-side view of
the same rule: no exposed parameter may carry the placeholder, and no docstring
entry may name a parameter the function does not have. Both read live objects,
so neither needs an exemption list.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import pkgutil
from typing import Any

import docstring_parser
import pytest

import strands_robots.tools as tools_pkg

# Every bound agent tool in the package. Derived from the package object rather
# than a path literal, so a module added later is covered without an edit.
_TOOLS_DIR = pathlib.Path(tools_pkg.__file__).parent


def _bound_tools() -> list[tuple[str, str, Any]]:
    """Return ``(module, name, tool)`` for every ``@tool`` in the package."""
    found: list[tuple[str, str, Any]] = []
    for info in pkgutil.iter_modules([str(_TOOLS_DIR)]):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"strands_robots.tools.{info.name}")
        for name, obj in vars(module).items():
            spec = getattr(obj, "tool_spec", None)
            if isinstance(spec, dict) and spec.get("name") == name:
                found.append((info.name, name, obj))
    return sorted(found)


_BOUND_TOOLS = _bound_tools()
_IDS = [f"{mod}.{name}" for mod, name, _ in _BOUND_TOOLS]

# The tools this guard is known to cover. An exact set, so a scan that resolved
# somewhere else - or stopped finding tools - fails rather than passing over
# nothing.
_EXPECTED_TOOLS = frozenset(
    {
        "download_assets",
        "gr00t_inference",
        "harness_memory",
        "lerobot_calibrate",
        "lerobot_camera",
        "lerobot_teleoperate",
        "lerobot_train",
        "load_episode",
        "pose_tool",
        "read_predicate_verdict",
        "robot_mesh",
        "run_policy",
        "sample_frames",
        "serial_tool",
        "train_policy",
        "use_lerobot",
        "use_ros",
        "use_rosbridge",
        "use_rtps",
        "write_label",
    }
)


def _placeholder(name: str) -> str:
    """The description the decorator substitutes for an undocumented param."""
    return f"Parameter {name}"


def _source_function(module_name: str, func_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    module = importlib.import_module(f"strands_robots.tools.{module_name}")
    assert module.__file__ is not None
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == func_name:
            return node
    raise AssertionError(f"{module_name}.{func_name} not found in source")


def _declared_parameters(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = func.args
    return {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}


class TestEveryExposedParameterIsDescribed:
    """The consumer-side view: what the model receives."""

    def test_the_scan_finds_every_known_tool(self) -> None:
        assert {name for _mod, name, _tool in _BOUND_TOOLS} == _EXPECTED_TOOLS

    @pytest.mark.parametrize(("module_name", "func_name", "tool"), _BOUND_TOOLS, ids=_IDS)
    def test_no_parameter_carries_the_placeholder_description(
        self, module_name: str, func_name: str, tool: Any
    ) -> None:
        properties = tool.tool_spec["inputSchema"]["json"]["properties"]
        undescribed = sorted(
            name for name, schema in properties.items() if schema.get("description", "").strip() == _placeholder(name)
        )
        assert undescribed == [], (
            f"{module_name}.{func_name} exposes parameters the model cannot learn anything "
            f"about: {undescribed}. Give each one an entry in the Args: section of "
            f"{func_name}'s docstring."
        )


class TestEveryDocstringEntryNamesARealParameter:
    """The producer-side view: what the docstring claims to document."""

    @pytest.mark.parametrize(("module_name", "func_name", "tool"), _BOUND_TOOLS, ids=_IDS)
    def test_no_entry_names_a_parameter_the_tool_does_not_have(
        self, module_name: str, func_name: str, tool: Any
    ) -> None:
        func = _source_function(module_name, func_name)
        declared = _declared_parameters(func)
        parsed = docstring_parser.parse(ast.get_docstring(func) or "")
        unmatched = sorted(p.arg_name for p in parsed.params if p.arg_name not in declared)
        assert unmatched == [], (
            f"{module_name}.{func_name} has Args: entries naming no parameter: {unmatched}. "
            f"An entry covering several parameters at once is read as one parameter with "
            f"that literal name, so it describes none of them - give each its own entry."
        )


class TestTheMechanism:
    """Pin why the three spellings lose the text, so the rule explains itself."""

    def test_an_entry_under_another_section_header_is_discarded(self) -> None:
        parsed = docstring_parser.parse(
            "Short.\n"
            "\n"
            "    Args:\n"
            "        documented: Reaches the schema.\n"
            "\n"
            "    Container lifecycle args:\n"
            "        orphan: Reads as documentation and reaches nothing.\n"
        )
        assert [p.arg_name for p in parsed.params] == ["documented"]

    def test_one_entry_naming_several_parameters_matches_none_of_them(self) -> None:
        parsed = docstring_parser.parse("Short.\n\n    Args:\n        lora_r / lora_alpha: Two at once.\n")
        assert [p.arg_name for p in parsed.params] == ["lora_r / lora_alpha"]

    def test_prose_reaches_the_description_only_before_args(self) -> None:
        before = docstring_parser.parse(
            "Short.\n\n    Operator-configured:\n        KEPT.\n\n    Args:\n        documented: Reaches the schema.\n"
        )
        after = docstring_parser.parse(
            "Short.\n"
            "\n"
            "    Args:\n"
            "        documented: Reaches the schema.\n"
            "\n"
            "    Operator-configured:\n"
            "        DROPPED.\n"
        )
        assert "KEPT." in (before.long_description or "")
        assert "DROPPED." not in (after.long_description or "")


class TestTheGuardWouldNoticeARegression:
    """A scan that matched nothing would report a clean sweep."""

    def test_a_planted_undocumented_parameter_is_reported(self) -> None:
        properties = {
            "described": {"description": "Something useful."},
            "planted": {"description": _placeholder("planted")},
        }
        undescribed = [
            name for name, schema in properties.items() if schema.get("description", "").strip() == _placeholder(name)
        ]
        assert undescribed == ["planted"]

    def test_a_planted_multi_parameter_entry_is_reported(self) -> None:
        parsed = docstring_parser.parse("Short.\n\n    Args:\n        real: Fine.\n        one, two: Both at once.\n")
        unmatched = [p.arg_name for p in parsed.params if p.arg_name not in {"real", "one", "two"}]
        assert unmatched == ["one, two"]


class TestTheLifecycleParametersAreDiscoverable:
    """The seven options the container-lifecycle actions read (see #148).

    Their descriptions existed in the source under a ``Container lifecycle
    args`` header, which the parser drops, so none of them reached the model.
    """

    LIFECYCLE_PARAMETERS = (
        "hf_repo",
        "hf_subfolder",
        "hf_local_dir",
        "hf_token",
        "lifecycle",
        "remove_volumes",
        "force",
    )

    def test_each_lifecycle_option_describes_itself(self) -> None:
        from strands_robots.tools.gr00t_inference import gr00t_inference

        properties = gr00t_inference.tool_spec["inputSchema"]["json"]["properties"]
        for name in self.LIFECYCLE_PARAMETERS:
            description = properties[name].get("description", "").strip()
            assert description != _placeholder(name)
            assert len(description) > len(_placeholder(name))

    def test_a_destructive_option_says_what_it_destroys(self) -> None:
        from strands_robots.tools.gr00t_inference import gr00t_inference

        properties = gr00t_inference.tool_spec["inputSchema"]["json"]["properties"]
        assert "checkpoint" in properties["remove_volumes"]["description"]

    def test_the_operator_configured_note_reaches_the_description(self) -> None:
        from strands_robots.tools.gr00t_inference import gr00t_inference

        description = gr00t_inference.tool_spec["description"]
        assert "host RCE" in description
        assert "STRANDS_GR00T_REPO_URL_ALLOW" in description
