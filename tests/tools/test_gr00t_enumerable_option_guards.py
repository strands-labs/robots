"""``gr00t_inference`` refuses an enumerable option it cannot honor.

Five agent-supplied selector strings ride the same detached ``docker exec``
command line as the numeric options ``test_gr00t_numeric_option_guards.py``
covers. ``data_config``, ``embodiment_tag`` and the three TensorRT dtype flags
are interpolated into the argv that :func:`_build_inference_command` builds and
``_start_service`` runs with ``-d``, so a value the inference server's own flag
parser rejects surfaces minutes later inside the container's log rather than as
the tool call's result -- the numerics beside them were refused up front for
exactly that reason, and these were carried through unchecked.

Each of the five names a vocabulary rather than a free string. Every value the
shipped ``data_configs.json`` catalogue and this tool's own docstring name is a
lowercase ``[a-z][a-z0-9_]+`` token, so that is the domain, and the catalogue
test below is what keeps it non-breaking: it is derived from the shipped JSON
rather than a copied list, so a config added there is admitted or the test says
so.

The refusal is asserted to reach neither docker nor a socket, and the
value-reaches-the-builder half is asserted through a ``_start_service`` spy: a
guard that ran after the container work started could not make the rejection a
property of the request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# The module object rather than its members, matching the numeric sibling: these
# tests reach ``gi.subprocess`` to make a side effect fatal and ``gi._start_service``
# to observe what the dispatch let through.
import strands_robots.tools.gr00t_inference as gi

# Values no selector flag can carry. Shell metacharacters and a newline are the
# injection shapes the argv is defended against even though it is argv-style;
# ``So100`` is the ordinary mistake (the catalogue is lowercase); the empty string
# names nothing; and the non-strings cannot index a vocabulary at all.
UNUSABLE_SELECTORS: list[Any] = [
    "so100; rm -rf /",
    "$(whoami)",
    "`id`",
    "a|b",
    "bogus\nInjected",
    "with space",
    "So100",
    "",
    "_leading",
    "9leading",
    123,
    None,
    ["so100"],
]

# The five options and, for each, the extra request state that makes it effective.
EFFECTIVE_SELECTORS: list[tuple[str, dict[str, Any]]] = [
    ("data_config", {}),
    ("embodiment_tag", {}),
    ("vit_dtype", {"use_tensorrt": True}),
    ("llm_dtype", {"use_tensorrt": True}),
    ("dit_dtype", {"use_tensorrt": True}),
]

_DOMAIN_PHRASE = "must be a lowercase selector token"


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool with deliberately off-type values.

    Routed through one ``**kwargs`` funnel because several tests pass values the
    signature's annotations forbid on purpose - that is the input class under
    test - and mypy does not narrow a splatted ``dict[str, Any]``.
    """
    return gi.gr00t_inference(**kwargs)


def _message(result: dict[str, Any]) -> str:
    return str(result.get("message", ""))


def _catalogue_values() -> list[str]:
    """Every ``data_config`` name and alias the package ships.

    Read from the shipped JSON rather than copied, so the admitted set below
    tracks the catalogue instead of a second list to keep in sync.
    """
    path = Path(gi.__file__).resolve().parents[1] / "policies" / "groot" / "data_configs.json"
    catalogue = json.loads(path.read_text(encoding="utf-8"))
    return [*catalogue["configs"], *catalogue.get("aliases", {})]


@pytest.fixture
def no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if a refused call reaches docker or a socket.

    A guard that runs after the container work has already started cannot make
    the rejection a property of the request.
    """

    def _no_subprocess(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("a refused call reached subprocess.run")

    def _no_socket(_port: Any) -> bool:
        raise AssertionError("a refused call opened a socket")

    monkeypatch.setattr(gi.subprocess, "run", _no_subprocess)
    monkeypatch.setattr(gi, "_is_service_running", _no_socket)


@pytest.fixture
def start_service_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the kwargs the dispatch hands the service starter.

    An accepted selector must reach it carrying the caller's value; a refused one
    must not reach it at all. Standing in for ``_start_service`` keeps every case
    off docker without hiding which half of the boundary the value stopped at.
    """
    seen: list[dict[str, Any]] = []

    def _spy(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return {"status": "success", "message": "stub"}

    monkeypatch.setattr(gi, "_start_service", _spy)
    return seen


class TestAnUnusableSelectorIsRefused:
    """Each effective selector is held to the token domain, by name."""

    @pytest.mark.parametrize("param,extra", EFFECTIVE_SELECTORS)
    @pytest.mark.parametrize("value", UNUSABLE_SELECTORS)
    def test_refused_naming_the_parameter(
        self,
        param: str,
        extra: dict[str, Any],
        value: Any,
        no_side_effects: None,
    ) -> None:
        result = _call(
            action="start",
            checkpoint_path="/tmp/ckpt",
            container_name="gr00t-test",
            **{param: value},
            **extra,
        )
        assert result["status"] == "error", result
        message = _message(result)
        assert param in message, message
        assert _DOMAIN_PHRASE in message, message

    def test_the_refusal_says_where_an_unreadable_value_would_have_surfaced(self, no_side_effects: None) -> None:
        """The message names the reason the value could not just be passed on."""
        result = _call(
            action="start",
            checkpoint_path="/tmp/ckpt",
            container_name="gr00t-test",
            data_config="so100; rm -rf /",
        )
        assert "container log" in _message(result), _message(result)

    def test_a_refused_selector_never_reaches_the_service_starter(
        self, start_service_spy: list[dict[str, Any]]
    ) -> None:
        result = _call(
            action="start",
            checkpoint_path="/tmp/ckpt",
            container_name="gr00t-test",
            embodiment_tag="gr1 && curl evil",
        )
        assert result["status"] == "error", result
        assert start_service_spy == [], start_service_spy


class TestTheShippedCatalogueIsAdmitted:
    """The domain refuses nothing the package itself names as a valid value."""

    def test_every_catalogue_data_config_reaches_the_service_starter(
        self, start_service_spy: list[dict[str, Any]]
    ) -> None:
        values = _catalogue_values()
        assert len(values) >= 20, f"premise: catalogue looks unread ({len(values)} values)"
        refused = []
        for value in values:
            start_service_spy.clear()
            result = _call(
                action="start",
                checkpoint_path="/tmp/ckpt",
                container_name="gr00t-test",
                data_config=value,
            )
            if _DOMAIN_PHRASE in _message(result):
                refused.append(value)
            elif start_service_spy:
                assert start_service_spy[0]["data_config"] == value
        assert refused == [], f"the domain refuses shipped catalogue values: {refused}"

    @pytest.mark.parametrize("dtype", ["fp8", "nvfp4", "fp16", "bf16", "fp32", "int8"])
    def test_a_real_tensorrt_precision_is_admitted(self, dtype: str, start_service_spy: list[dict[str, Any]]) -> None:
        result = _call(
            action="start",
            checkpoint_path="/tmp/ckpt",
            container_name="gr00t-test",
            use_tensorrt=True,
            vit_dtype=dtype,
            llm_dtype=dtype,
            dit_dtype=dtype,
        )
        assert _DOMAIN_PHRASE not in _message(result), _message(result)


class TestEffectiveOptionsOnly:
    """A caller is never refused for a value the requested action ignores."""

    def test_n1_7_ignores_data_config_because_its_entrypoint_drops_the_flag(
        self, start_service_spy: list[dict[str, Any]]
    ) -> None:
        result = _call(
            action="start",
            checkpoint_path="/tmp/ckpt",
            container_name="gr00t-test",
            protocol="n1.7",
            data_config="not; a config",
        )
        assert _DOMAIN_PHRASE not in _message(result), _message(result)

    @pytest.mark.parametrize("param", ["vit_dtype", "llm_dtype", "dit_dtype"])
    def test_a_dtype_is_inert_without_tensorrt(self, param: str, start_service_spy: list[dict[str, Any]]) -> None:
        result = _call(
            action="start",
            checkpoint_path="/tmp/ckpt",
            container_name="gr00t-test",
            use_tensorrt=False,
            **{param: "not; a dtype"},
        )
        assert _DOMAIN_PHRASE not in _message(result), _message(result)

    @pytest.mark.parametrize("action", ["status", "stop", "list", "find_containers"])
    def test_an_action_that_builds_no_command_line_reads_no_selector(
        self, action: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gi, "_find_gr00t_containers", lambda: {"status": "success", "containers": []})
        monkeypatch.setattr(gi, "_check_service_status", lambda _p: {"status": "success"})
        monkeypatch.setattr(gi, "_stop_service", lambda _p: {"status": "success"})
        monkeypatch.setattr(gi, "_list_running_services", lambda: {"status": "success"})
        result = _call(action=action, data_config="not; a config")
        assert _DOMAIN_PHRASE not in _message(result), _message(result)


class TestTheGuardCoversEveryCommandLineAction:
    """The action table is the set that reaches the command builder.

    Derived from the numeric table's own command-line entry rather than restated:
    ``denoising_steps`` is a ``_build_inference_command`` argument, so the actions
    that read it are exactly the actions that build an argv, and the selector
    strings in that argv must be held for the same set.
    """

    def test_the_selector_table_matches_the_command_line_actions(self) -> None:
        numeric = {key for key, options in gi._ACTION_NUMERIC_OPTIONS.items() if "denoising_steps" in options}
        assert numeric, "premise: no action reads denoising_steps"
        assert set(gi._ACTION_ENUMERABLE_OPTIONS) == numeric

    def test_every_selector_the_command_builder_takes_is_in_the_table(self) -> None:
        import inspect

        params = set(inspect.signature(gi._build_inference_command).parameters)
        covered = set().union(*gi._ACTION_ENUMERABLE_OPTIONS.values())
        assert covered <= params, covered - params
        assert covered == {"data_config", "embodiment_tag", "vit_dtype", "llm_dtype", "dit_dtype"}
