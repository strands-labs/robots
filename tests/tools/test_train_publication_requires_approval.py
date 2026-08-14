"""Blocked train flags reach the operator gate whichever spelling names them.

``lerobot_train`` refuses a set of LeRobot flags that control output paths,
telemetry, publication and code loading unless an operator approves them through
``tool_context``. Three of those blocked flags are *also* named parameters of the
tool, so a caller can name them without going through ``extra_flags`` at all --
and a gate that only inspects ``extra_flags`` never sees that spelling.

These tests pin the wiring rather than the gate helper (whose accepted domain is
already covered elsewhere):

1. Every gated input -- a blocked ``extra_flags`` key, ``pretrained_path`` and
   ``push_to_hub`` -- is refused *before* a training subprocess is launched, and
   the refusal names the flag.
2. ``push_to_hub=True`` and ``extra_flags={"push_to_hub": True}`` reach the same
   verdict, so the publication posture does not depend on how the caller spelled
   it.
3. The documented escape hatches (an approving ``tool_context``, the allowlist
   env var, ``BYPASS_TOOL_CONSENT``) still launch, and the default
   ``push_to_hub=False`` is not gated.
4. A parity net over the blocklist: any blocked flag that is also a named
   parameter must be gated, or be named in this module's documented exemption.
5. Which allowlist entry clears which spelling: push_to_hub is the one
   blocked flag named twice in the blocklist, so a headless run's
   STRANDS_TRAIN_EXTRA_FLAGS_ALLOW value has to match the spelling the call
   used -- and the description an agent reads has to say which.

Everything here is hardware-, GPU- and network-free: ``subprocess.Popen`` is
replaced by a recorder, so "launched" means "the tool would have started
training with this argv".
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.tools.lerobot_train as train_mod

lerobot_train = train_mod.lerobot_train

_ALLOW_ENV = train_mod._EXTRA_FLAGS_ALLOW_ENV
_BYPASS_ENV = train_mod._BYPASS_CONSENT_ENV

# ``output_dir`` is blocked in ``extra_flags`` yet ungated as a named parameter,
# deliberately: the builder always emits exactly one ``--output_dir``, so the
# blocklist entry stops a caller smuggling a second, conflicting one rather than
# stopping the tool writing where the caller asked. Gating the named parameter
# would refuse a documented, ordinary argument.
_UNGATED_BY_DESIGN = frozenset({"output_dir"})


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


class _FakeProc:
    """Stand-in for a launched training process."""

    pid = 4242

    def poll(self) -> None:
        return None


@pytest.fixture
def launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A dataset, an isolated session store, and a recording ``Popen``.

    The ambient environment of a harness run may carry ``BYPASS_TOOL_CONSENT``,
    which switches the gate off wholesale; these tests are about the gate, so
    both escape hatches are cleared unless a test sets one itself.
    """
    monkeypatch.delenv(_BYPASS_ENV, raising=False)
    monkeypatch.delenv(_ALLOW_ENV, raising=False)

    dataset = tmp_path / "ds"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps({"total_episodes": 10}))

    sessions = tmp_path / ".sessions"
    sessions.mkdir()
    monkeypatch.setattr(train_mod, "SESSION_DIR", sessions)

    launched: list[list[str]] = []

    def _fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        launched.append(list(cmd))
        return _FakeProc()

    monkeypatch.setattr(train_mod.subprocess, "Popen", _fake_popen)
    return {"dataset": dataset, "launched": launched}


def _start(launcher: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Drive ``action='start'`` through the tool with a unique session name."""
    kwargs.setdefault("tool_context", None)
    kwargs.setdefault("policy_type", "act")
    kwargs.setdefault("session_name", f"s{len(launcher['launched'])}")
    return dict(lerobot_train(action="start", dataset_root=str(launcher["dataset"]), **kwargs))


def _approving_context() -> MagicMock:
    ctx = MagicMock()
    ctx.interrupt.return_value = "y"
    return ctx


def _declining_context() -> MagicMock:
    ctx = MagicMock()
    ctx.interrupt.return_value = "no"
    return ctx


class TestEveryGatedInputRefusesBeforeLaunch:
    """A blocked flag is refused before a training subprocess exists."""

    @pytest.mark.parametrize(
        ("kwargs", "flag"),
        [
            ({"extra_flags": {"output_dir": "/tmp/elsewhere"}}, "output_dir"),
            ({"extra_flags": {"wandb.api_key": "sk-secret"}}, "wandb.api_key"),
            ({"extra_flags": {"config_path": "/tmp/attacker.yaml"}}, "config_path"),
            ({"pretrained_path": "attacker/model"}, "policy.pretrained_path"),
            ({"push_to_hub": True}, "policy.push_to_hub"),
        ],
        ids=["extra-output_dir", "extra-wandb_key", "extra-config_path", "named-pretrained_path", "named-push_to_hub"],
    )
    def test_refused_without_a_tool_context(self, launcher: dict[str, Any], kwargs: dict[str, Any], flag: str) -> None:
        result = _start(launcher, **kwargs)
        assert result["status"] == "error", result
        assert flag in _texts(result)
        assert launcher["launched"] == [], "a refused call must not launch training"

    def test_a_benign_extra_flag_still_launches(self, launcher: dict[str, Any]) -> None:
        """The gate refuses the blocklist, not every extra flag."""
        result = _start(launcher, extra_flags={"policy.optimizer_lr": "1e-4"})
        assert result["status"] == "success", _texts(result)
        assert len(launcher["launched"]) == 1
        assert "--policy.optimizer_lr=1e-4" in launcher["launched"][0]

    def test_the_default_publish_posture_is_not_gated(self, launcher: dict[str, Any]) -> None:
        """``push_to_hub=False`` is the default for every call and must stay free."""
        result = _start(launcher)
        assert result["status"] == "success", _texts(result)
        assert "--policy.push_to_hub=false" in launcher["launched"][0]


class TestPublicationPostureDoesNotDependOnSpelling:
    """One LeRobot flag, two spellings, one verdict."""

    def test_both_spellings_of_push_to_hub_are_refused_alike(self, launcher: dict[str, Any]) -> None:
        named = _start(launcher, push_to_hub=True)
        smuggled = _start(launcher, extra_flags={"push_to_hub": True})

        assert named["status"] == smuggled["status"] == "error"
        assert "push_to_hub" in _texts(named)
        assert "push_to_hub" in _texts(smuggled)
        assert launcher["launched"] == [], "neither spelling may launch unapproved"

    def test_declining_the_publish_launches_nothing(self, launcher: dict[str, Any]) -> None:
        result = _start(launcher, push_to_hub=True, tool_context=_declining_context())
        assert result["status"] == "error", result
        assert "declined" in _texts(result)
        assert launcher["launched"] == []

    def test_approving_the_publish_launches_with_the_flag(self, launcher: dict[str, Any]) -> None:
        ctx = _approving_context()
        result = _start(launcher, push_to_hub=True, tool_context=ctx)

        assert result["status"] == "success", _texts(result)
        assert "--policy.push_to_hub=true" in launcher["launched"][0]
        assert ctx.interrupt.called, "approval must be sought from the operator"

    def test_the_publish_destination_cannot_ride_along_unapproved(self, launcher: dict[str, Any]) -> None:
        """A destination is only reachable once the publish itself is approved.

        ``--policy.repo_id`` is a required LeRobot flag even for a purely local
        run, so it is deliberately not blocklisted. What must not happen is the
        pair reaching the argv with nobody asked: the publish decision is the
        gated one, and it carries the destination with it.
        """
        unapproved = _start(
            launcher,
            push_to_hub=True,
            extra_flags={"policy.repo_id": "attacker/stolen-policy"},
        )
        assert unapproved["status"] == "error", unapproved
        assert launcher["launched"] == []

        approved = _start(
            launcher,
            push_to_hub=True,
            extra_flags={"policy.repo_id": "operator/reviewed"},
            tool_context=_approving_context(),
        )
        assert approved["status"] == "success", _texts(approved)
        argv = launcher["launched"][0]
        assert "--policy.push_to_hub=true" in argv
        assert "--policy.repo_id=operator/reviewed" in argv

    @pytest.mark.parametrize("escape", ["allowlist", "bypass"])
    def test_the_documented_escape_hatches_still_publish(
        self, launcher: dict[str, Any], monkeypatch: pytest.MonkeyPatch, escape: str
    ) -> None:
        if escape == "allowlist":
            monkeypatch.setenv(_ALLOW_ENV, "policy.push_to_hub")
        else:
            monkeypatch.setenv(_BYPASS_ENV, "true")

        result = _start(launcher, push_to_hub=True)
        assert result["status"] == "success", _texts(result)
        assert "--policy.push_to_hub=true" in launcher["launched"][0]


def _tool_source() -> str:
    return inspect.getsource(train_mod)


def _named_parameters() -> list[str]:
    """Parameter names of the ``lerobot_train`` dispatcher, read from source.

    The tool is wrapped by ``@tool``, so its signature is read from the module
    source rather than from the decorated object.
    """
    tree = ast.parse(_tool_source())
    fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "lerobot_train")
    return [arg.arg for arg in fn.args.args + fn.args.kwonlyargs]


def _gated_flag_keys() -> set[str]:
    """Blocklist keys the dispatcher hands to the gate as literal dict keys."""
    tree = ast.parse(_tool_source())
    fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "lerobot_train")
    keys: set[str] = set()
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_gate_extra_flags"):
            continue
        first = node.args[0]
        if isinstance(first, ast.Dict):
            keys.update(key.value for key in first.keys if isinstance(key, ast.Constant) and isinstance(key.value, str))
    return keys


class TestBlockedFlagsThatAreAlsoNamedParametersAreGated:
    """Parity net: the blocklist and the named parameters cannot drift apart."""

    def test_the_survey_finds_the_known_named_parameters(self) -> None:
        """Non-vacuity: the scan really sees the tool's parameters and blocklist."""
        params = _named_parameters()
        assert {"pretrained_path", "push_to_hub", "output_dir", "extra_flags"} <= set(params)
        assert "policy.push_to_hub" in train_mod._BLOCKED_EXTRA_FLAGS

    def test_every_blocked_named_parameter_is_gated_or_documented(self) -> None:
        params = set(_named_parameters())
        gated_tails = {key.rsplit(".", 1)[-1] for key in _gated_flag_keys()}

        blocked_named = {
            flag.rsplit(".", 1)[-1] for flag in train_mod._BLOCKED_EXTRA_FLAGS if flag.rsplit(".", 1)[-1] in params
        }
        adrift = blocked_named - gated_tails - _UNGATED_BY_DESIGN
        assert not adrift, (
            f"blocked flags reachable through an ungated named parameter: {sorted(adrift)}. "
            "Route each through _gate_extra_flags, or record why it is exempt."
        )

    def test_the_exemption_still_describes_a_real_parameter(self) -> None:
        """A stale exemption must be deleted, not left as a hole."""
        params = set(_named_parameters())
        blocked_tails = {flag.rsplit(".", 1)[-1] for flag in train_mod._BLOCKED_EXTRA_FLAGS}
        for name in _UNGATED_BY_DESIGN:
            assert name in params, f"exempt name {name!r} is not a parameter any more"
            assert name in blocked_tails, f"exempt name {name!r} is not blocklisted any more"


def _push_to_hub_description() -> str:
    """The ``push_to_hub`` description an agent reads, from the tool's own schema."""
    props = lerobot_train.tool_spec["inputSchema"]["json"]["properties"]
    return str(props["push_to_hub"]["description"])


def _allow_key_from_schema() -> str:
    """The allowlist entry the schema tells a headless caller to set."""
    match = re.search(r"STRANDS_TRAIN_EXTRA_FLAGS_ALLOW=([A-Za-z0-9_.]+)", _push_to_hub_description())
    assert match, "the description must name the allowlist entry that clears this parameter"
    return match.group(1)


class TestWhichAllowlistEntryClearsWhichSpelling:
    """One flag, two spellings, two allowlist entries -- and the schema says which.

    ``push_to_hub`` is the only blocked flag the blocklist names twice, bare and
    ``policy.``-prefixed. The named parameter is gated under the prefixed key
    because that is the only spelling LeRobot accepts; the bare key covers the raw
    ``extra_flags`` passthrough. So the ``STRANDS_TRAIN_EXTRA_FLAGS_ALLOW`` value a
    headless run needs depends on how the call named the publish, and the
    description an agent reads has to say which one -- an unattended run that
    pre-approves the wrong spelling gets an approval prompt with nobody there to
    answer it.
    """

    @pytest.mark.parametrize(
        ("allow", "parameter_publishes", "expected_flag"),
        [
            ("policy.push_to_hub", True, "--policy.push_to_hub=true"),
            # The passthrough echoes the caller's own literal, hence the capital T:
            # the named parameter lowercases the bool, ``extra_flags`` does not.
            ("push_to_hub", False, "--push_to_hub=True"),
        ],
        ids=["prefixed-entry-clears-the-parameter", "bare-entry-clears-the-passthrough"],
    )
    def test_an_allowlist_entry_clears_only_its_own_spelling(
        self,
        launcher: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        allow: str,
        parameter_publishes: bool,
        expected_flag: str,
    ) -> None:
        monkeypatch.setenv(_ALLOW_ENV, allow)

        named = _start(launcher, session_name="named", push_to_hub=True)
        smuggled = _start(launcher, session_name="smuggled", extra_flags={"push_to_hub": True})

        assert (named["status"] == "success") is parameter_publishes, _texts(named)
        assert (smuggled["status"] == "success") is (not parameter_publishes), _texts(smuggled)

        assert len(launcher["launched"]) == 1, "one entry clears exactly one spelling"
        assert expected_flag in launcher["launched"][0]

    def test_following_the_schema_pre_approves_the_named_parameter(
        self, launcher: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The key the agent-facing description names really clears this parameter.

        This is the remedy the description prescribes, executed rather than
        restated: it fails if the description ever names a key the gate does not
        honour, whatever the wording around it.
        """
        monkeypatch.setenv(_ALLOW_ENV, _allow_key_from_schema())

        result = _start(launcher, push_to_hub=True)

        assert result["status"] == "success", _texts(result)
        assert "--policy.push_to_hub=true" in launcher["launched"][0]

    def test_the_description_does_not_claim_the_spellings_share_an_opt_out(self) -> None:
        """The two spellings share the approval prompt, not the allowlist entry."""
        description = _push_to_hub_description()
        assert "policy.push_to_hub" in description, "the description must name the key it is gated under"
        assert "exactly as" not in description, (
            "the parameter and the extra_flags passthrough are gated under different keys, "
            "so the description must not claim one opt-out covers both"
        )

    def test_only_the_prefixed_spelling_is_a_lerobot_flag(self) -> None:
        """``push_to_hub`` is a policy-config field, so ``--policy.`` is the only spelling.

        This is what makes the asymmetry correct rather than accidental: the named
        parameter has to synthesize the prefixed key because a bare
        ``--push_to_hub`` is not a flag LeRobot's parser knows, which leaves the
        bare blocklist entry covering the raw passthrough alone.
        """
        pytest.importorskip("lerobot")
        import dataclasses

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.train import TrainPipelineConfig

        assert "push_to_hub" not in {field.name for field in dataclasses.fields(TrainPipelineConfig)}
        assert "push_to_hub" in {field.name for field in dataclasses.fields(PreTrainedConfig)}
