"""The "no ONNX session loaded" refusal must name a seam that actually installs one.

:class:`~strands_robots.policies.wbc.WBCPolicy` accepts ``**kwargs`` as a
forward-compatibility absorber (providers must ignore unknown keywords rather
than raise), so a session handed to the constructor under a keyword the
signature does not declare is silently dropped. A refusal that points the caller
at such a keyword survives its own remedy: applying it verbatim reproduces the
identical error.

These tests grade the refusal mechanically - every keyword it advertises must be
a declared constructor parameter, and the attribute spelling it names must lift
it - so the wording cannot drift back onto a keyword the absorber eats.

The sibling :class:`~strands_robots.policies.protomotions.ProtoMotionsPolicy`
states the same principle from the other side: it declares every injectable as
an explicit parameter and takes no ``**kwargs``, "so a typo raises ``TypeError``
at build time rather than being swallowed".
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS, WBCPolicy

_NUM_ACTIONS = 15


class _StubInput:
    name = "obs"


class _StubSession:
    """Minimal stand-in for ``onnxruntime.InferenceSession``."""

    def get_inputs(self) -> list[_StubInput]:
        return [_StubInput()]

    def run(self, output_names: Any, feed: dict[str, Any]) -> list[np.ndarray]:
        return [np.zeros((1, _NUM_ACTIONS), dtype=np.float32)]


def _g1_keys() -> list[str]:
    """Joint key order as ``robot_joint_names`` reports it for the MuJoCo G1."""
    return ["floating_base_joint", *WBC_G1_ALL_JOINTS]


def _observation() -> dict[str, float]:
    return {k: 0.0 for k in _g1_keys()}


def _policy_with_no_session(**init_kwargs: Any) -> WBCPolicy:
    """A policy whose sessions were never loaded, ready to run one step."""
    policy = WBCPolicy(allow_missing_models=True, **init_kwargs)
    policy.set_robot_state_keys(_g1_keys())
    return policy


def _declared_parameters() -> set[str]:
    """Parameter names ``WBCPolicy.__init__`` binds, excluding the absorber."""
    return {
        name
        for name, param in inspect.signature(WBCPolicy.__init__).parameters.items()
        if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
    }


def _refusal_text() -> str:
    policy = _policy_with_no_session()
    with pytest.raises(RuntimeError, match="no ONNX session loaded") as excinfo:
        policy.get_actions_sync(_observation(), "")
    return str(excinfo.value)


class TestTheMissingSessionRefusalNamesAWorkingSeam:
    def test_every_keyword_it_advertises_is_a_declared_constructor_parameter(self) -> None:
        """A keyword the absorber eats is a dead-end remedy: it changes nothing."""
        message = _refusal_text()
        declared = _declared_parameters()
        absorbed = sorted({name for name in re.findall(r"\b([a-z_][a-z0-9_]*)=", message)} - declared)
        assert not absorbed, (
            f"the refusal advertises {absorbed} as constructor keyword(s), but "
            f"WBCPolicy.__init__ declares only {sorted(declared)} and absorbs the "
            "rest into **kwargs, so passing them installs nothing and the same "
            f"refusal is raised again. Message: {message!r}"
        )

    def test_assigning_the_attributes_it_names_lifts_the_refusal(self) -> None:
        """Apply the remedy the message states, then assert the step succeeds."""
        message = _refusal_text()
        named = [token for token in re.findall(r"`([a-z_][a-z0-9_]*)`", message) if token.endswith("_session")]
        assert named, (
            "the refusal names no session attribute to assign, so a caller who "
            f"reads it has no stated way out. Message: {message!r}"
        )

        policy = _policy_with_no_session()
        for attribute in named:
            assert hasattr(policy, attribute), f"WBCPolicy has no {attribute!r} attribute"
            setattr(policy, attribute, _StubSession())

        actions = policy.get_actions_sync(_observation(), "", target_velocity=[0.5, 0.0, 0.0])
        assert len(actions) == 1
        assert len(actions[0]) == _NUM_ACTIONS

    def test_the_refusal_fires_when_no_session_was_installed(self) -> None:
        """Premise: without a session the policy refuses instead of emitting zeros."""
        policy = _policy_with_no_session()
        with pytest.raises(RuntimeError, match="no ONNX session loaded"):
            policy.get_actions_sync(_observation(), "")

    def test_a_session_handed_to_the_constructor_is_still_dropped(self) -> None:
        """Pins the fact the message states, so the two cannot drift apart."""
        policy = _policy_with_no_session(policy_session=_StubSession(), walk_session=_StubSession())
        assert policy.policy_session is None
        assert policy.walk_session is None

    def test_an_unknown_keyword_is_still_absorbed_without_raising(self) -> None:
        """The provider contract stands: unknown keywords are ignored, not refused."""
        policy = _policy_with_no_session(some_future_provider_knob=object())
        assert isinstance(policy, WBCPolicy)
