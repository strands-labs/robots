"""``LerobotLocalPolicy.is_chunk_emitting()`` must detect every chunk-emitting
regime, not only the ones whose chunk shape is visible through
``execution_horizon``.

The async-RTC pipeline in :meth:`strands_robots.simulation.policy_runner.PolicyRunner.run`
reads :meth:`Policy.is_chunk_emitting` to auto-enable latency masking for
exactly the policies that benefit from overlapping inference with chunk
execution. :meth:`LerobotLocalPolicy.is_chunk_emitting` widens the base
horizon check with two extra signals, and a miss on any one silently drops a
policy back onto the synchronous loop:

* ``execution_horizon > 1`` -- the base signal (ACT, diffusion, pi0, SmolVLA).
* :attr:`~LerobotLocalPolicy.supports_rtc` -- a flow-matching model that blends
  chunk seams internally is chunk-emitting by construction, even when its RTC
  execution horizon is 1.
* :meth:`LerobotLocalPolicy._requires_action_chunk` -- MolmoAct2 must be driven
  via ``predict_action_chunk`` and its trained chunk is not always reflected in
  ``actions_per_step``.

These pin the observable output (the boolean) for each branch and the
all-false single-step case, so a refactor that drops a signal fails here.
"""

from __future__ import annotations

from unittest.mock import patch

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


def _unloaded(**kw) -> LerobotLocalPolicy:
    """Construct a policy without touching the network / a real checkpoint.

    ``_load_model`` is patched out, so the returned policy carries its unloaded
    defaults (``actions_per_step=1``, RTC off, ``_policy is None``) that each
    test then drives into a specific chunk-emitting regime.
    """
    with patch.object(LerobotLocalPolicy, "_load_model"):
        return LerobotLocalPolicy(pretrained_name_or_path="test/model", **kw)


def test_base_horizon_chunk_is_emitting() -> None:
    """A trained chunk longer than one action (ACT/diffusion/SmolVLA) emits
    chunks via the inherited ``execution_horizon > 1`` signal alone."""
    pol = _unloaded()
    pol.actions_per_step = 8  # execution_horizon follows actions_per_step off-RTC

    assert pol.execution_horizon == 8
    assert pol.is_chunk_emitting() is True


def test_rtc_policy_is_emitting_even_when_horizon_collapses_to_one() -> None:
    """A flow-matching RTC policy is chunk-emitting by construction: it stays
    True through ``supports_rtc`` even when its RTC execution horizon is 1 and
    the base ``execution_horizon > 1`` check would say False."""
    pol = _unloaded()
    pol.actions_per_step = 1
    pol._rtc_enabled = True
    pol._rtc_execution_horizon = 1  # base check sees horizon==1 -> False

    assert pol.execution_horizon == 1
    assert pol.supports_rtc is True
    assert pol.is_chunk_emitting() is True


def test_molmoact2_by_name_is_emitting() -> None:
    """MolmoAct2 (``PreTrainedPolicy.name == 'molmoact2'``) is detected through
    ``_requires_action_chunk`` even with a single-step ``actions_per_step`` and
    RTC off, because its chunk is served via ``predict_action_chunk``."""

    class _NamedMolmoAct2:
        name = "molmoact2"

    pol = _unloaded()
    pol.actions_per_step = 1
    pol._policy = _NamedMolmoAct2()

    assert pol.execution_horizon == 1
    assert pol.supports_rtc is False
    assert pol.is_chunk_emitting() is True


def test_molmoact2_by_class_name_fallback_is_emitting() -> None:
    """A stubbed/mocked MolmoAct2 policy that does not set ``name`` is still
    detected via the ``MolmoAct2Policy`` class-name fallback."""

    class MolmoAct2Policy:  # class-name fallback path
        pass

    pol = _unloaded()
    pol.actions_per_step = 1
    pol._policy = MolmoAct2Policy()

    assert pol.is_chunk_emitting() is True


def test_single_step_non_molmoact_is_not_emitting() -> None:
    """A single-step, non-RTC, non-MolmoAct2 policy emits one action per
    inference: all three signals are False, so it stays on the synchronous
    loop."""

    class _PlainPolicy:
        name = "act"

    pol = _unloaded()
    pol.actions_per_step = 1
    pol._policy = _PlainPolicy()

    assert pol.execution_horizon == 1
    assert pol.supports_rtc is False
    assert pol._requires_action_chunk() is False
    assert pol.is_chunk_emitting() is False
