# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: both owners of the parallel-environment count carry a domain.

An Isaac scene's environment count has two owners. ``IsaacConfig.num_envs`` is
the configured default; ``IsaacSimulation.replicate(num_envs=...)`` is the
per-call request that is honored *instead of* it. Neither was held to the shared
count domain (:func:`~strands_robots.utils.positive_count_error`), which the
sibling fields ``camera_width`` / ``camera_height`` in the same
``__post_init__`` already take, and which ``RLTrainSpec.num_envs`` -- the other
environment count in this library -- was moved onto for exactly this reason (see
``tests/training/test_rl_env_count_domain.py``, whose docstring works through
the same ``nan`` / ``inf`` / ``2.5`` / ``True`` values measured below).

Measured before this gate existed, over ``IsaacConfig(num_envs=8)``:

* The field's hand-rolled ``self.num_envs < 1`` tested only the floor, so it
  read ``True`` as a count of 1 while refusing ``False``, and accepted ``4.0``,
  ``2.7``, ``nan`` and ``inf`` to be stored and reported as an environment
  count. ``IsaacSimulation.__init__`` logs this field with ``%d``, so a stored
  ``2.7`` was announced as ``num_envs=2`` -- the log disagreeing with the config
  it describes -- and a stored ``nan`` or ``inf`` made that logging call itself
  raise. A ``str``, ``None`` or a list raised ``TypeError`` from the comparison,
  naming neither the field nor a remedy. Seven of the thirteen values these
  cells probe reached a verdict the shared domain does not.
* The argument had no domain at all and was read by truthiness
  (``num_envs or self._config.num_envs``), so every value in ``_REFUSED`` below
  was accepted alongside the counts in ``_ACCEPTED`` -- ten spellings no scene
  can replicate over, none refused. A supplied ``0`` or ``False`` was read as
  "not supplied" and replicated to the *configured* 8, announcing
  ``"Replicated to 8 environments"`` under ``status: "success"`` -- the caller's
  explicit request discarded with nothing saying so. Every truthy value was
  stored and reported three times over: by ``replicate``, by ``get_state``, and
  by ``destroy`` as ``num_envs_released``. ``num_envs="4"`` was the worst of
  them, rendering as ``"Replicated to 4 environments"`` -- byte-identical to
  what the int ``4`` produces -- while the payload carried the ``str``.

The count also latches: ``replicate`` sets ``_replicated``, and ``add_robot``
refuses once it is set, so an unusable count locked the scene until ``destroy``.

None of this needs Isaac Sim or a GPU. The domain is arithmetic, so the field
cells construct ``IsaacConfig`` alone and the argument cells drive the unbound
``replicate`` against the ``types.SimpleNamespace`` stand-in for ``self`` that
the neighbouring Isaac tests already use.
"""

from __future__ import annotations

import threading
import types
from typing import Any

import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.utils import positive_count_error

#: The configured default this scene carries, so a substituted count is
#: distinguishable from any requested one.
_CONFIGURED = 8

#: Counts that can be honored and must keep working. ``1024`` is the value the
#: Isaac page tells a fleet-RL caller to set, so it is the control that this is a
#: domain on the *kind* of value and not a cap.
_ACCEPTED = (1, 4, 1024)

#: ``value -> the reason it is not a count``. ``0`` and ``False`` are the
#: truthiness half: both are falsy, so both were read as "not supplied".
_REFUSED: dict[str, Any] = {
    "zero": 0,
    "negative": -1,
    "True reads as a flag, not a count of one": True,
    "False is falsy and was read as unstated": False,
    "an integral float still cannot index a range": 4.0,
    "a fractional count": 2.7,
    "not-a-number passes every ordered comparison": float("nan"),
    "an unbounded count": float("inf"),
    "the digits of a count are not a count": "4",
    "a sequence holding a count is not one": [4],
}


def _config(**kwargs: Any) -> IsaacConfig:
    """Construct through ``**Any`` so a deliberately off-type probe value type-checks.

    The point of these cells is values the annotation forbids, so they are handed
    over the way :meth:`IsaacConfig.from_kwargs` already accepts them rather than
    each call site carrying a suppression.
    """
    return IsaacConfig(**kwargs)


def _stub(configured: int = _CONFIGURED) -> types.SimpleNamespace:
    """A stand-in for ``self`` carrying only what ``replicate`` reads."""
    return types.SimpleNamespace(
        _lock=threading.RLock(),
        _world_created=True,
        _world=None,
        _config=IsaacConfig(num_envs=configured),
        _robots={"arm": object()},
        _objects={},
        _cameras={},
        _action_controllers={},
        _prim_registry=[],
        _replicated=False,
        _num_envs_active=1,
        _sim_time=0.0,
        _step_count=0,
    )


def _field_refusal(value: Any) -> str | None:
    """The configured owner's refusal for ``value``, read through the constructor."""
    try:
        _config(num_envs=value)
    except ValueError as exc:
        return str(exc)
    return None


def _argument_refusal(value: Any) -> str | None:
    """The requested owner's refusal for ``value``, read through ``replicate``."""
    result = IsaacSimulation.replicate(_stub(), num_envs=value)  # type: ignore[arg-type]
    if result["status"] == "error":
        return str(result["content"][0]["text"])
    return None


def test_the_probe_tables_are_populated() -> None:
    """A domain graded over an empty table would report success having read nothing."""
    assert len(_ACCEPTED) >= 3
    assert len(_REFUSED) >= 10
    assert all(reason.strip() for reason in _REFUSED)


class TestACountThatCanBeHonoredIsAccepted:
    """Both owners keep honoring the counts a caller actually replicates over."""

    @pytest.mark.parametrize("count", _ACCEPTED)
    def test_the_field_stores_it(self, count: int) -> None:
        assert IsaacConfig(num_envs=count).num_envs == count

    @pytest.mark.parametrize("count", _ACCEPTED)
    def test_the_request_is_honored_and_reported(self, count: int) -> None:
        stub = _stub()

        result = IsaacSimulation.replicate(stub, num_envs=count)  # type: ignore[arg-type]

        assert result["status"] == "success"
        assert result["content"][0]["json"]["num_envs"] == count
        assert stub._num_envs_active == count
        assert f"Replicated to {count} environments" in result["content"][0]["text"]


class TestACountThatCannotBeHonoredIsRefusedByBothOwners:
    """The same value reaches the same verdict whichever owner is handed it."""

    @pytest.mark.parametrize("value", list(_REFUSED.values()), ids=list(_REFUSED))
    def test_the_field_refuses_it(self, value: Any) -> None:
        refusal = _field_refusal(value)

        assert refusal is not None, f"IsaacConfig accepted num_envs={value!r} as an environment count"
        assert "num_envs" in refusal

    @pytest.mark.parametrize("value", list(_REFUSED.values()), ids=list(_REFUSED))
    def test_the_request_refuses_it(self, value: Any) -> None:
        refusal = _argument_refusal(value)

        assert refusal is not None, f"replicate accepted num_envs={value!r} as an environment count"
        assert "num_envs" in refusal

    @pytest.mark.parametrize("value", list(_REFUSED.values()), ids=list(_REFUSED))
    def test_neither_owner_raises_the_bare_comparison_TypeError(self, value: Any) -> None:
        """``'4' < 1`` named neither the field nor a remedy; a refusal names both."""
        assert _field_refusal(value) is not None
        assert _argument_refusal(value) is not None


class TestBothOwnersQuoteTheSharedDomain:
    """Routing is graded through the answer, so a hand-rolled copy cannot drift back."""

    @pytest.mark.parametrize("value", list(_REFUSED.values()), ids=list(_REFUSED))
    def test_the_field_message_is_the_domains_own(self, value: Any) -> None:
        assert _field_refusal(value) == positive_count_error(value, "num_envs", "IsaacConfig")

    @pytest.mark.parametrize("value", list(_REFUSED.values()), ids=list(_REFUSED))
    def test_the_request_message_is_the_domains_own(self, value: Any) -> None:
        assert _argument_refusal(value) == positive_count_error(value, "num_envs", "replicate")


class TestAnUnstatedCountStillTakesTheConfiguredOne:
    """``None`` is the one spelling of "not supplied", and it keeps its meaning."""

    def test_it_replicates_to_the_configured_count(self) -> None:
        stub = _stub()

        result = IsaacSimulation.replicate(stub, num_envs=None)  # type: ignore[arg-type]

        assert result["status"] == "success"
        assert result["content"][0]["json"]["num_envs"] == _CONFIGURED
        assert stub._num_envs_active == _CONFIGURED

    def test_the_default_is_reached_without_naming_the_argument(self) -> None:
        stub = _stub()

        assert IsaacSimulation.replicate(stub)["content"][0]["json"]["num_envs"] == _CONFIGURED  # type: ignore[arg-type]


class TestASuppliedZeroIsNotReadAsUnstated:
    """The truthiness half: a falsy count is the count asked for, not a silence."""

    @pytest.mark.parametrize("falsy", [0, False])
    def test_it_is_refused_rather_than_substituted(self, falsy: Any) -> None:
        stub = _stub()

        result = IsaacSimulation.replicate(stub, num_envs=falsy)  # type: ignore[arg-type]

        assert result["status"] == "error"
        assert f"Replicated to {_CONFIGURED} environments" not in result["content"][0]["text"]
        assert stub._num_envs_active != _CONFIGURED


class TestARefusedCountIsNotPartiallyApplied:
    """A refusal leaves the scene exactly as it was, replicable and extensible."""

    @pytest.mark.parametrize("value", list(_REFUSED.values()), ids=list(_REFUSED))
    def test_the_count_is_not_recorded(self, value: Any) -> None:
        stub = _stub()

        IsaacSimulation.replicate(stub, num_envs=value)  # type: ignore[arg-type]

        assert stub._num_envs_active == 1
        assert stub._replicated is False

    def test_the_scene_is_not_locked_against_further_robots(self) -> None:
        """``add_robot`` refuses once ``_replicated`` latches, so it must not latch."""
        stub = _stub()

        IsaacSimulation.replicate(stub, num_envs=2.7)  # type: ignore[arg-type]
        added = IsaacSimulation.add_robot(stub, "second", data_config="panda")  # type: ignore[arg-type]

        assert added["status"] == "success"

    def test_a_later_usable_request_still_replicates(self) -> None:
        stub = _stub()

        assert IsaacSimulation.replicate(stub, num_envs="4")["status"] == "error"  # type: ignore[arg-type]
        assert IsaacSimulation.replicate(stub, num_envs=4)["status"] == "success"  # type: ignore[arg-type]
        assert stub._num_envs_active == 4


class TestEachOwnerRefusesThroughItsOwnDocumentedChannel:
    """One domain, two channels: the field is a constructor, the request is a verb.

    The asymmetry is deliberate and is what each surface documents. ``IsaacConfig``
    is a dataclass, so an unusable field cannot be reported -- there is no return
    value -- and it raises ``ValueError`` like every other field in its
    ``__post_init__``. ``replicate`` documents a status dict as its channel and
    already answers its no-world and no-robot refusals that way, so it reports
    rather than raising.
    """

    def test_the_field_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="num_envs"):
            _config(num_envs=2.7)

    def test_the_request_reports_and_does_not_raise(self) -> None:
        result = IsaacSimulation.replicate(_stub(), num_envs=2.7)  # type: ignore[arg-type]

        assert result["status"] == "error"

    def test_the_request_channel_matches_its_sibling_refusals(self) -> None:
        """A no-robot scene is refused the same way, so the count is not special."""
        empty = _stub()
        empty._robots = {}

        assert IsaacSimulation.replicate(empty, num_envs=4)["status"] == "error"  # type: ignore[arg-type]
