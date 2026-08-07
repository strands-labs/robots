"""Value-domain contracts for the WBC per-call command overrides.

:meth:`WBCPolicy._resolve_command` builds the observation's command block from
four caller-supplied components. ONE of them was validated: ``target_velocity``
goes through :meth:`WBCPolicy._validate_velocity` (numeric sequence, at least
three entries, every component finite). The other three are the per-call
overrides of ``height_cmd``, ``freq_cmd`` and ``rpy_cmd`` - the three config
fields :meth:`WBCConfig.__post_init__` does validate - and each reached the
network raw:

* ``command[3] = float(height)`` - a ``nan`` height put a non-finite value in
  the frame the network is given; a numeric string raised from that ``float()``.
* ``command[4:7] = np.asarray(target_orientation)`` - same, per component, and a
  non-numeric sequence surfaced as NumPy's own ``could not convert string to
  float``, which names neither the kwarg nor the policy.
* ``command[4] = float(gait_frequency)`` (the gait variant's step-frequency
  slot) - :meth:`GaitClock.update` documents that ``freq`` "must be strictly
  positive" and refuses a non-positive one, but only once the block has been
  built and handed to it from inside :meth:`get_actions`, so the message named
  ``GaitClock.update`` rather than the parameter the caller supplied. ``True``
  and ``"0.75"`` were not refused at all - they were coerced to a silent 1.0 Hz
  and 0.75 Hz.

The command block is the observation's first ``command_dim`` entries, so a
non-finite component is not a wrong number in one slot: through the shipped
dense MLP it makes every one of the 15 joint targets non-finite. The docs table
for the config spellings states exactly that consequence ("A non-finite scale
poisons the observation frame the network is given") twelve lines above the
goal-kwargs table for their unguarded overrides.

These tests pin one contract: every source of a command component is validated
on the domain the config enforces for the field it overrides, so the spelling
documented to WIN the precedence contest cannot accept a value the spelling that
loses it would refuse. The step frequency is held to the gait clock's stricter
``> 0`` rule at all three of its sources, and the values that remain first-class
are pinned too, so the guard cannot creep into refusing a command a caller may
legitimately give.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import math
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.wbc import GaitClock, WBCConfig, WBCGaitPolicy, build_gait_frame
from strands_robots.policies.wbc.policy import WBCPolicy
from strands_robots.utils import finite_number_error, positive_finite_number_error

from .test_gait import _full_g1_obs, _gait_config

# Values no command component can be honored as: a non-finite one poisons the
# observation frame, and a non-real one raises from the ``float()`` that reads
# it. ``bool`` is in the list because ``float(True)`` is a silent 1.0 - the
# config refuses it for the same reason. ``None`` is NOT: it is how every one
# of these kwargs spells "not supplied", so it means the config default and is
# pinned as such below.
UNUSABLE_ANY_SIGN: list[Any] = [float("nan"), float("inf"), float("-inf"), True, False, "0.75", [0.75]]

# Additionally unusable as a step frequency: the gait clock divides by it.
UNUSABLE_AS_FREQUENCY: list[Any] = [*UNUSABLE_ANY_SIGN, 0, 0.0, -0.75]

# Signed quantities the config accepts for these fields, so the overrides must too.
USABLE_HEIGHTS: list[Any] = [0.74, 0.8, 0.0, -0.1, 1, np.float64(0.72)]
USABLE_FREQUENCIES: list[Any] = [0.75, 1.5, 1e-6, 3, np.float64(2.0)]


def _resolve(policy: Any, **kwargs: Any) -> Any:
    """Build a command block through one funnel.

    These tests deliberately supply values outside the declared kwarg types (a
    string where a ``float`` is documented, a list where a scalar is), which is
    the point - the runtime is what must refuse them. Splatting through one
    ``**kwargs: Any`` funnel states that intent once instead of scattering a
    per-call suppression over every case.
    """
    return policy._resolve_command(kwargs)


def _gait(**kwargs: Any) -> Any:
    """Construct a gait policy through one funnel (see :func:`_resolve`)."""
    return WBCGaitPolicy(allow_missing_models=True, **kwargs)


def _config(**kwargs: Any) -> WBCConfig:
    """Build a config through one funnel (see :func:`_resolve`)."""
    return WBCConfig(policy_path="policy.onnx", **kwargs)


class _DenseSession:
    """A seeded dense layer standing in for the shipped gait MLP.

    The point is propagation, not fidelity: every ONNX policy in this family is
    a dense network, so a single non-finite entry in the 570-wide input reaches
    all 15 outputs. A stub returning a CONSTANT action would hide exactly that.
    """

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self.weights = rng.normal(0.0, 0.02, (570, 15)).astype(np.float32)
        self.seen: list[np.ndarray] = []

    def get_inputs(self) -> list[Any]:
        class _In:
            name = "obs"

        return [_In()]

    def run(self, _outputs: Any, feeds: dict[str, Any]) -> list[np.ndarray]:
        obs = np.asarray(next(iter(feeds.values())), dtype=np.float32)
        self.seen.append(obs)
        return [(obs @ self.weights).astype(np.float32)]


def _tick(policy: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Run one real ``get_actions`` tick; return (network input, joint targets)."""
    session = _DenseSession()
    policy.policy_session = session
    with np.errstate(all="ignore"):
        actions = asyncio.run(policy.get_actions(_full_g1_obs(), "", **kwargs))
    return session.seen[-1], np.asarray(list(actions[0].values()), dtype=np.float64)


class TestAPerCallHeightOverrideSharesTheConfigDomain:
    """``height`` is the per-call override of ``height_cmd``: one field, one domain."""

    @pytest.mark.parametrize("policy_factory", [WBCPolicy, _gait], ids=["non_gait", "gait"])
    @pytest.mark.parametrize("value", UNUSABLE_ANY_SIGN, ids=repr)
    def test_an_unusable_height_is_refused_naming_the_kwarg(self, policy_factory: Any, value: Any) -> None:
        policy = policy_factory(allow_missing_models=True) if policy_factory is WBCPolicy else policy_factory()
        with pytest.raises(ValueError, match="height"):
            _resolve(policy, height=value)

    @pytest.mark.parametrize("value", UNUSABLE_ANY_SIGN, ids=repr)
    def test_the_override_refuses_exactly_what_the_config_field_refuses(self, value: Any) -> None:
        """Neither spelling of the field may accept what the other rejects."""
        config_refuses = finite_number_error(value, "height_cmd", "WBCConfig") is not None
        try:
            _resolve(WBCPolicy(allow_missing_models=True), height=value)
            override_refuses = False
        except ValueError:
            override_refuses = True
        assert override_refuses is config_refuses, f"verdicts differ for height={value!r}"

    @pytest.mark.parametrize("value", USABLE_HEIGHTS, ids=repr)
    def test_a_usable_height_still_reaches_the_command_block(self, value: Any) -> None:
        command, _ = _resolve(WBCPolicy(allow_missing_models=True), height=value)
        assert command[3] == pytest.approx(float(value))

    def test_none_means_not_supplied_and_keeps_the_config_default(self) -> None:
        """The one asymmetry with the config field, and it is deliberate.

        ``kwargs.get("height")`` returns ``None`` when the caller did not supply
        one, so ``None`` selects ``config.height_cmd`` rather than being a value
        the guard must refuse. The config field itself has no such spelling - it
        always holds a number - which is why the parity above excludes it.
        """
        policy = WBCPolicy(config=_config(height_cmd=0.71), allow_missing_models=True)
        assert _resolve(policy, height=None)[0][3] == pytest.approx(0.71)
        assert _resolve(policy)[0][3] == pytest.approx(0.71)
        assert finite_number_error(None, "height_cmd", "WBCConfig") is not None


class TestAPerCallOrientationOverrideSharesTheConfigDomain:
    """``target_orientation`` is the per-call override of ``rpy_cmd``."""

    @pytest.mark.parametrize("policy_factory", [WBCPolicy, _gait], ids=["non_gait", "gait"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), True, None], ids=repr)
    def test_an_unusable_component_is_refused_naming_its_index(self, policy_factory: Any, bad: Any) -> None:
        policy = policy_factory(allow_missing_models=True) if policy_factory is WBCPolicy else policy_factory()
        with pytest.raises(ValueError, match=r"target_orientation\[1\]"):
            _resolve(policy, target_orientation=[0.0, bad, 0.0])

    def test_a_non_numeric_sequence_is_refused_naming_the_kwarg(self) -> None:
        """Not with NumPy's ``could not convert string to float``, which names nothing."""
        with pytest.raises(ValueError, match="target_orientation must be a numeric sequence"):
            _resolve(WBCPolicy(allow_missing_models=True), target_orientation="abc")

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), True], ids=repr)
    def test_the_override_refuses_exactly_what_the_config_field_refuses(self, bad: Any) -> None:
        config_refuses = finite_number_error(bad, "rpy_cmd[0]", "WBCConfig") is not None
        try:
            _resolve(WBCPolicy(allow_missing_models=True), target_orientation=[bad, 0.0, 0.0])
            override_refuses = False
        except ValueError:
            override_refuses = True
        assert override_refuses is config_refuses, f"verdicts differ for rpy component {bad!r}"

    def test_a_usable_orientation_still_reaches_the_command_block(self) -> None:
        command, _ = _resolve(WBCPolicy(allow_missing_models=True), target_orientation=[0.05, -0.1, 0.2])
        assert command[4:7] == pytest.approx([0.05, -0.1, 0.2])

    def test_a_bool_component_is_refused_despite_the_float_coercion(self) -> None:
        """``np.asarray([0.0, True, 0.0], float)`` is ``[0., 1., 0.]``.

        The coercion would hide the bool the config refuses for ``rpy_cmd``, so
        the components are inspected before it.
        """
        assert np.asarray([0.0, True, 0.0], dtype=np.float64)[1] == 1.0
        with pytest.raises(ValueError, match=r"target_orientation\[1\]"):
            _resolve(WBCPolicy(allow_missing_models=True), target_orientation=[0.0, True, 0.0])

    def test_none_means_not_supplied_and_keeps_the_config_default(self) -> None:
        policy = WBCPolicy(config=_config(rpy_cmd=[0.01, 0.02, 0.03]), allow_missing_models=True)
        assert _resolve(policy, target_orientation=None)[0][4:7] == pytest.approx([0.01, 0.02, 0.03])


class TestEveryStepFrequencySourceSharesOneDomain:
    """Per-call kwarg, constructor default and ``config.freq_cmd`` - one rule.

    The precedence chain is documented as per-call > constructor > config, so a
    guard on only the config would sit behind both spellings that win.
    """

    @pytest.mark.parametrize("value", UNUSABLE_AS_FREQUENCY, ids=repr)
    def test_the_per_call_kwarg_is_refused_naming_the_kwarg(self, value: Any) -> None:
        with pytest.raises(ValueError, match="gait_frequency"):
            _resolve(_gait(), gait_frequency=value)

    @pytest.mark.parametrize("value", UNUSABLE_AS_FREQUENCY, ids=repr)
    def test_the_constructor_default_is_refused_naming_the_kwarg(self, value: Any) -> None:
        with pytest.raises(ValueError, match="gait_frequency"):
            _gait(gait_frequency=value)

    @pytest.mark.parametrize("value", [0, 0.0, -0.75], ids=repr)
    def test_the_config_field_is_refused_for_the_gait_layout(self, value: Any) -> None:
        """The third source: a config the gait clock could not honor either."""
        config = _config(freq_cmd=value, single_obs_dim=95, command_dim=8)
        with pytest.raises(ValueError, match="freq_cmd"):
            _gait(config=config)

    @pytest.mark.parametrize("value", UNUSABLE_AS_FREQUENCY, ids=repr)
    def test_all_three_sources_agree(self, value: Any) -> None:
        """No source may accept a step frequency another source refuses."""
        verdicts = {}
        try:
            _resolve(_gait(), gait_frequency=value)
            verdicts["per_call"] = "accepted"
        except ValueError:
            verdicts["per_call"] = "refused"
        try:
            _gait(gait_frequency=value)
            verdicts["constructor"] = "accepted"
        except ValueError:
            verdicts["constructor"] = "refused"
        if isinstance(value, float | int) and not isinstance(value, bool):
            try:
                _gait(config=_config(freq_cmd=value, single_obs_dim=95, command_dim=8))
                verdicts["config"] = "accepted"
            except ValueError:
                verdicts["config"] = "refused"
        assert len(set(verdicts.values())) == 1, f"sources disagree for {value!r}: {verdicts}"

    def test_none_means_not_supplied_at_both_the_kwarg_and_the_constructor(self) -> None:
        """``None`` selects the next source in the precedence chain, as documented."""
        config = _config(freq_cmd=0.9, single_obs_dim=95, command_dim=8)
        assert _resolve(_gait(config=config), gait_frequency=None)[0][4] == pytest.approx(0.9)
        assert _resolve(_gait(config=config, gait_frequency=None))[0][4] == pytest.approx(0.9)
        assert _resolve(_gait(config=config, gait_frequency=1.4))[0][4] == pytest.approx(1.4)

    @pytest.mark.parametrize("value", USABLE_FREQUENCIES, ids=repr)
    def test_a_usable_frequency_still_reaches_the_slot(self, value: Any) -> None:
        command, _ = _resolve(_gait(gait_frequency=value))
        assert command[4] == pytest.approx(float(value))
        per_call, _ = _resolve(_gait(), gait_frequency=value)
        assert per_call[4] == pytest.approx(float(value))

    def test_the_frequency_rule_is_deliberately_stricter_than_the_config_rule(self) -> None:
        """``WBCConfig`` cannot demand positivity on the gait variant's behalf.

        The non-gait 7-wide command block has no step-frequency slot, so a
        ``freq_cmd`` of ``0`` is inert for :class:`WBCPolicy` and the config's
        rule is finiteness. The gait variant reads the slot, so the positivity
        rule lives at the gait layer - which is also why the two rules must not
        be collapsed into one.
        """
        assert finite_number_error(0.0, "freq_cmd", "WBCConfig") is None
        assert positive_finite_number_error(0.0, "gait_frequency", "WBCGaitPolicy") is not None
        # A non-gait config with freq_cmd=0 stays constructible and usable.
        non_gait = WBCPolicy(config=_config(freq_cmd=0.0), allow_missing_models=True)
        command, _ = _resolve(non_gait)
        assert len(command) == 7, "the non-gait block has no step-frequency slot to poison"


class TestTheGaitClockDomainIsWhatTheGuardEnforces:
    """The guard is the clock's own documented rule, decided where the value arrives."""

    @pytest.mark.parametrize("value", [0.0, -0.75, float("nan"), float("inf")], ids=repr)
    def test_every_frequency_the_clock_refuses_is_refused_up_front(self, value: Any) -> None:
        with pytest.raises(ValueError, match="freq must be finite and > 0"):
            GaitClock().update(np.array([0.5, 0.0, 0.0]), value)
        with pytest.raises(ValueError, match="gait_frequency"):
            _resolve(_gait(), gait_frequency=value)

    def test_the_guard_also_refuses_what_the_clock_would_silently_coerce(self) -> None:
        """``True`` reaches the clock as a 1.0 Hz gait nobody asked for."""
        clock_output = GaitClock().update(np.array([0.5, 0.0, 0.0]), True)
        assert np.all(np.isfinite(clock_output)), "the clock accepts a bool as 1.0 Hz"
        with pytest.raises(ValueError, match="gait_frequency"):
            _resolve(_gait(), gait_frequency=True)


class TestARefusalPrecedesTheModelLoader:
    """A constructor-time refusal must not first load the ONNX session(s)."""

    def test_a_bad_constructor_frequency_never_reaches_the_loader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loads: list[str] = []

        def _fatal(self: Any, *args: Any, **kwargs: Any) -> Any:
            loads.append("loaded")
            raise AssertionError("the ONNX loader must not run for a refused frequency")

        monkeypatch.setattr(WBCPolicy, "_load_sessions", _fatal, raising=False)
        with pytest.raises(ValueError, match="gait_frequency"):
            _gait(gait_frequency=0.0)
        assert loads == []

    def test_a_usable_constructor_frequency_still_constructs(self) -> None:
        assert _gait(gait_frequency=1.5)._gait_frequency == pytest.approx(1.5)


class TestAnUnusableComponentNeverReachesTheNetwork:
    """Why the domain matters: one non-finite slot is 15 non-finite joint targets."""

    def test_a_non_finite_command_slot_poisons_every_joint_target(self) -> None:
        """The premise, measured on the command block directly."""
        config = _gait_config()
        command = np.zeros(config.command_dim, dtype=np.float64)
        command[3] = float("nan")  # the height slot, as an unguarded override wrote it
        frame = build_gait_frame(
            config,
            command=command,
            base_ang_vel=np.zeros(3),
            proj_gravity=np.array([0.0, 0.0, -1.0]),
            qj=np.zeros(29),
            dqj=np.zeros(29),
            prev_action=np.zeros(15),
            clock=np.zeros(2),
        )
        assert int((~np.isfinite(frame)).sum()) == 1, "one poisoned slot"
        session = _DenseSession()
        stacked = np.tile(frame, 6).reshape(1, 570).astype(np.float32)
        with np.errstate(all="ignore"):
            out = session.run(None, {"obs": stacked})[0]
        assert int((~np.isfinite(out)).sum()) == 15, "a dense layer spreads it to every target"

    @pytest.mark.parametrize(
        ("kwargs_", "pattern"),
        [
            ({"height": float("nan")}, "height"),
            ({"target_orientation": [float("inf"), 0.0, 0.0]}, r"target_orientation\[0\]"),
            ({"gait_frequency": 0.0}, "gait_frequency"),
        ],
        ids=["height", "target_orientation", "gait_frequency"],
    )
    def test_a_full_rollout_tick_refuses_instead_of_feeding_the_network(
        self, kwargs_: dict[str, Any], pattern: str
    ) -> None:
        with pytest.raises(ValueError, match=pattern):
            _tick(_gait(), **kwargs_)

    def test_a_fully_specified_usable_command_keeps_the_frame_finite(self) -> None:
        obs, targets = _tick(
            _gait(),
            target_velocity=[0.5, 0.0, 0.0],
            height=0.8,
            gait_frequency=1.5,
            target_orientation=[0.0, 0.05, 0.1],
        )
        assert obs.shape == (1, 570)
        assert np.all(np.isfinite(obs)), "a usable command must not be refused or poisoned"
        assert np.all(np.isfinite(targets)) and targets.size == 15


class TestEveryCallerSuppliedCommandComponentIsValidated:
    """Structural guard: a fifth component cannot be added without a domain.

    Both ``_resolve_command`` implementations read their caller-supplied
    components out of ``kwargs`` by name. Every such name must reach a
    ``_validate_*`` call in the same method, so a new goal kwarg cannot start
    flowing into the observation block unchecked.
    """

    @staticmethod
    def _components_and_validators(func: Any) -> tuple[set[str], set[str]]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        read: set[str] = set()
        validated: set[str] = set()
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            if isinstance(call.func, ast.Attribute):
                if call.func.attr == "get" and call.args and isinstance(call.args[0], ast.Constant):
                    read.add(str(call.args[0].value))
                elif call.func.attr.startswith("_validate_"):
                    validated.add(call.func.attr)
        return read, validated

    @pytest.mark.parametrize(
        "method", [WBCPolicy._resolve_command, WBCGaitPolicy._resolve_command], ids=["non_gait", "gait"]
    )
    def test_every_component_read_from_kwargs_has_a_validator(self, method: Any) -> None:
        read, validated = self._components_and_validators(method)
        assert read, "the scanner found no kwargs.get(...) component at all"
        expected = {
            "target_velocity": "_validate_velocity",
            "height": "_validate_height",
            "target_orientation": "_validate_orientation",
            "gait_frequency": "_validate_gait_frequency",
        }
        missing = {name for name in read if expected.get(name, "") not in validated}
        assert not missing, f"command components read without a domain: {sorted(missing)}"

    def test_the_scanner_would_notice_an_unguarded_component(self) -> None:
        """Non-vacuity: a planted component with no validator must be reported."""

        def _planted(self: Any, kwargs: dict[str, Any]) -> None:
            _ = kwargs.get("target_velocity")
            _ = kwargs.get("stride_length")  # no domain
            self._validate_velocity(_)

        read, validated = self._components_and_validators(_planted)
        assert read == {"target_velocity", "stride_length"}
        assert validated == {"_validate_velocity"}

    def test_both_implementations_are_the_real_ones(self) -> None:
        """Non-vacuity: the scan root must resolve to the shipped module."""
        for method in (WBCPolicy._resolve_command, WBCGaitPolicy._resolve_command):
            assert Path(inspect.getfile(method)).parent.name == "wbc"


class TestTheGuardsDoNotDependOnMathIsnanDirectly:
    """The overrides delegate to the shared numeric rule, not a local copy."""

    def test_the_height_and_orientation_rules_are_the_shared_domain(self) -> None:
        for value in UNUSABLE_ANY_SIGN:
            assert finite_number_error(value, "height", "WBCPolicy") is not None
        assert finite_number_error(0.0, "height", "WBCPolicy") is None
        assert not math.isnan(0.0)  # the sibling velocity rule still uses math directly
