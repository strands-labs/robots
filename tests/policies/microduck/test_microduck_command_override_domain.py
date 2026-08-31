"""The two command overrides refuse a vector they cannot honor, before it lands.

:meth:`MicroduckPolicy.get_actions` recognises two per-call overrides that write
the command vector - ``command`` (wholesale) and ``target_velocity`` (the twist
slots). Both are caller-supplied numeric vectors, and neither was held to a
domain:

* ``target_velocity`` was written as ``self._command[:n] = tv[:n]`` with
  ``n = min(3, len(tv), len(command))``, so a width the method does not document
  was silently absorbed. A LONGER vector lost its tail; a SHORTER one wrote only
  the slots it covered, and this policy's command vector persists across ticks -
  so ``target_velocity=[0.3]`` left the previous tick's lateral and yaw
  components commanding the robot, under a reported success.
* A non-finite component in either override was assigned first and refused
  afterwards by :func:`~strands_robots.policies.microduck.observation.build_observation`,
  which names ``command`` and the assembled observation rather than the parameter
  the caller passed. The refusal also arrived too late to protect the state: the
  poisoned vector stayed in ``self._command`` for every later tick.
* A non-numeric ``command`` reached ``np.asarray(..., dtype=np.float32)`` and
  surfaced as a bare ``could not convert string to float``, naming neither this
  policy nor the parameter.

Two sibling providers reading this same well-known goal key already hold it to a
domain that names it - ``WBCPolicy._validate_velocity`` and the
``param_name="target_velocity"`` guard MotionBricks applies on both its
constructor and its per-call path - which is the convention these cells hold this
one to. The two-component ``target_velocity`` spelling is deliberately KEPT
(see :data:`TARGET_VELOCITY_WIDTHS`); the family's other readers require three
because their command is rebuilt per call rather than carried across ticks.

Everything here runs through an injected stub session, so no ``onnxruntime`` and
no weights are needed.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import math
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.microduck import MicroduckPolicy
from strands_robots.policies.microduck import policy as policy_mod
from strands_robots.policies.microduck.policy import TARGET_VELOCITY_WIDTHS
from tests.policies.microduck.test_microduck_policy import _obs_dict, _StubSession

#: The shared vector domain both overrides must consult. Named locally so these
#: cells state the rule rather than reading it back out of the code they grade.
_SHARED_VECTOR_DOMAIN = "finite_vector_error"

#: A twist the first tick establishes, so a partial write is observable as the
#: PREVIOUS tick's components surviving rather than as a zero.
_STALE_TWIST = (7.0, 8.0, 9.0)


def _policy() -> MicroduckPolicy:
    return MicroduckPolicy(session=_StubSession())


def _ticked() -> MicroduckPolicy:
    """A policy whose twist slots already hold :data:`_STALE_TWIST`."""
    p = _policy()
    asyncio.run(p.get_actions(_obs_dict(), "", target_velocity=list(_STALE_TWIST)))
    assert np.allclose(_cmd(p)[:3], _STALE_TWIST), "fixture did not establish the stale twist"
    return p


def _tick(p: MicroduckPolicy, **kwargs: Any) -> None:
    asyncio.run(p.get_actions(_obs_dict(), "", **kwargs))


def _cmd(p: MicroduckPolicy) -> np.ndarray:
    """The running command vector, narrowed once.

    ``_command`` is ``NDArray | None`` until ``_ensure_config`` has run; every
    cell here has ticked the policy at least once, so the narrowing is asserted
    here rather than at each read.
    """
    assert p._command is not None, "the policy has not been configured by a tick"
    return p._command


def _method_source() -> str:
    return textwrap.dedent(inspect.getsource(MicroduckPolicy._apply_command_kwargs))


def _recognised_overrides() -> set[str]:
    """Every kwarg name ``_apply_command_kwargs`` reads, derived from its source.

    Keyed on the read rather than on a list, so an override added later - the
    third command semantics the family's non-locomotion exports want, say -
    is held to the same rule the hour it lands instead of inheriting an
    exemption by being absent from a tuple here.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(_method_source())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                names.add(node.args[0].value)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                names.add(node.slice.value)
    return names


def _domain_guarded_params() -> set[str]:
    """The ``param_name`` each shared-domain call in the method reports under."""
    guarded: set[str] = set()
    for node in ast.walk(ast.parse(_method_source())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != _SHARED_VECTOR_DOMAIN:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                guarded.add(arg.value)
    return guarded


class TestPremises:
    """Facts these cells rest on. They hold before and after the fix."""

    def test_the_command_vector_persists_across_ticks(self):
        # This is why a PARTIAL write is a live command rather than a zero: the
        # slots the caller did not cover still carry the previous tick's request.
        p = _ticked()
        _tick(p)  # no override at all
        assert np.allclose(_cmd(p)[:3], _STALE_TWIST)

    def test_the_shared_domain_is_a_real_export(self):
        from strands_robots import utils

        assert callable(getattr(utils, _SHARED_VECTOR_DOMAIN, None))

    def test_the_documented_widths_are_exactly_two_and_three(self):
        assert TARGET_VELOCITY_WIDTHS == frozenset({2, 3})

    def test_the_method_recognises_both_overrides(self):
        # Non-vacuity for the derived cell below: it grades a set, and an empty
        # set would satisfy any subset rule.
        assert {"command", "target_velocity"} <= _recognised_overrides()

    def test_a_sibling_provider_holds_the_same_key_to_a_named_domain(self):
        # The convention these cells hold this provider to. If the family stops
        # validating the shared key, this premise fails and the reader learns the
        # convention moved rather than reading a rule with no basis.
        from strands_robots.policies.wbc.policy import WBCPolicy

        with pytest.raises(ValueError, match="target_velocity"):
            WBCPolicy._validate_velocity([float("nan"), 0.0, 0.0])


class TestAWidthTheMethodDoesNotDocumentIsRefused:
    @pytest.mark.parametrize("width", [0, 1, 4, 7])
    def test_an_undocumented_width_is_refused(self, width):
        p = _ticked()
        with pytest.raises(ValueError, match="target_velocity has"):
            _tick(p, target_velocity=[0.3] * width)

    @pytest.mark.parametrize("width", [1, 4])
    def test_the_refusal_names_the_count_and_both_accepted_spellings(self, width):
        p = _ticked()
        with pytest.raises(ValueError) as exc:
            _tick(p, target_velocity=[0.3] * width)
        text = str(exc.value)
        assert f"{width} component(s)" in text
        assert "[vx, vy, omega]" in text and "[vx, vy]" in text

    def test_a_scalar_is_refused_rather_than_read_as_one_component(self):
        p = _ticked()
        with pytest.raises(ValueError, match="target_velocity"):
            _tick(p, target_velocity=0.3)

    @pytest.mark.parametrize("width", [1, 4])
    def test_a_refused_width_leaves_the_running_command_untouched(self, width):
        p = _ticked()
        with pytest.raises(ValueError):
            _tick(p, target_velocity=[0.3] * width)
        assert np.allclose(_cmd(p)[:3], _STALE_TWIST)


class TestANonFiniteOverrideIsRefusedAtTheSeamThatNamesIt:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("param", ["command", "target_velocity"])
    def test_the_refusal_names_the_parameter_the_caller_passed(self, param, bad):
        p = _ticked()
        width = 13 if param == "command" else 3
        vec = [bad] + [0.0] * (width - 1)
        with pytest.raises(ValueError) as exc:
            _tick(p, **{param: vec})
        text = str(exc.value)
        assert f"'{param}'" in text, text
        assert "MicroduckPolicy" in text, text

    @pytest.mark.parametrize("param", ["command", "target_velocity"])
    def test_a_non_finite_override_does_not_reach_the_running_command(self, param):
        # The refusal used to arrive after the assignment, so a caller that
        # handled it and kept ticking carried nan in its command for the rest of
        # the rollout AND into every later episode.
        p = _ticked()
        width = 13 if param == "command" else 3
        with pytest.raises(ValueError):
            _tick(p, **{param: [float("nan")] + [0.0] * (width - 1)})
        assert np.allclose(_cmd(p)[:3], _STALE_TWIST)
        assert not np.isnan(_cmd(p)).any()

    @pytest.mark.parametrize("param", ["command", "target_velocity"])
    def test_a_non_numeric_override_is_named_rather_than_coerced(self, param):
        p = _ticked()
        width = 13 if param == "command" else 3
        with pytest.raises(ValueError) as exc:
            _tick(p, **{param: ["a"] * width})
        text = str(exc.value)
        assert f"'{param}'" in text, text
        assert "convert string to float" not in text, text

    def test_a_later_tick_still_works_after_a_refusal(self):
        p = _ticked()
        with pytest.raises(ValueError):
            _tick(p, target_velocity=[float("nan"), 0.0, 0.0])
        _tick(p, target_velocity=[0.1, 0.2, 0.3])
        assert np.allclose(_cmd(p)[:3], [0.1, 0.2, 0.3])


class TestTheDomainIsAskedOfEveryOverrideTheMethodReads:
    def test_every_recognised_override_reports_under_the_shared_domain(self):
        missing = sorted(_recognised_overrides() - _domain_guarded_params())
        assert not missing, (
            f"{sorted(_recognised_overrides())} are read from kwargs by "
            f"_apply_command_kwargs, but {missing} never reach "
            f"{_SHARED_VECTOR_DOMAIN}, so a non-numeric or non-finite component "
            f"there is either coerced by numpy or written into the command and "
            f"reported later against a name the caller never used."
        )

    def test_the_width_refusal_reads_the_shared_constant(self):
        # Not a restated literal: the docstring and the refusal both derive from
        # TARGET_VELOCITY_WIDTHS, so they cannot drift apart.
        source = _method_source()
        assert "TARGET_VELOCITY_WIDTHS" in source, source

    def test_the_docstring_documents_the_widths_it_enforces(self):
        doc = " ".join((MicroduckPolicy.get_actions.__doc__ or "").split())
        assert "[vx, vy, omega]" in doc and "[vx, vy]" in doc
        assert "TARGET_VELOCITY_WIDTHS" in doc


class TestWhatIsDeliberatelyUnchanged:
    """The documented spellings, and the meaning of the slots.

    Every cell here holds on the pre-fix code too. They are the boundary: the
    change refuses widths the method never documented and values it cannot
    honor, and it must not narrow what a locomotion caller may ask for.
    """

    def test_the_three_component_spelling_writes_all_three_slots(self):
        p = _ticked()
        _tick(p, target_velocity=[0.3, -0.2, 0.1])
        assert np.allclose(_cmd(p)[:3], [0.3, -0.2, 0.1])

    def test_the_two_component_spelling_leaves_omega_at_its_current_value(self):
        # Documented, and coherent because the command persists: "set vx and vy,
        # leave omega". Refusing it would narrow the documented surface.
        p = _ticked()
        _tick(p, target_velocity=[0.3, -0.2])
        assert np.allclose(_cmd(p)[:3], [0.3, -0.2, _STALE_TWIST[2]])

    def test_a_full_width_finite_command_override_still_replaces_the_vector(self):
        p = _ticked()
        _tick(p, command=[0.5] * 13)
        assert np.allclose(_cmd(p), [0.5] * 13)

    def test_the_command_width_refusal_still_names_its_source(self):
        p = _ticked()
        with pytest.raises(ValueError, match="from command_names"):
            _tick(p, command=[0.1] * 5)

    def test_a_zero_twist_is_a_legal_request(self):
        # The stand-in-place command for the locomotion exports; it must not be
        # mistaken for an absent override.
        p = _ticked()
        _tick(p, target_velocity=[0.0, 0.0, 0.0])
        assert np.allclose(_cmd(p)[:3], [0.0, 0.0, 0.0])

    def test_integer_components_are_accepted(self):
        p = _ticked()
        _tick(p, target_velocity=[1, 0, 0])
        assert np.allclose(_cmd(p)[:3], [1.0, 0.0, 0.0])

    def test_a_numpy_vector_is_accepted(self):
        p = _ticked()
        _tick(p, target_velocity=np.array([0.4, 0.0, 0.0], dtype=np.float32))
        assert np.allclose(_cmd(p)[:3], [0.4, 0.0, 0.0])

    def test_the_widths_the_method_documents_are_not_a_hardcoded_pair_here(self):
        # Non-vacuity for the boundary above: the accepted spellings are read
        # from the module, so widening TARGET_VELOCITY_WIDTHS widens these cells
        # rather than leaving them contradicting the code.
        assert all(math.isfinite(w) and w > 0 for w in TARGET_VELOCITY_WIDTHS)
        assert getattr(policy_mod, "TARGET_VELOCITY_WIDTHS") is TARGET_VELOCITY_WIDTHS
