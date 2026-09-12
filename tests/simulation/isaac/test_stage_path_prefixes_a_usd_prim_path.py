# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: both halves of an Isaac prim path carry a domain.

Every prim ``IsaacSimulation`` creates is addressed by a path interpolated from
two caller-supplied components -- ``f"{stage_path}/Robots/{name}"``, and the
same shape for ``/Objects/`` and ``/Cameras/``. The ``name`` half has a domain:
``add_robot`` refuses a name that cannot address the robot it creates on the
shared ``entity_name_error``, whose own docstring gives the reason in terms of
this very f-string. ``IsaacConfig.stage_path`` -- the other half of the same
string -- had none, so the values that half refuses by name were accepted
through this one, and the interpolated result was recorded in
``_prim_registry``, which is what ``destroy`` releases and counts:

* ``stage_path=None`` recorded ``None/Robots/arm`` -- the literal four
  characters -- ``stage_path=7`` recorded ``7/Robots/arm``, and
  ``stage_path=["/World"]`` recorded ``['/World']/Robots/arm``.
* ``stage_path="World"`` recorded the relative path ``World/Robots/arm``.
  ``get_body_state`` routes on exactly that distinction (``if
  body_name.startswith("/")`` takes the stage lookup; ``elif "/" in body_name``
  reads the value as ``robot_name/link_name``), so a caller handing back the
  path this backend recorded is routed to the wrong branch.
* ``stage_path="/World/"`` recorded ``/World//Robots/arm``: a doubled separator
  is not a path component, and a trailing separator is the likeliest way to
  write one, because the field is documented as a *prefix*.
* ``stage_path="/My World"`` and ``"/World\\x00x"`` were recorded verbatim. USD
  transcodes a prim name outside its identifier alphabet, so the prim does not
  land at the path recorded for it. That transcoding is not hypothetical here:
  ``demangle_usd_joint_names`` exists to undo it for joint names, and this
  package carries ``_tf_make_valid_identifier``, a clone of USD's own mangle,
  which is used below as the oracle for what each refused component becomes.

None of this needs Isaac Sim or a GPU. The domain is lexical, so most cells
construct ``IsaacConfig`` alone; the recorded-path cells drive the unbound
``add_robot`` against the ``types.SimpleNamespace`` stand-in for ``self`` that
the neighbouring Isaac prim-path tests already use, because the procedural
branch touches no stage.
"""

from __future__ import annotations

import dataclasses
import pathlib
import threading
import types

import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.joint_names import _tf_make_valid_identifier
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.utils import entity_name_error

#: Prefixes that address a prim and must keep working. ``/World`` is the
#: declared default; ``/World/Robots`` is unusual but is a real USD path, so it
#: is the control that the domain checks the path's *shape* and not a vocabulary
#: of blessed roots.
_ACCEPTED = ("/World", "/Root", "/World/Env_0", "/World/Robots", "/_private")

#: ``value -> the reason it cannot prefix a prim path``. Every reason is
#: asserted non-empty below, so a row cannot be added without stating why.
_REFUSED: dict[object, str] = {
    None: "not a str: the f-string renders it as the literal text 'None'",
    7: "not a str: renders as '7', an int the tool surface never sends",
    True: "not a str: a bool renders as 'True'",
    "": "empty: names no root at all, and is what entity_name_error calls unaddressable",
    "/": "the root names no component, and prefixing it yields '//Robots/<name>'",
    "World": "relative: get_body_state routes on the leading '/'",
    "World/Sub": "relative, and read as a '<robot>/<link>' pair instead",
    "/World/": "trailing separator leaves an empty final component",
    "/World//Sub": "doubled separator is not a path component",
    "/My World": "a space is outside USD's prim-name alphabet",
    "/World\x00x": "a NUL, which entity_name_error refuses in the name half by name",
    "/1World": "a leading digit is not a valid first identifier character",
    "/World-2": "a hyphen is outside USD's prim-name alphabet",
    "/World/bad name": "a nested component is checked too, not only the first",
}

#: The floor both halves of the f-string share. ``entity_name_error`` documents
#: exactly these three -- a non-``str``, the empty string, a string containing a
#: NUL -- so a value one half refuses must not be accepted by the other.
_SHARED_FLOOR = (None, 7, True, ["x"], "", "a\x00b")


def _refusal(stage_path: object) -> str | None:
    """The message ``IsaacConfig`` refuses ``stage_path`` with, or ``None``.

    Read through the public constructor rather than the domain helper, so these
    cells grade the behaviour a caller sees and stay valid if the check moves.
    """
    try:
        IsaacConfig(stage_path=stage_path)  # type: ignore[arg-type]
    except ValueError as e:
        return str(e)
    return None


def _stub(stage_path: str) -> types.SimpleNamespace:
    """A stand-in for ``self`` carrying only what ``add_robot`` reads."""
    return types.SimpleNamespace(
        _lock=threading.RLock(),
        _world_created=True,
        _world=None,
        _config=IsaacConfig(stage_path=stage_path),
        _robots={},
        _objects={},
        _cameras={},
        _action_controllers={},
        _replicated=False,
        _prim_registry=[],
    )


class TestAnAddressablePrefixIsAccepted:
    """The domain checks the path's shape, and shapes that address a prim pass."""

    @pytest.mark.parametrize("stage_path", _ACCEPTED)
    def test_it_constructs(self, stage_path):
        assert IsaacConfig(stage_path=stage_path).stage_path == stage_path

    @pytest.mark.parametrize("stage_path", _ACCEPTED)
    def test_the_recorded_prim_path_is_the_prefix_plus_the_name(self, stage_path):
        """The prefix reaches ``_prim_registry`` unchanged -- nothing is rewritten.

        Reached through ``usd_path=`` with the USD loader stood in. It used to go
        through ``data_config="panda"``, which needed no asset files because it
        resolved a *procedural builder* - a route that has since been deleted for
        reporting success while creating no prims at all. The prefix contract this
        grades is unchanged; only the way in is, and an explicit asset path is the
        more direct way to reach the registry write anyway.
        """
        stub = _stub(stage_path)
        stub._load_usd_robot = lambda prim_path, usd_path, position: (["j0"], None)

        result = IsaacSimulation.add_robot(stub, "arm", usd_path="/assets/arm.usda")  # type: ignore[arg-type]

        assert result["status"] == "success", result
        assert stub._prim_registry == [f"{stage_path}/Robots/arm"]


class TestAPrefixThatCannotAddressAPrimIsRefused:
    """Each refused spelling raises on construction, naming the field."""

    @pytest.mark.parametrize("value", list(_REFUSED))
    def test_it_raises_value_error(self, value):
        with pytest.raises(ValueError, match="IsaacConfig.stage_path"):
            IsaacConfig(stage_path=value)

    def test_every_refused_row_states_a_reason(self):
        """A row may not be added to the roster without saying why."""
        assert all(reason.strip() for reason in _REFUSED.values())

    @pytest.mark.parametrize("value", list(_REFUSED))
    def test_the_message_quotes_the_path_the_value_would_have_produced(self, value):
        """A refusal that does not show the resulting path does not say what is wrong.

        The path is quoted with ``!r`` so that a value whose damage is invisible
        in raw text -- a NUL, a trailing space -- is legible in the message.
        """
        message = _refusal(value)

        assert message is not None
        assert repr(f"{value}/Robots/<name>") in message


class TestTheRefusedPrefixesAreTheOnesThatWereRecorded:
    """The harm each refused spelling used to cause, stated as the recorded path."""

    @pytest.mark.parametrize(
        ("value", "recorded"),
        [
            (None, "None/Robots/arm"),
            (7, "7/Robots/arm"),
            (["/World"], "['/World']/Robots/arm"),
            ("", "/Robots/arm"),
            ("/", "//Robots/arm"),
            ("World", "World/Robots/arm"),
            ("/World/", "/World//Robots/arm"),
            ("/My World", "/My World/Robots/arm"),
        ],
    )
    def test_the_interpolation_is_what_reached_the_registry(self, value, recorded):
        """Pins the pre-fix behaviour: ``add_robot`` recorded this string.

        The interpolation itself is asserted rather than re-run through
        ``add_robot``, because ``add_robot`` can no longer be reached with these
        values -- the config refuses them first, which is the fix.
        """
        assert f"{value}/Robots/arm" == recorded

    @pytest.mark.parametrize("value", ["/My World", "/World\x00x", "/1World", "/World-2"])
    def test_usd_would_not_carry_the_recorded_path(self, value):
        """The component USD keeps differs from the one that was recorded.

        ``_tf_make_valid_identifier`` is this package's clone of USD's own
        mangle, already relied on by ``demangle_usd_joint_names``. A component
        it rewrites is a component the stage does not carry under the name
        ``_prim_registry`` recorded, so ``destroy`` would count a prim it cannot
        release.
        """
        component = value.split("/")[-1]

        assert _tf_make_valid_identifier(component) != component

    def test_a_component_usd_keeps_verbatim_is_accepted(self):
        """The control for the cell above: the rule is USD's, not a stricter one."""
        assert _tf_make_valid_identifier("Env_0") == "Env_0"
        assert IsaacConfig(stage_path="/World/Env_0").stage_path == "/World/Env_0"


class TestBothHalvesOfThePrimPathShareOneFloor:
    """A value the name half refuses is not accepted through the prefix half.

    The prim path is one string built from two components. Grading the two
    domains against each other is what keeps them from drifting apart again:
    ``entity_name_error`` documents a non-``str``, the empty string and an
    embedded NUL as unaddressable, and gives ``{stage_path}/Robots/{name}`` as
    the reason, so the prefix cannot accept what the name refuses.
    """

    @pytest.mark.parametrize("value", _SHARED_FLOOR)
    def test_the_name_half_refuses_it(self, value):
        assert entity_name_error("add_robot", "name", value) is not None

    @pytest.mark.parametrize("value", _SHARED_FLOOR)
    def test_the_prefix_half_refuses_it_too(self, value):
        # A NUL arrives inside a component when it arrives through the prefix.
        candidate = f"/{value}" if isinstance(value, str) and value else value

        assert _refusal(candidate) is not None

    def test_the_identifier_rule_is_the_prefix_halfs_alone(self):
        """A deliberate asymmetry, recorded so it reads as a decision.

        ``stage_path`` means one thing -- a USD prim path -- and has one
        consumer package, so it takes USD's whole prim-name rule.
        ``entity_name_error`` is shared with the MuJoCo and Newton backends,
        whose entity names are not USD identifiers, so widening it to the
        identifier alphabet is a cross-backend decision this domain does not
        make. A name outside the alphabet stays accepted there.
        """
        assert entity_name_error("add_robot", "name", "My World") is None
        assert _refusal("/My World") is not None


class TestTheDomainIsReachedThroughEveryConstructionDoor:
    """No door onto ``stage_path`` skips the domain."""

    def test_the_dataclass_constructor(self):
        with pytest.raises(ValueError, match="IsaacConfig.stage_path"):
            IsaacConfig(stage_path="/World/")

    def test_from_kwargs(self):
        with pytest.raises(ValueError, match="IsaacConfig.stage_path"):
            IsaacConfig.from_kwargs(stage_path="World")

    def test_dataclasses_replace_on_a_valid_config(self):
        """``IsaacSimulation.__init__`` merges shortcut kwargs this way."""
        with pytest.raises(ValueError, match="IsaacConfig.stage_path"):
            dataclasses.replace(IsaacConfig(), stage_path="/World//Sub")

    def test_the_simulation_shortcut_kwarg(self):
        with pytest.raises(ValueError, match="IsaacConfig.stage_path"):
            IsaacSimulation(stage_path="/My World")

    def test_a_path_object_is_refused_although_it_rendered_correctly(self):
        """The one spelling this widens: a ``PurePosixPath`` used to work.

        ``f"{PurePosixPath('/World')}"`` renders ``/World``, so the recorded
        path was right and only the declared type was wrong. It is refused for
        the same reason the name half refuses a non-``str`` name that renders
        fine: the field is annotated ``str``, and admitting one non-``str`` that
        happens to render correctly is what admits ``None``, whose rendering is
        the literal text ``None``.
        """
        assert f"{pathlib.PurePosixPath('/World')}/Robots/arm" == "/World/Robots/arm"

        with pytest.raises(ValueError, match="must be a str"):
            IsaacConfig(stage_path=pathlib.PurePosixPath("/World"))
