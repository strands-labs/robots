# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: a camera mount the backend lacks is reported, not a TypeError.

``parent_body`` mounts a camera ON a moving body so a wrist view rides with the
arm. Two of the three backends implement it; Isaac does not. Until this fix Isaac
did not *declare* the parameter either, so the gap was answered by Python rather
than by the backend - measured on this tree, ``IsaacSimulation.add_camera`` had no
``parent_body`` in its signature and no ``**kwargs``, so::

    sim.add_camera(name="wrist", parent_body="so101/gripper")
    TypeError: IsaacSimulation.add_camera() got an unexpected keyword argument 'parent_body'

That names the parameter and nothing else: not that mounting is the capability at
stake, not that two sibling backends do mount, and not that omitting it yields a
world-fixed camera which Isaac does support. It also arrives as a ``TypeError``
out of a method whose whole contract is the ``{"status", "content"}`` envelope.

The call above is not hypothetical - it is the *documented remedy*.
``docs/policies/camera-naming.md`` prescribes it, backend-agnostically, as the
first of "Two ways to satisfy the check" for a VLA whose model card declares an
``observation.images.wrist_image`` feature, and ``README.md`` states the rule
("Wrist cameras mount on a body"). So the guidance a caller follows to make a
manipulation VLA see its wrist stream cannot be followed on one of the three
backends, and the failure points at neither the capability nor the alternative.

The fix is the one this repository already applies to the same situation one
method away. ``IsaacSimulation.add_robot`` *declares* ``mjcf_path`` - a loader
that backend genuinely lacks - and refuses it with a structured error naming the
reason and the backend that has it ("Convert the MJCF to URDF/USD and pass
urdf_path/usd_path, or use create_simulation(backend='mujoco') to load MJCF").
``add_camera`` now answers ``parent_body`` the same way, in the same position:
before the lock and before validating the parameters the call would not use, so
the answer does not depend on the world state.

This reports the gap; it does not close it. Isaac *can* parent prims, so mounting
is unimplemented rather than impossible, and implementing it needs a stage and an
Isaac Sim runtime to verify against. That is tracked separately - the refusal is
what makes the unimplemented capability visible in the meantime, and it names the
two backends that do mount so a caller is never stuck.

No test here needs Isaac Sim, a GPU or GL: the refusal runs before the lock and
before anything touches the stage, so the ``__new__`` skeleton reaches it, and the
MuJoCo/Newton halves reuse the world-fixed patterns their own ``add_camera``
tests already use (``add_camera`` compiles the spec but renders nothing).
"""

from __future__ import annotations

import inspect
import pathlib
import threading
import types
from typing import Any

import pytest

from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
from strands_robots.simulation.newton.simulation import NewtonSimEngine

#: The backends whose ``add_camera`` mounts a camera on a body. The Isaac refusal
#: must name every one of them, or a caller told "not supported here" still has to
#: go and find where it *is* supported.
_MOUNTING_BACKENDS = ("mujoco", "newton")

#: Mount points a caller plausibly passes - the namespaced ``<robot>/<body>`` form
#: ``list_bodies`` returns, which is what the docs prescribe.
_MOUNTS = ("so101/gripper", "arm/gripper", "panda/hand")


def _isaac_skeleton() -> Any:
    """An ``IsaacSimulation`` that has not run ``__init__``.

    The ``parent_body`` refusal is answered before ``with self._lock``, so it
    reads no instance state at all. Deliberately the *worst* case: an instance
    with no attributes proves the answer cannot depend on the world, the stage or
    the camera registry - which is the placement property under test.
    """
    return IsaacSimulation.__new__(IsaacSimulation)


def _isaac_add_camera(**kwargs: Any) -> dict[str, Any]:
    """Drive Isaac's ``add_camera`` on the bare skeleton.

    A funnel so the deliberately off-nominal ``self`` is stated once rather than
    at every call site.
    """
    return IsaacSimulation.add_camera(_isaac_skeleton(), **kwargs)


def _isaac_past_the_guard(**kwargs: Any) -> dict[str, Any]:
    """Drive Isaac's ``add_camera`` on a skeleton that can reach the world check.

    The bare skeleton cannot: ``with self._lock`` needs an attribute it has no
    reason to carry, and that asymmetry is itself the placement evidence. The
    refusal above answers an instance with *no* attributes, while a call the
    guard lets through immediately needs the lock - so the guard provably sits
    before it. This skeleton adds only the lock and a world that was never
    created, so a call past the guard lands on "No world created".
    """
    skeleton = _isaac_skeleton()
    skeleton._lock = threading.RLock()
    skeleton._world_created = False
    return IsaacSimulation.add_camera(skeleton, **kwargs)


def _newton_stub() -> Any:
    """A stand-in for ``self`` carrying only what Newton's ``add_camera`` reads.

    The three attributes the sibling
    ``tests/simulation/newton/test_add_camera_numeric_validation.py`` establishes,
    so the accepted path runs without the optional ``newton`` / ``warp`` packages.
    """
    return types.SimpleNamespace(
        _world=types.SimpleNamespace(cameras={}),
        _model=types.SimpleNamespace(body_label=["ground", "so101/gripper"]),
        _lock=threading.RLock(),
    )


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []))


class TestIsaacReportsTheMountItLacks:
    """The refusal names the capability, the reason and where it is supported."""

    @pytest.mark.parametrize("mount", _MOUNTS)
    def test_a_mount_request_is_refused_through_the_envelope(self, mount: str) -> None:
        result = _isaac_add_camera(name="wrist", parent_body=mount)
        assert result["status"] == "error", result
        # Through the documented envelope - not as a TypeError, which is what the
        # undeclared parameter produced.
        assert isinstance(result.get("content"), list)

    def test_the_refusal_quotes_the_value_and_names_the_parameter(self) -> None:
        text = _text(_isaac_add_camera(name="wrist", parent_body="so101/gripper"))
        assert "parent_body" in text
        assert "so101/gripper" in text
        assert "add_camera" in text

    def test_the_refusal_names_every_backend_that_does_mount(self) -> None:
        """A caller turned away here must be told where the capability lives.

        This is the half a bare ``TypeError`` could never carry, and the reason
        the refusal is worth more than the exception it replaces.
        """
        text = _text(_isaac_add_camera(name="wrist", parent_body="so101/gripper"))
        for backend in _MOUNTING_BACKENDS:
            assert backend in text, f"{backend} not named in: {text}"

    def test_the_refusal_names_the_world_fixed_alternative(self) -> None:
        """Omitting the mount is supported here, so the refusal says so."""
        text = _text(_isaac_add_camera(name="wrist", parent_body="so101/gripper"))
        assert "Omit" in text
        assert "world-fixed" in text


class TestTheRefusalDoesNotDependOnTheWorld:
    """Guard placement: the capability answer precedes every other check.

    Mirrors ``add_robot``, which answers ``mjcf_path`` at ``body[2]`` - before its
    lock and before ``coerce_pose_vector``. A caller asking for a capability the
    backend does not have should hear that, not "No world created" and not a
    complaint about a pose the call would never use.
    """

    def test_it_precedes_the_world_check(self) -> None:
        text = _text(_isaac_add_camera(name="wrist", parent_body="so101/gripper"))
        assert "parent_body" in text
        assert "No world created" not in text

    def test_it_precedes_the_pose_validation(self) -> None:
        """A mount request with an unusable pose still reports the mount."""
        text = _text(_isaac_add_camera(name="wrist", parent_body="so101/gripper", position=[float("nan"), 0.0, 0.0]))
        assert "parent_body" in text
        assert "position" not in text

    def test_it_precedes_the_name_validation(self) -> None:
        """A mount request with an unusable name still reports the mount."""
        text = _text(_isaac_add_camera(name="", parent_body="so101/gripper"))
        assert "parent_body" in text


class TestOmittingTheMountIsUnaffected:
    """Over-reach control: the world-fixed camera Isaac does support still works.

    Without these, "refuse ``parent_body``" would be satisfied by a guard that
    refused every call.
    """

    def test_omitting_it_passes_the_guard(self) -> None:
        """The call reaches the *world* check, so the guard let it by."""
        text = _text(_isaac_past_the_guard(name="front", position=[1.0, 0.0, 0.5]))
        assert "No world created" in text
        assert "parent_body" not in text

    def test_an_explicit_none_passes_the_guard(self) -> None:
        """``None`` is the documented default and means "world-fixed"."""
        text = _text(_isaac_past_the_guard(name="front", parent_body=None))
        assert "No world created" in text
        assert "parent_body" not in text

    def test_the_refusal_needs_less_state_than_the_accepted_path(self) -> None:
        """Placement, stated as a measurement.

        A refused mount is answered on an instance carrying no attributes at all.
        The same call with the mount omitted cannot be: it reaches
        ``with self._lock`` and raises for the missing attribute. So the guard is
        strictly before the lock - the ``add_robot``/``mjcf_path`` position.
        """
        refused = _isaac_add_camera(name="wrist", parent_body="so101/gripper")
        assert refused["status"] == "error"
        assert "parent_body" in _text(refused)

        with pytest.raises(AttributeError, match="_lock"):
            _isaac_add_camera(name="wrist")


class TestTheMountingBackendsStillMount:
    """No regression: the two backends that implement the mount still accept it."""

    def test_newton_still_mounts(self) -> None:
        result = NewtonSimEngine.add_camera(
            _newton_stub(), name="wrist", parent_body="so101/gripper", position=[0.0, 0.0, 0.05]
        )
        assert result["status"] == "success", result

    def test_mujoco_still_mounts(self) -> None:
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine as Engine

        sim = Engine(tool_name="mount_parity_sim", mesh=False)
        try:
            assert sim.create_world()["status"] == "success"
            assert sim.add_robot(name="so101")["status"] == "success"
            result = sim.add_camera(
                name="wrist", parent_body="so101/gripper", position=[0.0, 0.0, 0.05], target=[0.0, 0.0, 0.1]
            )
            assert result["status"] == "success", result
            assert "so101/gripper" in _text(result)
        finally:
            sim.cleanup()


class TestEveryBackendDeclaresTheMount:
    """The parameter is on all three signatures, so no caller gets a TypeError.

    Declaring it is what moves the answer from Python's argument binding into the
    backend, where it can name the capability and the alternative. A backend that
    silently dropped it would be worse than the ``TypeError``; a backend that does
    not declare it cannot answer at all.

    The ``engine`` parameter is typed ``Any`` because ``add_camera`` is not on the
    ``SimEngine`` ABC - the abstract set is ``add_object`` / ``add_robot`` /
    ``create_world`` / ``step`` / ... and ``add_camera`` is defined independently
    by each backend. That absence is exactly why the three signatures could drift,
    so there is no common base to annotate against.
    """

    @pytest.mark.parametrize(
        ("backend", "engine"),
        [("mujoco", MuJoCoSimEngine), ("newton", NewtonSimEngine), ("isaac", IsaacSimulation)],
    )
    def test_the_parameter_is_declared(self, backend: str, engine: Any) -> None:
        params = inspect.signature(engine.add_camera).parameters
        assert "parent_body" in params, f"{backend}: {sorted(params)}"

    @pytest.mark.parametrize(
        ("backend", "engine"),
        [("mujoco", MuJoCoSimEngine), ("newton", NewtonSimEngine), ("isaac", IsaacSimulation)],
    )
    def test_omitting_it_is_the_world_fixed_default(self, backend: str, engine: Any) -> None:
        """Every backend defaults it to ``None`` - a world-fixed camera."""
        assert inspect.signature(engine.add_camera).parameters["parent_body"].default is None


class TestTheDocumentedRemedySaysWhichBackendsMount:
    """The backend-agnostic guidance that prescribes the mount carries the caveat.

    ``docs/policies/camera-naming.md`` is a *policy* document - it applies to any
    simulation backend - and its first remedy is a ``parent_body`` call. Without a
    caveat it reads as universally available, which is how a reader arrives at the
    refusal above.
    """

    @staticmethod
    def _doc() -> str:
        root = pathlib.Path(inspect.getfile(IsaacSimulation)).parents[3]
        return (root / "docs" / "policies" / "camera-naming.md").read_text(encoding="utf-8")

    def test_the_doc_still_prescribes_the_mount(self) -> None:
        """Non-vacuity: the caveat is about a remedy the doc really gives."""
        assert "parent_body" in self._doc()

    def test_the_prescription_names_the_backends_that_support_it(self) -> None:
        doc = self._doc()
        idx = doc.index("parent_body")
        window = " ".join(doc[idx : idx + 900].split())
        for backend in _MOUNTING_BACKENDS:
            assert backend in window, f"{backend} not named near the prescription: {window[:400]}"
