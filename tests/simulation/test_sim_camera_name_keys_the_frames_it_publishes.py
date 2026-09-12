# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: ``add_camera`` refuses a name its own frames cannot travel under.

A simulation camera's name is not only a registry key. When the world is on the
mesh, every frame it renders is published on ``strands/<peer_id>/camera/<name>``
and, with the IoT offload enabled, written to
``<prefix>/<peer_id>/<name>/<ts>.jpg`` -- so the name is read as transport
structure by both consumers. Until this fix ``add_camera`` accepted any string:
measured on this tree (zenoh 1.10.1, one ``create_world``), ``'**'``, ``'*'``,
``'a$b'``, ``'cam#1'``, ``'cam?1'``, ``'a+b'``, ``'a//b'``, ``'/wrist'``,
``'..'``, ``'.'`` and ``'sub/../etc'`` all returned ``status="success"``,
registered, and compiled into the model.

What each one then did to a consumer is the subject of
``TestTheConsumerThatReadsTheNameAsStructure``: a wildcard is routed by
intersection, so one camera's frames reach a peer subscribed to another; a
character Zenoh forbids makes the topic unpublishable, and the publish loop logs
that at debug, so the camera renders forever and never reaches the mesh; a
``..`` segment walks the object key out of the peer's own prefix.

The rule is deliberately narrower than
:func:`~strands_robots.utils.camera_token_error`, which the hardware ``cameras``
mapping applies. A sim camera name carries the backend's own namespacing --
``arm0/wrist_cam`` is the form ``docs/recording.md`` documents recording under --
so ``/`` is structure the backend put there and cannot be refused, and a name
with a space or an inner dot keys its frames intact. Only what no consumer can
carry is refused; ``TestTheStricterDoorStaysAStrictSuperset`` pins that
relationship in both directions.

These tests need no GL: ``add_camera`` compiles the spec but renders nothing, and
the Newton and Isaac halves need neither optional package because the guard runs
before the method touches a solver (the unbound-method stand-in pattern
``tests/simulation/newton/test_add_camera_numeric_validation.py`` uses).
"""

from __future__ import annotations

import posixpath
import threading
import types
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import camera_frame_key_error, camera_token_error

# One row per reserved character or segment, with the phrase the refusal must
# answer it with -- not one row per spelling of the same fault.
UNUSABLE = [
    pytest.param("*", "must not contain '*'", id="zenoh-wildcard"),
    pytest.param("**", "must not contain '*'", id="zenoh-multi-wildcard"),
    pytest.param("arm0/**", "must not contain '*'", id="wildcard-under-a-namespace"),
    pytest.param("a$b", "must not contain '$'", id="zenoh-verbatim-wildcard-marker"),
    pytest.param("cam#1", "must not contain '#'", id="mqtt-multi-wildcard"),
    pytest.param("cam?1", "must not contain '?'", id="zenoh-forbidden-character"),
    pytest.param("a+b", "must not contain '+'", id="mqtt-single-level-wildcard"),
    pytest.param("..", "'..' path segment", id="parent-of-the-peer-prefix"),
    pytest.param(".", "'.' path segment", id="the-peer-prefix-itself"),
    pytest.param("sub/../etc", "'..' path segment", id="parent-inside-a-namespaced-name"),
    pytest.param("a//b", "empty path segment", id="empty-topic-level"),
    pytest.param("/wrist", "empty path segment", id="leading-separator"),
    pytest.param("wrist/", "empty path segment", id="trailing-separator"),
]

#: Names that must keep working. The last three are the point of the narrower
#: rule: the namespaced form the backend itself produces, and two names the
#: stricter ``cameras``-mapping door refuses on style while every frame consumer
#: carries them unchanged.
USABLE = ["wrist", "front_cam", "cam-2", "0", "arm0/wrist_cam", "a b", "wrist.rgb"]


class TestTheDomain:
    """The rule, read directly, before any backend applies it."""

    @pytest.mark.parametrize(("name", "phrase"), UNUSABLE)
    def test_a_name_no_consumer_could_carry_is_refused(self, name: str, phrase: str) -> None:
        message = camera_frame_key_error("add_camera", "name", name)
        assert message is not None, name
        assert phrase in message, message
        assert repr(name) in message, message

    @pytest.mark.parametrize("name", USABLE)
    def test_a_name_that_keys_its_frames_intact_is_accepted(self, name: str) -> None:
        assert camera_frame_key_error("add_camera", "name", name) is None

    @pytest.mark.parametrize("name", [None, 7, "", ["wrist"]])
    def test_a_value_that_is_no_name_at_all_is_left_to_the_addressability_domain(self, name: Any) -> None:
        """A non-name is refused for not being one, not for its punctuation."""
        message = camera_frame_key_error("add_camera", "name", name)
        assert message is not None
        assert "must not contain" not in message, message

    def test_a_hostile_str_subclass_is_judged_without_running_its_code(self) -> None:
        """The guard reads the name with a pattern, never with the caller's methods.

        ``entity_name_error`` accepts a ``str`` subclass - it is a string by every
        operation the registry performs - so the segment rule cannot be spelled
        ``name.split("/")``: that runs an override while judging the value, which
        is the read ``tests/test_refusal_messages_never_raise.py`` scans for.
        """

        class Hostile(str):
            def split(self, *args: Any, **kwargs: Any) -> list[str]:
                raise RuntimeError("the caller's code ran")

        assert camera_frame_key_error("add_camera", "name", Hostile("a/../b")) is not None
        assert camera_frame_key_error("add_camera", "name", Hostile("arm0/wrist_cam")) is None

    @pytest.mark.parametrize(("name", "_phrase"), UNUSABLE)
    def test_the_message_names_the_method_and_stays_ascii(self, name: str, _phrase: str) -> None:
        message = camera_frame_key_error("add_camera", "name", name)
        assert message is not None
        assert message.startswith("add_camera: name=")
        message.encode("ascii")


class TestTheStricterDoorStaysAStrictSuperset:
    """Two doors, one fact, and a stated difference between their domains.

    The ``cameras`` mapping's key is a label the caller invents, so that door
    requires a bare token; a sim camera's name carries the backend's namespacing,
    so this one cannot. Pinning both directions is what keeps the pair from
    drifting into two unrelated alphabets.
    """

    @pytest.mark.parametrize(("name", "_phrase"), UNUSABLE)
    def test_the_token_rule_refuses_everything_this_rule_refuses(self, name: str, _phrase: str) -> None:
        assert camera_token_error("Robot(cameras=...)", "camera name", name) is not None

    @pytest.mark.parametrize("name", ["arm0/wrist_cam", "a b", "wrist.rgb"])
    def test_and_refuses_more_than_it_does(self, name: str) -> None:
        """The names that make the two domains genuinely different."""
        assert camera_token_error("Robot(cameras=...)", "camera name", name) is not None
        assert camera_frame_key_error("add_camera", "name", name) is None


@pytest.fixture
def sim():
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    s = Simulation(tool_name="test_sim_camera_frame_key", mesh=False)
    assert s.create_world()["status"] == "success"
    yield s
    s.cleanup()


def _compiled_camera_names(sim) -> list[str]:
    import mujoco

    model = sim._world._model
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]


def _newton_add_camera(name: Any, **kwargs: Any) -> dict[str, Any]:
    """Call Newton's unbound ``add_camera`` with a stand-in for ``self``.

    The guard runs before the method touches a solver, so the stand-in carries
    only what the body reads and neither ``newton`` nor ``warp`` is needed.
    """
    stub = types.SimpleNamespace(
        _world=types.SimpleNamespace(cameras={}),
        _model=types.SimpleNamespace(body_label=("ground", "ball")),
        _lock=threading.RLock(),
    )
    request: dict[str, Any] = {"position": [1.0, 1.0, 1.0], "target": [0.0, 0.0, 0.0]}
    request.update(kwargs)
    result = NewtonSimEngine.add_camera(cast(NewtonSimEngine, stub), name, **request)
    return {**result, "registry": dict(stub._world.cameras)}


def _isaac_engine():
    from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation

    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = IsaacConfig()
    engine._lock = threading.RLock()
    engine._world = None
    engine._world_created = True
    engine._robots = {}
    engine._objects = {}
    engine._cameras = {}
    engine._prim_registry = []
    engine._cam_out_size = {}
    engine._camera_warmup_steps = 0
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._main_tid = threading.get_ident()
    engine._create_camera_prim = lambda **kwargs: (object(), 24.0)  # type: ignore[method-assign]
    return engine


class TestEveryBackendAppliesIt:
    """One request, one verdict, one sentence, whichever engine is held.

    Isaac is included even though it opts out of the free-camera routing rule:
    that rule follows from what *its* renderer resolves, and this one follows
    from what the mesh publishes, which is the same for all three.
    """

    @pytest.mark.parametrize(("name", "phrase"), UNUSABLE)
    def test_mujoco_refuses_and_creates_nothing(self, sim, name: str, phrase: str) -> None:
        result = sim.add_camera(name=name, position=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.0])
        assert result["status"] == "error", (name, result)
        assert phrase in result["content"][0]["text"]
        assert name not in sim._world.cameras
        assert name not in _compiled_camera_names(sim)

    @pytest.mark.parametrize(("name", "phrase"), UNUSABLE)
    def test_newton_refuses_with_the_same_sentence(self, sim, name: str, phrase: str) -> None:
        newton = _newton_add_camera(name)
        mujoco_result = sim.add_camera(name=name, position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
        assert newton["status"] == "error"
        assert newton["registry"] == {}
        assert newton["content"][0]["text"] == mujoco_result["content"][0]["text"]

    @pytest.mark.parametrize(("name", "phrase"), UNUSABLE)
    def test_isaac_refuses_it_too(self, name: str, phrase: str) -> None:
        engine = _isaac_engine()
        result = engine.add_camera(name, position=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.0])
        assert result["status"] == "error", (name, result)
        assert phrase in result["content"][0]["text"]
        assert engine._cameras == {}

    @pytest.mark.parametrize("name", USABLE)
    def test_a_usable_name_still_registers_on_every_backend(self, sim, name: str) -> None:
        result = sim.add_camera(name=name, position=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.0])
        assert result["status"] == "success", (name, result)
        assert name in sim._world.cameras
        assert name in _compiled_camera_names(sim)
        assert _newton_add_camera(name)["status"] == "success"
        assert _isaac_engine().add_camera(name, position=[2.0, 2.0, 2.0])["status"] == "success"


class TestTheConsumerThatReadsTheNameAsStructure:
    """Why the rule: what each refused name did to a consumer before the fix."""

    def test_a_wildcard_name_publishes_where_another_cameras_subscriber_listens(self) -> None:
        """Zenoh routes a ``put`` by intersection, so ``'**'`` is not a name."""
        zenoh = pytest.importorskip("zenoh")
        asking_for_wrist = zenoh.KeyExpr("strands/rover01/camera/wrist")
        for name in ("*", "**"):
            published_on = zenoh.KeyExpr(f"strands/rover01/camera/{name}")
            assert published_on.intersects(asking_for_wrist), name
            assert camera_frame_key_error("add_camera", "name", name) is not None
        # '$' opens one too: '$*' matches inside a chunk, so 'a$*b' is delivered
        # to the peer subscribed to the real camera 'a_b'.
        assert zenoh.KeyExpr("strands/rover01/camera/a$*b").intersects(zenoh.KeyExpr("strands/rover01/camera/a_b"))
        assert camera_frame_key_error("add_camera", "name", "a$*b") is not None
        # The control: a real name reaches only the peer that asked for it.
        assert not zenoh.KeyExpr("strands/rover01/camera/front").intersects(asking_for_wrist)

    @pytest.mark.parametrize("name", ["cam#1", "cam?1", "a$b", "a//b", "/wrist", "wrist/"])
    def test_a_forbidden_character_makes_the_topic_unpublishable(self, name: str) -> None:
        """The frames are not misrouted, they are never published at all."""
        zenoh = pytest.importorskip("zenoh")
        with pytest.raises(Exception):  # noqa: B017 -- zenoh raises its own ZError type
            zenoh.KeyExpr(f"strands/rover01/camera/{name}")
        assert camera_frame_key_error("add_camera", "name", name) is not None

    def test_and_the_publish_loop_reports_nothing_when_it_fails(self) -> None:
        """Which is why the door has to refuse it: the failure is logged at debug."""
        from strands_robots.mesh import Mesh

        inner = types.SimpleNamespace(is_connected=True, name="so101", config=types.SimpleNamespace(cameras={}))
        mesh = Mesh(types.SimpleNamespace(tool_name_str="so101", robot=inner), peer_id="rover01")
        frame = np.full((48, 64, 3), 128, dtype=np.uint8)
        with patch("strands_robots.mesh.core.put", side_effect=RuntimeError("Invalid Key Expr")) as mock_put:
            mesh._encode_and_publish_frames({"cam#1": frame}, ["cam#1"])
        assert mock_put.call_args[0][0] == "strands/rover01/camera/cam#1"

    @pytest.mark.parametrize(("name", "resolved"), [("..", "frames/123.jpg"), (".", "frames/rover01/123.jpg")])
    def test_a_relative_segment_writes_outside_the_key_it_asked_for(self, name: str, resolved: str) -> None:
        from strands_robots.mesh.iot.camera_offload import CameraOffloader

        offloader = CameraOffloader(bucket="fleet-frames", prefix="frames")
        key = offloader.s3_key_for("rover01", name, 123)
        assert posixpath.normpath(key) == resolved
        assert camera_frame_key_error("add_camera", "name", name) is not None
        # The control: a usable name writes exactly where it says it does.
        assert offloader.s3_key_for("rover01", "wrist", 123) == "frames/rover01/wrist/123.jpg"
