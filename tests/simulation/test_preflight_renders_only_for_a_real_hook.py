"""The policy preflight gathers its observation only when a hook will read it.

``SimEngine._preflight_policy_config`` runs a provider's class-level
:meth:`~strands_robots.policies.base.Policy.preflight` hook before every
rollout, and sources that hook's ``observation_keys`` argument from
``get_observation``. Without ``skip_images`` that call renders EVERY camera in
the scene, so the cost is paid once per ``run_policy`` / ``eval_policy`` /
``start_policy`` - including for the providers that never override ``preflight``
(every shipped provider except ``lerobot_local``), whose keys
:func:`~strands_robots.policies.preflight_policy` discards without reading.

Two rules, pinned through the public MuJoCo surface on a scene that has a
camera to render:

1. A provider with the default no-op ``preflight`` never has an image-bearing
   observation gathered on its behalf. ``mock`` also declares
   ``requires_images = False``, so across a whole rollout NO call asks for
   frames.
2. A provider that DOES override ``preflight`` still gets the full observation,
   and the hook still receives the camera keys. ``skip_images`` omits camera
   keys altogether, so a fix that skipped the render for an overriding hook
   would silently strip the routing information the hook validates.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies import factory as policy_factory
from strands_robots.policies import register_policy
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation

_CAMERA = "corridor-cam"
_OVERRIDING_PROVIDER = "preflight_observes_keys_probe"


class _KeyRecordingPolicy(MockPolicy):
    """Overrides ``preflight`` and records the observation keys it is handed."""

    seen_keys: list[set[str]] = []

    @classmethod
    def preflight(cls, observation_keys, **policy_config):
        cls.seen_keys.append(set(observation_keys))


@pytest.fixture
def sim_with_camera():
    """A robot plus one python camera, so a full observation has a frame to render."""
    sim = Simulation(tool_name="preflight_render_probe", mesh=False)
    sim.create_world()
    sim.add_robot(name="alice", data_config="so100")
    sim.add_camera(_CAMERA, position=[1.0, 0.0, 0.6], target=[0.0, 0.0, 0.2])
    yield sim
    sim.cleanup()


@pytest.fixture
def image_requests(sim_with_camera, monkeypatch):
    """Record whether each ``get_observation`` call asked for camera frames."""
    real = sim_with_camera.get_observation
    asked_for_images: list[bool] = []

    def spy(robot_name=None, skip_images=False, *args, **kwargs):
        asked_for_images.append(not skip_images)
        return real(robot_name, *args, skip_images=skip_images, **kwargs)

    monkeypatch.setattr(sim_with_camera, "get_observation", spy)
    return asked_for_images


@pytest.fixture
def overriding_provider():
    _KeyRecordingPolicy.seen_keys.clear()
    register_policy(_OVERRIDING_PROVIDER, lambda: _KeyRecordingPolicy)
    try:
        yield _OVERRIDING_PROVIDER
    finally:
        policy_factory._runtime_registry.pop(_OVERRIDING_PROVIDER, None)
        _KeyRecordingPolicy.seen_keys.clear()


def _rollout(sim, provider):
    return sim.run_policy(
        robot_name="alice",
        policy_provider=provider,
        duration=0.1,
        control_frequency=50,
        fast_mode=True,
    )


class TestAHookThatWillNotReadTheKeysIsNotFedThem:
    def test_a_default_noop_preflight_never_has_frames_rendered_for_it(self, sim_with_camera, image_requests):
        """``mock`` leaves ``preflight`` alone and declares no need for images,
        so nothing across the rollout - preflight included - asks for frames.
        """
        result = _rollout(sim_with_camera, "mock")

        assert result["status"] == "success"
        assert image_requests, "the rollout must have read the observation at all"
        assert not any(image_requests), (
            "no call may request camera frames for a provider with a no-op "
            f"preflight and requires_images=False; got {image_requests}"
        )

    def test_an_overriding_preflight_still_receives_the_camera_keys(
        self, sim_with_camera, image_requests, overriding_provider
    ):
        """The hook that validates camera routing still sees the camera key, so
        the full observation is still gathered on its behalf.
        """
        result = _rollout(sim_with_camera, overriding_provider)

        assert result["status"] == "success"
        assert any(image_requests), "an overriding preflight needs the full observation"
        assert len(_KeyRecordingPolicy.seen_keys) == 1
        assert _CAMERA in _KeyRecordingPolicy.seen_keys[0]


class TestRequiresImagesCannotAnswerThisQuestion:
    def test_reading_it_off_the_class_reports_the_property_object(self):
        """Why ``requires_images`` is not the predicate the preflight consults:
        it is an instance property, so an uninstantiated class hands back the
        ``property`` object - truthy - even though ``MockPolicy`` declares
        ``False``. A skip conditioned on it would skip nothing here.
        """
        assert bool(MockPolicy.requires_images) is True
        assert MockPolicy().requires_images is False
