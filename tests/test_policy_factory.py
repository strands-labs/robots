"""Policy(spec) picks a provider from the spec's FORM; ChunkedPolicy replays a chunk."""

import numpy as np
import pytest

from strands_robots import Policy, Robot, Simulation
from strands_robots.policy import CallablePolicy
from strands_robots.policy.chunked import ChunkedPolicy


def test_a_callable_is_a_policy_and_a_subclass_is_a_plain_constructor():
    assert isinstance(Policy(lambda obs: obs.joints.positions), CallablePolicy)
    assert type(CallablePolicy(lambda obs: 0)) is CallablePolicy


@pytest.mark.parametrize(
    "spec",
    [
        "Not-A-Preset?",
        "act",
        "pi0",
        "not a spec",
        "zenoh://host-without-port/x",
        "zenoh://h:7447/",
        "zenoh://h:7447/a/b",
        "zenoh://ftp/h:7447/d",
        "zenoh+carrier-pigeon://h:7447/d",
        "zenoh://tls/h:port/d",
    ],
)
def test_an_unknown_form_is_refused_listing_the_forms(spec):
    with pytest.raises(ValueError, match=r"zenoh://"):
        Policy(spec)


@pytest.mark.parametrize(
    ("spec", "endpoint", "deployment"),
    [
        ("zenoh://10.0.0.5:7447/manipulation", "tcp/10.0.0.5:7447", "manipulation"),
        ("zenoh://tls/zenoh.api.agi.diy:7447/d_abc", "tls/zenoh.api.agi.diy:7447", "d_abc"),
        ("zenoh+tls://zenoh.api.agi.diy:7447/d_abc", "tls/zenoh.api.agi.diy:7447", "d_abc"),
        ("zenoh://quic/[2001:db8::1]:7447/d", "quic/[2001:db8::1]:7447", "d"),
        ("zenoh://tcp/127.0.0.1:17449/d_wfm_ttt_ec2", "tcp/127.0.0.1:17449", "d_wfm_ttt_ec2"),
    ],
)
def test_a_zenoh_url_names_the_link_a_lease_gave(spec, endpoint, deployment):
    """A lease says ``tls/host:port``; written straight into the URL it is the locator LeRobot connects to."""
    from strands_robots.policy.factory import zenoh_locator

    assert zenoh_locator(spec) == (endpoint, deployment)


def test_zenoh_url_and_hub_repo_need_no_network_to_construct():
    pytest.importorskip("lerobot")
    p = Policy("zenoh://10.0.0.5:7447/manipulation", chunk_merge="aligned")
    assert repr(p) == "Policy('zenoh://10.0.0.5:7447/manipulation')"
    assert p.endpoint == "tcp/10.0.0.5:7447" and p.deployment == "manipulation"
    far = Policy("zenoh://tls/zenoh.api.agi.diy:7447/d_abc")
    assert (
        repr(far) == "Policy('zenoh://tls/zenoh.api.agi.diy:7447/d_abc')"
        and far.endpoint == "tls/zenoh.api.agi.diy:7447"
    )
    hub = Policy("lerobot/smolvla_base", device="cpu")
    assert isinstance(hub, ChunkedPolicy) and repr(hub.inner) == "Policy('lerobot/smolvla_base')"


class _Count:
    def __init__(self):
        self.calls = 0

    def act_chunk(self, obs, *, instruction=None):
        self.calls += 1
        return np.tile(np.arange(4.0)[:, None], (1, 6)) * 0.01 * self.calls

    def reset(self):
        pass


def test_chunked_policy_plays_the_chunk_then_asks_again():
    inner = _Count()
    arm = Robot("so101")
    arm.run(ChunkedPolicy(inner), steps=8)
    assert inner.calls == 2
    inner2 = _Count()
    arm.run(ChunkedPolicy(inner2, replan_every=2), steps=8)
    assert inner2.calls == 4


# One render per step the policy looks at: every step for a plain policy, once per chunk for a ChunkedPolicy.
_LOOKS = {
    "plain": (lambda: lambda obs: obs.joints.positions, 8),
    "chunked": (lambda: ChunkedPolicy(_Count()), 2),
    "chunked replan_every=2": (lambda: ChunkedPolicy(_Count(), replan_every=2), 4),
}
_LOOPS = {
    "robot.run": lambda arm, pol: arm.run(pol, steps=8),
    "sim.run": lambda arm, pol: arm.sim.run({arm: pol}, steps=8),
}


@pytest.mark.parametrize("loop", _LOOPS.values(), ids=_LOOPS.keys())
@pytest.mark.parametrize("make,looks", _LOOKS.values(), ids=_LOOKS.keys())
def test_the_loop_renders_only_the_steps_the_policy_looks_at(make, looks, loop):
    with Simulation() as sim:
        arm = Robot("so101", sim=sim)
        sim.camera("wrist", size=(64, 64), attach=arm)
        rendered, render = [], sim.backend.render
        sim.backend.render = lambda name: rendered.append(name) or render(name)
        pol = make()
        loop(arm, pol)
        assert len(rendered) == looks + 1  # + the one frame bind's requirements check reads
        if isinstance(pol, ChunkedPolicy):
            assert pol.inner.calls == looks


def test_stop_when_still_sees_a_frame_on_every_step():
    with Simulation() as sim:
        arm = Robot("so101", sim=sim)
        sim.camera("wrist", size=(64, 64), attach=arm)
        seen = []
        arm.run(ChunkedPolicy(_Count()), steps=8, stop_when=lambda o: seen.append(sorted(o.images)) and False)
        assert seen == [["wrist"]] * 8


def test_a_hardware_camera_is_drained_on_every_step_even_while_a_chunk_replays():
    from test_drivers import make_driver

    d, _ = make_driver()
    arm = Robot("so101", driver=d, twin=False, confirm=lambda a: True)
    reads, images = [], d.images
    d.images = lambda: reads.append(1) or images()  # a V4L2 ring goes stale when nobody dequeues
    arm.run(ChunkedPolicy(_Count()), steps=8)
    assert len(reads) >= 8


class _LeRobotUnits:
    units = "lerobot"

    def act_chunk(self, obs, *, instruction=None):
        return np.full((4, 6), 50.0)

    def act(self, obs, *, instruction=None):
        return np.full(6, 50.0)

    def reset(self):
        pass


def test_chunked_policy_speaks_the_units_its_inner_declares():
    wrapped, bare = (Robot("so101", sim=Simulation()) for _ in range(2))
    wrapped.run(ChunkedPolicy(_LeRobotUnits()), steps=50)
    bare.run(_LeRobotUnits(), steps=50)
    np.testing.assert_allclose(wrapped.observe().joints.positions, bare.observe().joints.positions, atol=1e-6)
    assert ChunkedPolicy(_LeRobotUnits()).units == "lerobot"
