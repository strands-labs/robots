"""A remotely-served policy's declared bodies reach the host that supplies them.

``Policy.required_bodies`` is the "policy declares, runtime supplies" contract:
the simulation runtime resolves the names once before a rollout, refuses one the
scene does not contain, and merges each body's world pose into every observation
it hands to ``get_actions``. Both halves run on the machine driving the rollout,
so when the policy is behind a WebSocket the declaration has to cross the wire -
otherwise the proxy declares nothing, no pose is ever merged, no name is ever
checked, and the rollout reports success having supplied none of it.

The end-to-end cells drive a real MuJoCo rollout twice over one table - the
policy in-process, and the same policy behind a real ``PolicyServer`` - and
assert the two agree. The declaration-mirroring cell is derived from ``Policy``
itself, so a fifth declaration property added later joins the requirement
without an edit here.
"""

import pytest

from strands_robots.inference import PolicyServer
from strands_robots.inference.client import RemotePolicy
from strands_robots.policies.base import Policy
from strands_robots.policies.persistent import PersistentPolicy

pytest.importorskip("mujoco", reason="sim rollout requires mujoco - pip install 'strands-robots[sim-mujoco]'")

import strands_robots as sr  # noqa: E402

ANCHOR = "cube"  # a body the scene below contains
GHOST = "no_such_anchor_link"  # a body it does not


class AnchorPolicy(Policy):
    """A policy that consumes a named body's world pose, as a mimic tracker does."""

    def __init__(self, *bodies: str, horizon: int = 1) -> None:
        self._bodies = bodies
        self._horizon = horizon
        self._keys: list[str] = []
        #: Per tick: did ``body.<name>.quat`` arrive for every declared body?
        self.pose_arrived: list[bool] = []

    @property
    def provider_name(self) -> str:
        return "anchor"

    @property
    def requires_images(self) -> bool:
        return False

    @property
    def required_bodies(self) -> tuple[str, ...]:
        return self._bodies

    @property
    def execution_horizon(self) -> int:
        return self._horizon

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = list(robot_state_keys)

    async def get_actions(self, observation, instruction=""):  # noqa: ANN001, ANN201, ARG002
        self.pose_arrived.append(all(f"body.{name}.quat" in observation for name in self._bodies))
        return [{key: 0.0 for key in self._keys} for _ in range(self._horizon)]


class BareStrPolicy(AnchorPolicy):
    """Declares a bare ``str`` - the shape that iterates into one entry per character."""

    @property
    def required_bodies(self):  # type: ignore[override]  # deliberately the wrong shape
        return "torso_link"


def _scene():
    """A real so101 world holding one extra body named ``cube``."""
    sim = sr.Robot("so101", mode="sim", mesh=False)
    sim.add_object(name=ANCHOR, shape="box", size=[0.04] * 3, position=[0.2, 0.0, 0.05])
    return sim


def _drive(policy: Policy, *, remote: bool, steps: int = 5):
    """Roll ``policy`` out in MuJoCo, in-process or behind a real PolicyServer.

    Returns the rollout's status dict, or the message of a pre-rollout refusal
    prefixed with ``"raised: "`` so a caller can compare the two paths' verdicts.
    """
    sim = _scene()
    server = PolicyServer(policy=policy, host="127.0.0.1", port=0).start() if remote else None
    try:
        driven = RemotePolicy(host="127.0.0.1", port=server.port) if server is not None else policy
        try:
            return sim.run_policy(
                robot_name="so101",
                policy_object=driven,
                n_steps=steps,
                control_frequency=30.0,
                fast_mode=True,
            )
        except (RuntimeError, TypeError) as refusal:
            return f"raised: {refusal}"
    finally:
        if server is not None:
            server.stop()
        sim.cleanup()


@pytest.mark.parametrize("remote", [False, True], ids=["in_process", "over_the_wire"])
class TestTheDeclarationReachesTheRuntimeEitherWay:
    """Every guarantee the local path gives holds when the policy is remote."""

    def test_the_declared_body_pose_is_merged_into_every_observation(self, remote):
        policy = AnchorPolicy(ANCHOR)
        result = _drive(policy, remote=remote)
        assert not isinstance(result, str), result
        assert result["status"] == "success", result
        assert policy.pose_arrived, "the policy was never asked for an action, so nothing was measured"
        assert all(policy.pose_arrived), (
            f"body.{ANCHOR}.quat reached {sum(policy.pose_arrived)} of "
            f"{len(policy.pose_arrived)} observations (remote={remote})"
        )

    def test_a_body_the_scene_lacks_is_refused_before_the_rollout(self, remote):
        policy = AnchorPolicy(GHOST)
        result = _drive(policy, remote=remote)
        assert isinstance(result, str), f"the rollout ran instead of refusing (remote={remote}): {result}"
        assert GHOST in result and "does not resolve to a body" in result, result
        # The refusal is worth having because it names what the scene does hold.
        assert ANCHOR in result, result
        assert not policy.pose_arrived, "the policy was stepped despite an unresolvable declaration"

    def test_a_policy_declaring_nothing_is_unaffected(self, remote):
        """Control: the ungated path is identical either way, and stays that way."""
        policy = AnchorPolicy()
        result = _drive(policy, remote=remote)
        assert not isinstance(result, str), result
        assert result["status"] == "success", result
        assert policy.pose_arrived and all(policy.pose_arrived), "vacuously true only if never stepped"


class TestTheHandshakeCarriesTheDeclaration:
    """The wire payload is what makes the client half able to declare anything."""

    def test_the_served_tree_declaration_is_advertised(self):
        server = PolicyServer(policy=AnchorPolicy("torso_link", "pelvis"), host="127.0.0.1", port=0)
        assert server._metadata()["required_bodies"] == ["torso_link", "pelvis"]

    def test_a_wrapper_does_not_hide_the_policy_it_serves(self):
        """The advertised set is collected over the served TREE, not off its root."""
        wrapped = PersistentPolicy("mock", policy_object=AnchorPolicy("torso_link"))
        assert wrapped.required_bodies == (), "PersistentPolicy is expected not to override the property"
        server = PolicyServer(policy=wrapped, host="127.0.0.1", port=0)
        assert server._metadata()["required_bodies"] == ["torso_link"]

    def test_a_malformed_declaration_is_refused_when_the_server_is_built(self):
        """A bare str would advertise one body per character, so it is refused."""
        with pytest.raises(TypeError, match="not a bare str"):
            PolicyServer(policy=BareStrPolicy(), host="127.0.0.1", port=0)

    @pytest.mark.parametrize(
        "advertised",
        ["torso_link", 7, {"torso_link": 1}, ["", "  ", 3, None]],
        ids=["bare_str", "int", "mapping", "unusable_entries"],
    )
    def test_a_peer_advertising_something_unusable_is_ignored(self, advertised):
        """A value the runtime could not validate never becomes this half's declaration."""
        client = RemotePolicy(host="127.0.0.1", port=1)
        client._apply_metadata({"required_bodies": advertised})
        assert client._required_bodies == ()


class TestEveryDeclarationPropertyTracksTheServedPolicy:
    """Derived: a declaration property the proxy does not mirror is a dropped contract.

    Enumerated from ``Policy`` rather than listed, so a property added later is
    graded here on arrival. Four are exempt, each because the proxy answers for
    itself rather than for its peer, and each exemption is checked to name a
    real ``Policy`` member so a rename cannot silently widen the carve-out.
    """

    #: property -> why the proxy's answer is deliberately its own.
    NOT_MIRRORED = {
        "provider_name": "identifies the client half; read remote_provider_name for the peer's",
        "children": "the proxy is a leaf on this host - the served tree is walked on the server",
        "control_frequency": "set by the local runner and forwarded, not declared by the peer",
        "rtc_observed_delay_steps": "counted by the local runner and forwarded on each request",
    }

    @staticmethod
    def _declaration_properties() -> set[str]:
        return {
            name
            for name in dir(Policy)
            if not name.startswith("_")
            and isinstance(getattr(Policy, name, None), property)
            and name not in TestEveryDeclarationPropertyTracksTheServedPolicy.NOT_MIRRORED
        }

    def test_the_enumeration_finds_the_contract(self):
        found = self._declaration_properties()
        assert {"requires_images", "required_bodies", "execution_horizon"} <= found, found
        assert not (set(self.NOT_MIRRORED) - set(dir(Policy))), "an exemption names no Policy member"
        # ``is_chunk_emitting`` answers a declaration too but is a method, so it
        # is outside this enumeration and graded by name in the cell below.
        assert callable(Policy.__dict__["is_chunk_emitting"])

    def test_each_one_reports_the_served_policy_value(self):
        # Every value below differs from the Policy default, so a property that
        # is not mirrored reports the default and fails rather than coinciding.
        served = AnchorPolicy(ANCHOR, horizon=6)
        assert served.requires_images is False
        assert served.execution_horizon == 6
        assert served.is_chunk_emitting() is True

        server = PolicyServer(policy=served, host="127.0.0.1", port=0).start()
        try:
            client = RemotePolicy(host="127.0.0.1", port=server.port)
            drifted = {
                name: (getattr(client, name), getattr(served, name))
                for name in sorted(self._declaration_properties())
                if getattr(client, name) != getattr(served, name)
            }
            assert not drifted, f"the proxy does not report the served policy's declaration: {drifted}"
            # Derived from ``execution_horizon``, so mirroring that property is
            # what makes the proxy answer this correctly. Graded, not assumed.
            assert client.is_chunk_emitting() == served.is_chunk_emitting() is True
        finally:
            server.stop()
