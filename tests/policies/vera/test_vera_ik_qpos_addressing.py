"""Contract tests for VERA IK joint-to-qpos addressing and IK-input resolution.

Exercise the embodiment-agnostic IK plumbing in
:mod:`strands_robots.policies.vera.provider` that does NOT need a running VERA
server, GPU, or the optional ``mink`` IK solver:

* ``VeraPolicy._joint_qpos_addr`` -- maps unqualified ``robot_state_keys`` to
  their qpos addresses in a compiled MuJoCo model by namespaced-joint suffix,
  so the IK seed and output read/write the correct slots even when unrelated
  DOFs (free bodies, other robots) shift the addresses. Plus the cache - keyed
  on the model *and* the state keys, since one scene binds one model to several
  robots in turn - and the positional-identity fallback for introspection-less
  stubs.
* ``VeraPolicy._resolve_ik_inputs`` -- gathers ``(mj_model, ee_frame, q_init)``
  for an IK solve, returning ``None`` for each "not enough wiring" guard
  (no model/ee-frame, no state keys, missing observation key) and seeding
  ``q_init`` from the model rest pose with the observed joint values written
  into their qpos addresses.

All assertions are on observable outputs (the returned mapping / tuple), not
internal state.
"""

from __future__ import annotations

from strands_robots.policies.vera.provider import VeraPolicy

# Two arm joints (namespaced ``myarm/...``) sit AFTER a 7-DoF free joint, so
# their qpos addresses are 7 and 8 -- never 0/1. A positional map would bind
# the wrong slots; only correct suffix matching recovers (shoulder->7, elbow->8).
_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="free1">
      <freejoint name="obj/free"/>
      <geom type="sphere" size="0.05"/>
    </body>
    <body name="b1">
      <joint name="myarm/shoulder" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="b2" pos="0.2 0 0">
        <joint name="myarm/elbow" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# A SECOND robot in the SAME compiled world. This is what SimEngine binds: one
# world model handed to whichever robot it is starting a rollout for, so the
# model is the input that does NOT change between two robots' key sets.
# ``otherarm`` joints sit at qpos 9/10, past ``myarm``'s 7/8.
_TWO_ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="free1">
      <freejoint name="obj/free"/>
      <geom type="sphere" size="0.05"/>
    </body>
    <body name="b1">
      <joint name="myarm/shoulder" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="b2" pos="0.2 0 0">
        <joint name="myarm/elbow" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </body>
    <body name="c1" pos="1 0 0">
      <joint name="otherarm/hip" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="c2" pos="0.2 0 0">
        <joint name="otherarm/knee" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class _FakeClient:
    """Minimal VeraWebsocketClient stand-in so VeraPolicy needs no server."""

    def get_server_metadata(self):
        return {}

    def reset(self, info):
        pass

    def configure(self, p):
        return {}

    def infer(self, req):
        return {}

    def close(self):
        pass


def _make_policy(state_keys):
    p = VeraPolicy(client=_FakeClient(), auto_launch_server=False)
    p._runner = None
    p.set_robot_state_keys(state_keys)
    return p


def _build_model():
    import mujoco

    return mujoco.MjModel.from_xml_string(_MODEL_XML)


def _build_two_robot_model():
    import mujoco

    return mujoco.MjModel.from_xml_string(_TWO_ROBOT_XML)


class TestJointQposAddr:
    """``_joint_qpos_addr`` maps unqualified keys to namespaced-joint qpos slots."""

    def test_suffix_matches_namespaced_joints_past_free_dofs(self):
        model = _build_model()
        p = _make_policy(["shoulder", "elbow"])

        addr = p._joint_qpos_addr(model)

        # shoulder/elbow live at qpos 7/8 (a 7-DoF free joint precedes them);
        # suffix matching must recover those exact addresses, not 0/1.
        assert addr == {"shoulder": 7, "elbow": 8}

    def test_result_is_cached_per_model(self):
        model = _build_model()
        p = _make_policy(["shoulder", "elbow"])

        first = p._joint_qpos_addr(model)
        second = p._joint_qpos_addr(model)

        # Same model id -> cache hit returns the identical mapping object.
        assert second is first

    def test_positional_identity_fallback_for_introspection_less_model(self):
        # A stub without ``njnt`` raises AttributeError inside the introspection
        # loop; the addressing degrades to a positional identity map so a test
        # double still drives the IK seed deterministically (key i -> qpos i).
        p = _make_policy(["a", "b", "c"])

        addr = p._joint_qpos_addr(object())

        assert addr == {"a": 0, "b": 1, "c": 2}


class TestResolveIkInputs:
    """``_resolve_ik_inputs`` returns None on missing wiring, else the IK seed."""

    def test_returns_none_without_model(self):
        p = _make_policy(["shoulder", "elbow"])
        # No model injected and no ee-frame configured.
        assert p._resolve_ik_inputs({"shoulder": 0.0, "elbow": 0.0}) is None

    def test_returns_none_without_state_keys(self):
        model = _build_model()
        p = _make_policy([])
        p.set_ik_target(model, ee_frame_name="b2", ee_frame_type="body")
        assert p._resolve_ik_inputs({"shoulder": 0.0}) is None

    def test_returns_none_when_observation_missing_a_joint(self):
        model = _build_model()
        p = _make_policy(["shoulder", "elbow"])
        p.set_ik_target(model, ee_frame_name="b2", ee_frame_type="body")
        # "elbow" absent from the observation -> cannot build a full seed.
        assert p._resolve_ik_inputs({"shoulder": 0.1}) is None

    def test_seeds_qinit_from_rest_pose_with_observed_joint_values(self):
        model = _build_model()
        p = _make_policy(["shoulder", "elbow"])
        p.set_ik_target(model, ee_frame_name="b2", ee_frame_type="body")

        out = p._resolve_ik_inputs({"shoulder": 0.3, "elbow": -0.4})

        assert out is not None
        ret_model, ee_frame, q_full = out
        assert ret_model is model
        assert ee_frame == "b2"
        # Full nq-length seed: rest-pose free-joint quaternion preserved at
        # indices 3-6, observed joint values written at their qpos addresses.
        assert q_full.shape == (model.nq,)
        assert list(q_full[3:7]) == [1.0, 0.0, 0.0, 0.0]
        assert q_full[7] == 0.3
        assert q_full[8] == -0.4


class TestCacheKeyCoversTheStateKeys:
    """The mapping is re-derived when either input it is built from changes.

    ``SimEngine`` sets a robot's keys and then hands it the one compiled world
    model (``base.py`` -> ``bind_policy_sim_context``), so binding a second
    robot in the same scene changes the keys and *not* the model. The keys are
    this mapping's whole domain, so serving the previous robot's map is not a
    near miss - every lookup misses, and both consumers skip a missing key
    silently rather than refusing.
    """

    def test_a_rebind_against_one_model_is_not_served_the_previous_map(self):
        model = _build_two_robot_model()
        p = _make_policy(["shoulder", "elbow"])

        first = p._joint_qpos_addr(model)
        assert first == {"shoulder": 7, "elbow": 8}

        # Same model object - only the robot changed.
        p.set_robot_state_keys(["hip", "knee"])
        second = p._joint_qpos_addr(model)

        assert second == {"hip": 9, "knee": 10}

    def test_the_seed_carries_the_rebound_robots_joint_values(self):
        # The consequence at _resolve_ik_inputs: a stale map carries none of the
        # new keys, so every observed value is skipped and the seed degrades to
        # the model rest pose with nothing reporting it.
        model = _build_two_robot_model()
        p = _make_policy(["shoulder", "elbow"])
        p.set_ik_target(model, ee_frame_name="b2", ee_frame_type="body")
        p._resolve_ik_inputs({"shoulder": 0.3, "elbow": -0.4})  # warms the cache

        # A rebind does NOT re-enter set_ik_target: autoconfigure_ik returns
        # early once an ee-frame is configured, so nothing else invalidates.
        p.set_robot_state_keys(["hip", "knee"])
        out = p._resolve_ik_inputs({"hip": 0.3, "knee": -0.4})

        assert out is not None
        q_full = out[2]
        assert q_full[9] == 0.3
        assert q_full[10] == -0.4
        # Not the rest pose: the observed values reached the seed.
        import numpy as np

        assert not np.array_equal(q_full, np.asarray(model.qpos0, dtype=np.float64))

    def test_the_decoded_targets_name_the_rebound_robots_joints(self):
        # The consequence at the output consumer: reading each arm joint back
        # from a stale map drops every key, emitting an action dict carrying no
        # arm joint at all.
        model = _build_two_robot_model()
        p = _make_policy(["shoulder", "elbow"])
        p._joint_qpos_addr(model)  # warms the cache for myarm

        p.set_robot_state_keys(["hip", "knee"])
        addr = p._joint_qpos_addr(model)

        row_width = model.nq
        emitted = {k: addr[k] for k in ["hip", "knee"] if k in addr and addr[k] < row_width}
        assert emitted == {"hip": 9, "knee": 10}

    def test_the_positional_fallback_is_rebuilt_for_new_keys(self):
        # The introspection-less path builds a positional map, so its values
        # depend on the key list alone - a stale one is wrong by construction.
        p = _make_policy(["a", "b", "c"])
        stub = object()
        assert p._joint_qpos_addr(stub) == {"a": 0, "b": 1, "c": 2}

        p.set_robot_state_keys(["x", "y"])
        assert p._joint_qpos_addr(stub) == {"x": 0, "y": 1}

    def test_every_input_the_mapping_reads_is_keyed(self):
        # Differing-value table over the two inputs _joint_qpos_addr reads.
        # Each row varies exactly one and must re-derive; the control varies
        # neither and must still be served from the cache.
        model = _build_two_robot_model()
        other_model = _build_two_robot_model()
        p = _make_policy(["shoulder", "elbow"])
        baseline = p._joint_qpos_addr(model)

        # Row 1: the model differs (equal contents, distinct object).
        assert p._joint_qpos_addr(other_model) is not baseline

        # Row 2: the state keys differ.
        p2 = _make_policy(["shoulder", "elbow"])
        base2 = p2._joint_qpos_addr(model)
        p2.set_robot_state_keys(["hip", "knee"])
        assert p2._joint_qpos_addr(model) is not base2

        # Control: neither differs -> the cache this exists for is not disabled.
        p3 = _make_policy(["shoulder", "elbow"])
        base3 = p3._joint_qpos_addr(model)
        assert p3._joint_qpos_addr(model) is base3

    def test_the_key_cannot_alias_a_released_models_address(self):
        # An id() is unique only while its object lives, and an int key keeps
        # nothing alive - so a model-address key can be matched by whatever is
        # allocated there next. Holding the model is what forecloses that, and
        # it is observable: the entry keeps its model reachable.
        import gc
        import weakref

        p = _make_policy(["shoulder", "elbow"])
        model = _build_two_robot_model()
        ref = weakref.ref(model)
        # No set_ik_target here, so the cache entry is the only strong reference.
        p._joint_qpos_addr(model)

        del model
        gc.collect()

        assert ref() is not None
