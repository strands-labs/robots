"""``robot_state_keys`` is an ordered list of distinct joint names on every surface.

``robot_state_keys`` names the actuators a policy emits actions for - they are the
keys ``send_action`` resolves - so the list decides which actuator each action
value is sent to. Fifteen ``set_robot_state_keys`` surfaces accept it (thirteen
providers, the remote client, and the abstract declaration on ``Policy``) and
none of them validated its shape.

Its two sibling setters on the same class do. ``set_control_frequency`` and
``set_rtc_observed_delay`` are concrete on ``Policy`` and each raises through a
shared domain in :mod:`strands_robots.utils`; ``set_robot_state_keys`` is
``@abstractmethod`` with a ``pass`` body, so there was no shared implementation
to carry one and each provider re-implemented the setter without it. Measured on
one ``MockPolicy`` instance before this change, ``set_control_frequency(True)``
and ``set_rtc_observed_delay(1.5)`` raised while ``set_robot_state_keys("wrist")``
and ``set_robot_state_keys([1])`` were accepted.

The consequences, all measured on ``MockPolicy``:

* A single joint name passed as a bare string is iterable per character, so
  ``set_robot_state_keys("shoulder_pan.pos")`` bound 16 per-character joints and
  ``get_actions`` then emitted 13-key action dicts - the width silently shrinking
  from 16 to 13 as the repeated characters collapsed in the dict. Those single
  characters are the keys a robot would have been commanded on.
* A repeated name was bound at width 3 and emitted at width 2, because the
  names key the emitted action dict and the duplicate collapses there. The same
  collapse narrows ``lerobot_async``'s ``{key: float for key in
  self.robot_state_keys}`` hardware-feature map, which then declares fewer
  columns than ``align_action_values`` is handed.
* A ``Mapping`` was accepted with its values silently discarded, a one-shot
  iterator was bound and exhausted by the first read, and non-string or blank
  entries were bound as given.

The rule already existed as :func:`strands_robots.utils.name_list_error`, whose
docstring claims to be the domain for "every parameter that carries an ordered
list of KEY NAMES"; it was wired to the two ``image_keys`` consumers and the
simulation ``cameras`` subset, but not to this one.

One coercion had to go with it. ``PolicyServer._dispatch`` handled
``MSG_SET_STATE_KEYS`` as ``set_robot_state_keys(list(message.get("keys", [])))``,
and ``list("wrist")`` is ``['w', 'r', 'i', 's', 't']`` - five *distinct,
non-blank* names that pass every shape check. So the coercion laundered a
mis-typed parameter into a well-formed joint list, and a setter-only guard could
not see it. The same file already states that rule for the sibling handler
(``hz`` is "forwarded verbatim" because coercing "would let the wire accept a
rate the in-process API refuses"), so ``MSG_SET_STATE_KEYS`` now forwards
verbatim too. ``RemotePolicy`` validates before its own ``list(...)`` for the
same reason, on the outbound side.

Three providers needed no guard and deliberately did not get one.
``WBCPolicy``, ``MotionBricksPolicy`` and ``KimodoPolicy`` resolve every G1 joint
they drive BY NAME inside the caller's list, so a bare string, a mapping, a
one-shot iterator, and non-string or blank entries all fail that membership
check already - measured, all five refused with a message naming the missing
joints. They also tolerate a repeated name on purpose: for the two
index-resolved ones, ``test_flat_state_name_resolved_first_occurrence_wins``
pins that a duplicate resolves to its FIRST occurrence and must not shift the
resolved slot, which is a reviewed decision this change does not reopen.
``KimodoPolicy`` is total for a second, stronger reason: it keys the emitted
action dict off the canonical ``KIMODO_G1_JOINTS`` tuple rather than off the
caller's list, so the width this file is about cannot be narrowed by a
duplicate at all - pinned below. All three are therefore classified as
already-total rather than exempted, and each claim is pinned behaviourally so
the classification cannot hide a silent accept.

``None`` and an empty list keep their existing "auto-detect" meaning: like every
other consumer of the shared domain, the check is gated on a truthy value.

The AST classifier below proves that each of the nine owning surfaces CALLS
the shared domain. It cannot prove that any of them RAISES: a body keeping the
``name_list_error(...)`` call and dropping the ``raise`` satisfies it unchanged.
Only ``MockPolicy`` and ``RemotePolicy`` were driven behaviourally, so on the
other seven the refusal was asserted structurally and had never fired - measured
with coverage over the suite, the ``raise ValueError(error)`` line was unexecuted
in ``cosmos3``, ``curobo``, ``groot``, ``lerobot_async``, ``lerobot_local``,
``moveit2`` and ``vera``. Each is now constructed and driven directly, and the
table that does so is derived from ``_MUST_VALIDATE`` so a provider added later
cannot quietly join the structurally-only half.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

import strands_robots
from strands_robots.inference.client import RemotePolicy
from strands_robots.policies.composite import CompositePolicy
from strands_robots.policies.mock import MockPolicy
from strands_robots.policies.motionbricks.policy import MotionBricksPolicy
from strands_robots.policies.wbc.policy import WBCConfig, WBCPolicy
from strands_robots.utils import name_list_error

_PACKAGE = pathlib.Path(strands_robots.__file__).parent

# The surfaces that must resolve through the shared domain themselves, as
# "<path relative to strands_robots/>::<class>". Pinned by name so a provider
# added later shows up as a failure of test_every_surface_is_classified rather
# than quietly widening a count.
_MUST_VALIDATE = {
    "inference/client.py::RemotePolicy",
    "policies/cosmos3/policy.py::Cosmos3Policy",
    "policies/curobo/policy.py::CuroboPolicy",
    "policies/groot/policy.py::Gr00tPolicy",
    "policies/lerobot_async/policy.py::LerobotAsyncPolicy",
    "policies/lerobot_local/policy.py::LerobotLocalPolicy",
    "policies/mock.py::MockPolicy",
    "policies/moveit2/policy.py::MoveIt2Policy",
    "policies/vera/provider.py::VeraPolicy",
}

# Already total without the shared domain: every joint they drive is resolved by
# name inside the caller's list, so a malformed shape fails that check instead.
# The two index-resolved ones deliberately tolerate a repeated name (first
# occurrence wins), so wiring them to the shared domain would reopen a reviewed
# decision; KimodoPolicy keys its action dict off the canonical joint tuple, so
# a duplicate cannot narrow the emitted width either way.
_TOTAL_BY_MEMBERSHIP = {
    "policies/kimodo/policy.py::KimodoPolicy",
    "policies/motionbricks/policy.py::MotionBricksPolicy",
    "policies/protomotions/policy.py::ProtoMotionsPolicy",
    "policies/wbc/policy.py::WBCPolicy",
}

# Pure delegators: they bind nothing themselves, so the wrapped policy's guard
# is the one that fires.
_MUST_FORWARD = {
    "policies/composite.py::CompositePolicy",
    "policies/persistent.py::PersistentPolicy",
}

# The abstract declaration carries the contract in its docstring, not in code.
_ABSTRACT = {"policies/base.py::Policy"}


def _classify(source: str) -> dict[str, dict[str, bool]]:
    """Classify every ``set_robot_state_keys`` in ``source`` by what it does.

    Returns a mapping of class name to flags: whether the body calls the shared
    domain, whether it forwards to another ``set_robot_state_keys``, and whether
    it is the abstract declaration.
    """
    out: dict[str, dict[str, bool]] = {}
    tree = ast.parse(source)
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name != "set_robot_state_keys":
                continue
            validates = False
            forwards = False
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Name) and func.id == "name_list_error":
                    validates = True
                elif isinstance(func, ast.Attribute) and func.attr == "name_list_error":
                    validates = True
                elif isinstance(func, ast.Attribute) and func.attr == "set_robot_state_keys":
                    forwards = True
            abstract = any(
                (isinstance(d, ast.Name) and d.id == "abstractmethod")
                or (isinstance(d, ast.Attribute) and d.attr == "abstractmethod")
                for d in fn.decorator_list
            )
            out[cls.name] = {"validates": validates, "forwards": forwards, "abstract": abstract}
    return out


def _discover() -> dict[str, dict[str, bool]]:
    """Classify every ``set_robot_state_keys`` implementation in the package."""
    found: dict[str, dict[str, bool]] = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        for cls_name, flags in _classify(path.read_text(encoding="utf-8")).items():
            found[f"{path.relative_to(_PACKAGE).as_posix()}::{cls_name}"] = flags
    return found


# --------------------------------------------------------------------------
# Structural parity: no surface may bind a joint-name list it did not check
# --------------------------------------------------------------------------


def test_every_surface_is_classified() -> None:
    """The set of ``set_robot_state_keys`` surfaces is exactly the pinned set.

    Non-vacuity guard. If this fails because a provider was added, the fix is to
    give it the shared domain and add it to ``_MUST_VALIDATE`` - not to relax the
    assertion.
    """
    assert set(_discover()) == _MUST_VALIDATE | _TOTAL_BY_MEMBERSHIP | _MUST_FORWARD | _ABSTRACT


def test_every_binding_surface_calls_the_shared_domain() -> None:
    """Each provider that binds the names resolves them through the one domain."""
    found = _discover()
    missing = sorted(name for name in _MUST_VALIDATE if not found[name]["validates"])
    assert not missing, f"these surfaces bind robot_state_keys without name_list_error: {missing}"


def test_delegating_wrappers_forward_rather_than_re_validate() -> None:
    """The two wrappers bind nothing, so they forward to the policy that does."""
    found = _discover()
    for name in _MUST_FORWARD:
        assert found[name]["forwards"], f"{name} must forward to the wrapped policy"


def test_the_base_declaration_stays_abstract() -> None:
    """The contract lives on the abstract declaration; the domain lives per-provider."""
    found = _discover()
    for name in _ABSTRACT:
        assert found[name]["abstract"]
        assert not found[name]["validates"]


def test_the_classifier_catches_a_provider_that_skips_the_domain() -> None:
    """Planted defect: the structural test must fail on an unguarded provider.

    Without this, a classifier that silently matched nothing would let every
    assertion above pass vacuously.
    """
    guarded = _classify(
        "class Guarded:\n"
        "    def set_robot_state_keys(self, robot_state_keys):\n"
        '        if robot_state_keys and (error := name_list_error(robot_state_keys, "p", "c")):\n'
        "            raise ValueError(error)\n"
        "        self._keys = list(robot_state_keys)\n"
    )
    assert guarded["Guarded"]["validates"] is True

    unguarded = _classify(
        "class Unguarded:\n"
        "    def set_robot_state_keys(self, robot_state_keys):\n"
        "        self._keys = list(robot_state_keys)\n"
    )
    assert unguarded["Unguarded"]["validates"] is False
    assert unguarded["Unguarded"]["forwards"] is False


# --------------------------------------------------------------------------
# The wire must not launder the value into a shape the domain accepts
# --------------------------------------------------------------------------


def test_the_state_keys_handler_forwards_the_wire_value_verbatim() -> None:
    """``MSG_SET_STATE_KEYS`` must not wrap the payload in ``list(...)``.

    ``list("wrist")`` is five distinct non-blank names, so it passes the shared
    domain: coercing on the server would launder a mis-typed parameter past every
    guard. The policy owns the domain, exactly as the ``hz`` handler documents.
    """
    source = (_PACKAGE / "inference" / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_robot_state_keys"
    ]
    assert len(calls) == 1, "expected exactly one set_robot_state_keys forward in the server"
    (arg,) = calls[0].args
    assert not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "list"), (
        "the wire value is coerced with list(...), which turns a bare string into "
        "one name per character and passes the shared domain"
    )


def test_list_of_a_bare_string_would_pass_the_domain() -> None:
    """Why the coercion had to go: its output is indistinguishable from valid input."""
    assert name_list_error("wrist", "robot_state_keys", "c") is not None
    assert name_list_error(list("wrist"), "robot_state_keys", "c") is None


# --------------------------------------------------------------------------
# Behaviour: every malformed shape is refused, and refusal binds nothing
# --------------------------------------------------------------------------

_MALFORMED: list[tuple[str, Any]] = [
    ("bare string", "shoulder_pan.pos"),
    ("repeated name", ["elbow", "elbow", "wrist"]),
    ("mapping", {"elbow": 1.0, "wrist": 2.0}),
    ("non-string entry", ["elbow", 2.5]),
    ("blank name", ["elbow", "  "]),
    ("one-shot iterator", None),  # built per-case below; an iterator cannot be reused
]


@pytest.mark.parametrize("label,value", [c for c in _MALFORMED if c[0] != "one-shot iterator"])
def test_a_malformed_joint_name_list_is_refused(label: str, value: Any) -> None:
    """Each shape the domain names is refused, naming the parameter and surface."""
    policy = MockPolicy()
    with pytest.raises(ValueError, match="robot_state_keys"):
        policy.set_robot_state_keys(value)


def test_a_one_shot_iterator_is_refused() -> None:
    """A generator would present as empty to the second read of the list."""
    policy = MockPolicy()
    with pytest.raises(ValueError, match="one-shot iterator"):
        policy.set_robot_state_keys(iter(["elbow", "wrist"]))  # type: ignore[arg-type]


def test_the_bare_string_message_names_the_per_character_reading() -> None:
    """The message has to say what the string WOULD have bound, or it reads as pedantry."""
    policy = MockPolicy()
    with pytest.raises(ValueError) as excinfo:
        policy.set_robot_state_keys("shoulder_pan.pos")  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "not a single string" in message
    assert "16 name(s)" in message
    assert "['shoulder_pan.pos']" in message


def test_a_refused_list_binds_nothing() -> None:
    """Refusal leaves the policy exactly as it was - no half-applied layout."""
    policy = MockPolicy()
    policy.set_robot_state_keys(["elbow", "wrist"])
    with pytest.raises(ValueError):
        policy.set_robot_state_keys("gripper")  # type: ignore[arg-type]
    assert policy.robot_state_keys == ["elbow", "wrist"]


def test_a_refusal_is_raised_before_any_joint_is_bound_on_a_fresh_policy() -> None:
    """A first call that is refused leaves the constructor default in place."""
    policy = MockPolicy()
    with pytest.raises(ValueError):
        policy.set_robot_state_keys("gripper")  # type: ignore[arg-type]
    assert policy.robot_state_keys == []


# --------------------------------------------------------------------------
# Behaviour: nothing that worked before changes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,value",
    [
        ("distinct names", ["elbow", "wrist", "shoulder_pan.pos"]),
        ("a tuple", ("elbow", "wrist")),
        ("empty list means auto-detect", []),
    ],
)
def test_a_usable_joint_name_list_is_accepted(label: str, value: Any) -> None:
    """The check is gated on a truthy value, so ``[]`` keeps its existing meaning."""
    MockPolicy().set_robot_state_keys(value)


def test_none_is_the_callers_to_skip_not_this_checks_to_reject() -> None:
    """``None`` means "not supplied", as on every other consumer of the domain."""
    MockPolicy().set_robot_state_keys(None)  # type: ignore[arg-type]


def test_a_distinct_list_still_drives_the_action_dict_unchanged() -> None:
    """End to end: the accepted path is untouched by the guard."""
    policy = MockPolicy()
    policy.set_robot_state_keys(["elbow", "wrist"])
    actions = asyncio.run(policy.get_actions({"elbow": 0.0, "wrist": 0.0}, "task"))
    assert actions
    assert all(sorted(step) == ["elbow", "wrist"] for step in actions)


# --------------------------------------------------------------------------
# The wrappers and the remote client reach the same verdict
# --------------------------------------------------------------------------


def test_a_composite_refuses_through_its_children() -> None:
    """The wrapper binds nothing, so the child's guard is the one that fires."""
    lower, upper = MockPolicy(), MockPolicy()
    composite = CompositePolicy(lower, upper)
    with pytest.raises(ValueError, match="robot_state_keys"):
        composite.set_robot_state_keys("wrist")  # type: ignore[arg-type]
    assert lower.robot_state_keys == []
    assert upper.robot_state_keys == []


def test_the_remote_client_refuses_before_anything_reaches_the_wire() -> None:
    """Validated on the outbound side, ahead of its own ``list(...)`` flattening."""
    client = RemotePolicy(host="127.0.0.1", port=1)
    with pytest.raises(ValueError, match="robot_state_keys"):
        client.set_robot_state_keys("wrist")  # type: ignore[arg-type]
    assert client._robot_state_keys == []


def test_the_remote_client_still_accepts_a_distinct_list_while_disconnected() -> None:
    """The keys are stored for replay on the next handshake, as before."""
    client = RemotePolicy(host="127.0.0.1", port=1)
    client.set_robot_state_keys(["elbow", "wrist"])
    assert client._robot_state_keys == ["elbow", "wrist"]


# --------------------------------------------------------------------------
# Sibling parity on one instance: the asymmetry that motivated this
# --------------------------------------------------------------------------


def test_the_three_configuration_setters_now_agree() -> None:
    """All three refuse a value they cannot honor, instead of two of three."""
    policy = MockPolicy()
    with pytest.raises(ValueError):
        policy.set_control_frequency(True)
    with pytest.raises(ValueError):
        policy.set_rtc_observed_delay(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        policy.set_robot_state_keys("wrist")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        policy.set_robot_state_keys([1])  # type: ignore[list-item]


# --------------------------------------------------------------------------
# The two already-total providers: pinned, not merely excused
# --------------------------------------------------------------------------


def _wbc_policy() -> WBCPolicy:
    """A WBC policy on the real G1 layout, with no model files on disk."""
    return WBCPolicy(
        config=WBCConfig(
            policy_path="policy.onnx",
            num_actions=15,
            n_obs_joints=29,
            command_dim=7,
            single_obs_dim=86,
            obs_history_len=1,
            default_angles=[0.1] * 15,
            kps=[100.0] * 15,
            kds=[1.0] * 15,
        ),
        allow_missing_models=True,
    )


@pytest.mark.parametrize(
    "label,value",
    [
        ("bare string", "left_hip_pitch_joint"),
        ("mapping", {"left_hip_pitch_joint": 1.0}),
        ("non-string entry", [1, 2.5, None]),
        ("blank name", ["", "  "]),
    ],
)
def test_wbc_refuses_a_malformed_list_through_its_by_name_check(label: str, value: Any) -> None:
    """No shared domain needed: the membership check is already total here."""
    with pytest.raises(ValueError, match="missing expected G1"):
        _wbc_policy().set_robot_state_keys(value)


def test_wbc_refuses_a_one_shot_iterator_too() -> None:
    """The iterator is consumed by the ``list(...)``, then fails membership."""
    with pytest.raises(ValueError, match="missing expected G1"):
        _wbc_policy().set_robot_state_keys(iter(["left_hip_pitch_joint"]))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "label,value",
    [
        ("bare string", "left_hip_pitch_joint"),
        ("mapping", {"left_hip_pitch_joint": 1.0}),
        ("non-string entry", [1, 2.5, None]),
        ("blank name", ["", "  "]),
    ],
)
def test_motionbricks_refuses_a_malformed_list_through_its_by_name_check(label: str, value: Any) -> None:
    """Same shape of check, same verdict, so the same exemption is justified.

    ``set_robot_state_keys`` reads no constructor state before it raises - only
    the module-level joint list - so a bare instance is enough, as the config
    resolution tests in ``tests/policies/wbc`` already do.
    """
    policy = object.__new__(MotionBricksPolicy)
    with pytest.raises(ValueError, match="missing expected G1 joints"):
        policy.set_robot_state_keys(value)


def test_the_by_name_providers_still_tolerate_a_repeated_name() -> None:
    """A reviewed decision this change deliberately does not reopen.

    ``test_flat_state_name_resolved_first_occurrence_wins`` pins that a
    duplicated joint name resolves to its FIRST occurrence. Wiring these two to
    the shared domain would have turned that into a refusal, so the exemption is
    load-bearing rather than convenient.
    """
    from strands_robots.policies.wbc.policy import WBC_G1_ALL_JOINTS

    policy = _wbc_policy()
    keys = ["floating_base_joint", *WBC_G1_ALL_JOINTS, "left_hip_pitch_joint"]
    policy.set_robot_state_keys(keys)
    assert policy._robot_state_keys == keys


class _RampAgent:
    """A motion agent returning a per-joint ramp, so no sampler is needed.

    ``KimodoPolicy`` takes its agent by injection, so the construction below is
    the real class with a real config - not a bare ``object.__new__`` instance.
    """

    def sample(
        self,
        prompt: str,
        num_frames: int,
        diffusion_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> Any:
        import numpy as np

        from strands_robots.policies.kimodo.policy import KIMODO_G1_JOINTS

        out = np.zeros((num_frames, 7 + len(KIMODO_G1_JOINTS)), dtype=np.float32)
        out[:, 6] = 1.0  # identity quaternion
        out[:, 7:] = np.linspace(0.0, 1.0, len(KIMODO_G1_JOINTS))
        return out


def _kimodo_policy() -> Any:
    """A Kimodo policy with an injected agent: no diffusers, weights, or CUDA."""
    from strands_robots.policies.kimodo.policy import KimodoConfig, KimodoPolicy

    return KimodoPolicy(
        config=KimodoConfig(num_frames=4, native_fps=30, tracker_fps=30),
        motion_agent=_RampAgent(),
    )


@pytest.mark.parametrize(
    "label,value",
    [
        ("bare string", "left_hip_pitch_joint"),
        ("mapping", {"left_hip_pitch_joint": 1.0}),
        ("non-string entry", [1, 2.5, None]),
        ("blank name", ["", "  "]),
    ],
)
def test_kimodo_refuses_a_malformed_list_through_its_by_name_check(label: str, value: Any) -> None:
    """Third by-name provider, same verdict, so the same exemption is justified."""
    with pytest.raises(ValueError, match="missing expected G1 joints"):
        _kimodo_policy().set_robot_state_keys(value)


def test_kimodo_refuses_a_one_shot_iterator_too() -> None:
    """Consumed by the ``list(...)``, then it fails membership rather than binding."""
    with pytest.raises(ValueError, match="missing expected G1 joints"):
        _kimodo_policy().set_robot_state_keys(iter(["left_hip_pitch_joint"]))


def test_kimodo_refusal_leaves_the_previous_joint_list_bound() -> None:
    """Refusal binds nothing: no half-applied layout, as on every other surface."""
    from strands_robots.policies.kimodo.policy import KIMODO_G1_JOINTS

    policy = _kimodo_policy()
    good = ["floating_base_joint", *KIMODO_G1_JOINTS]
    policy.set_robot_state_keys(good)
    with pytest.raises(ValueError):
        policy.set_robot_state_keys("left_hip_pitch_joint")
    assert policy._robot_state_keys == good


def test_kimodo_emits_all_29_joints_even_when_the_caller_repeats_a_name() -> None:
    """The width this file is about cannot be narrowed by a duplicate here.

    The other two by-name providers tolerate a repeated name because they
    resolve a slot by index. ``KimodoPolicy`` keys the emitted action dict off
    the canonical ``KIMODO_G1_JOINTS`` tuple instead of off the caller's list,
    so a duplicate cannot collapse the emitted keys - the failure mode that
    motivated the shared domain is unreachable rather than merely tolerated.
    """
    from strands_robots.policies.kimodo.policy import KIMODO_G1_JOINTS

    policy = _kimodo_policy()
    policy.set_robot_state_keys(["floating_base_joint", *KIMODO_G1_JOINTS, "left_hip_pitch_joint"])
    (action,) = asyncio.run(policy.get_actions({}, "walk forward"))
    assert sorted(action) == sorted(KIMODO_G1_JOINTS)
    assert len(action) == len(KIMODO_G1_JOINTS) == 29


# --------------------------------------------------------------------------
# Behaviour on every provider that owns the check, not only MockPolicy
# --------------------------------------------------------------------------


def _cosmos3() -> Any:
    """Service backend: the constructor records host/port and dials nothing."""
    from strands_robots.policies.cosmos3.policy import Cosmos3Policy

    return Cosmos3Policy(backend="service")


def _curobo() -> Any:
    """An injected planner keeps cuRobo itself out of the construction."""
    from strands_robots.policies.curobo.policy import CuroboPolicy

    return CuroboPolicy(motion_gen=object(), warmup=False)


def _groot() -> Any:
    """Service mode: the ZMQ socket is opened on first inference, not here."""
    from strands_robots.policies.groot.policy import Gr00tPolicy

    return Gr00tPolicy()


def _lerobot_async() -> Any:
    """Both required kwargs supplied; the inference client connects lazily."""
    from strands_robots.policies.lerobot_async.policy import LerobotAsyncPolicy

    return LerobotAsyncPolicy(policy_type="act", pretrained_name_or_path="unused/checkpoint")


def _lerobot_local() -> Any:
    """An empty checkpoint path leaves the model unloaded."""
    from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

    return LerobotLocalPolicy()


def _moveit2() -> Any:
    """Service mode: the ZMQ socket is opened on the first plan request."""
    from strands_robots.policies.moveit2.policy import MoveIt2Policy

    return MoveIt2Policy()


def _vera() -> Any:
    """An injected client with no auto-launch keeps VERA out of the process."""
    from strands_robots.policies.vera.provider import VeraPolicy

    client: Any = object()  # dependency injection: nothing dials in these tests
    return VeraPolicy(auto_launch_server=False, client=client)


# (surface id as classified above, factory, the attribute the setter binds into,
# the import its constructor needs). A ``None`` attribute means the provider
# validates without storing; a ``None`` import means it needs no extra.
_Surface = tuple[str, Callable[[], Any], str | None, str | None]

_OWNING_SURFACES: list[_Surface] = [
    ("policies/cosmos3/policy.py::Cosmos3Policy", _cosmos3, "robot_state_keys", None),
    ("policies/curobo/policy.py::CuroboPolicy", _curobo, "_robot_state_keys", None),
    ("policies/groot/policy.py::Gr00tPolicy", _groot, None, "zmq"),
    ("policies/lerobot_async/policy.py::LerobotAsyncPolicy", _lerobot_async, "robot_state_keys", None),
    ("policies/lerobot_local/policy.py::LerobotLocalPolicy", _lerobot_local, "robot_state_keys", "torch"),
    ("policies/moveit2/policy.py::MoveIt2Policy", _moveit2, "_robot_state_keys", "zmq"),
    ("policies/vera/provider.py::VeraPolicy", _vera, "_robot_state_keys", None),
]
_OWNING_IDS = [surface.split("::")[1] for surface, *_ in _OWNING_SURFACES]

_STORING_SURFACES = [entry for entry in _OWNING_SURFACES if entry[2] is not None]
_STORING_IDS = [surface.split("::")[1] for surface, *_ in _STORING_SURFACES]


def _build(entry: _Surface) -> Any:
    """Construct the provider, skipping cleanly when its extra is absent."""
    _, factory, _, requires = entry
    if requires is not None:
        pytest.importorskip(requires, reason=f"{requires} needed to construct this provider")
    return factory()


def test_the_behavioural_table_covers_every_surface_that_owns_the_check() -> None:
    """Derived, so a provider added later cannot skip the behavioural half.

    ``MockPolicy`` and ``RemotePolicy`` are driven by the sections above; these
    seven are the remainder. A tenth owning surface fails here rather than
    joining the set whose refusal only the AST classifier ever sees.
    """
    driven_above = {"policies/mock.py::MockPolicy", "inference/client.py::RemotePolicy"}
    assert {entry[0] for entry in _OWNING_SURFACES} | driven_above == _MUST_VALIDATE


@pytest.mark.parametrize("entry", _OWNING_SURFACES, ids=_OWNING_IDS)
def test_every_owning_provider_refuses_the_bare_string(entry: _Surface) -> None:
    """The headline mistake, refused with the shared domain's message verbatim.

    Equality rather than a substring: the provider has to return the shared
    verdict, not a locally re-worded copy that could drift from the other eight.
    """
    policy = _build(entry)
    with pytest.raises(ValueError) as excinfo:
        policy.set_robot_state_keys("shoulder_pan.pos")
    assert str(excinfo.value) == name_list_error("shoulder_pan.pos", "robot_state_keys", "set_robot_state_keys")


@pytest.mark.parametrize("entry", _OWNING_SURFACES, ids=_OWNING_IDS)
def test_every_owning_provider_refuses_a_non_string_entry(entry: _Surface) -> None:
    """A second shape, so the claim is about the surface rather than one value."""
    policy = _build(entry)
    with pytest.raises(ValueError, match="robot_state_keys"):
        policy.set_robot_state_keys(["elbow", 2.5])


@pytest.mark.parametrize("entry", _STORING_SURFACES, ids=_STORING_IDS)
def test_a_refusal_leaves_the_previously_bound_layout(entry: _Surface) -> None:
    """No half-applied layout: for the stored keys the refused call is a no-op."""
    policy = _build(entry)
    attribute = entry[2]
    assert attribute is not None  # _STORING_SURFACES is filtered on this
    policy.set_robot_state_keys(["elbow", "wrist"])
    with pytest.raises(ValueError):
        policy.set_robot_state_keys("gripper")
    assert getattr(policy, attribute) == ["elbow", "wrist"]


def test_the_validate_only_provider_stores_nothing_either_way() -> None:
    """Gr00t translates keys through its own mappings, so it binds none of them.

    Its setter exists to reach the same verdict as the others rather than to
    record anything, which is why it has no attribute to check above.
    """
    policy = _build(("", _groot, None, "zmq"))
    with pytest.raises(ValueError, match="robot_state_keys"):
        policy.set_robot_state_keys("shoulder_pan.pos")
    policy.set_robot_state_keys(["elbow", "wrist"])
    assert not hasattr(policy, "robot_state_keys")


@pytest.mark.parametrize("entry", _OWNING_SURFACES, ids=_OWNING_IDS)
def test_every_owning_provider_accepts_a_distinct_list(entry: _Surface) -> None:
    """The over-reach control: only what the domain names is refused."""
    policy = _build(entry)
    policy.set_robot_state_keys(["elbow", "wrist", "shoulder_pan.pos"])
