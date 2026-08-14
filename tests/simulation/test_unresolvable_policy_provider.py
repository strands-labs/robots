# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""An unresolvable ``policy_provider`` is reported on every rollout surface.

``preflight_policy`` swallows resolution failures deliberately, and says why:
"Resolution failures are swallowed (the matching error is surfaced
authoritatively by the subsequent ``create_policy``)". That premise holds for a
library caller, which sees the raise. It does not hold for the agent-tool
surfaces, and each one failed differently for the same caller mistake:

* ``run_policy`` / ``eval_policy`` -- ``create_policy`` raised a bare
  ``ValueError`` **past** the ``status=error`` envelope an agent tool is
  documented to return, so the caller got a traceback rather than a result.
* ``start_policy`` -- the policy is built on a worker thread, so the raise was
  captured in the future and never surfaced: the call reported
  ``status="success"``, "Policy started on 'so100' (async)", while
  ``list_policies_running`` reported "No policies running." A rollout that
  could never build its policy was reported as started.

The message itself already names every registered provider, so the discovery a
caller needs after guessing a name existed all along -- it simply could not
reach them through either channel. A deferral is only sound when the thing it
defers to can report, and on two of these three surfaces it cannot.

Each refusal is asserted as the shared verdict verbatim (envelope equality, not
a substring), matching the sibling pre-flight module: a facade that re-words a
shared rule locally is how two surfaces drift on what one value means.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
from typing import Any

import pytest

pytest.importorskip("mujoco")

import strands_robots  # noqa: E402
from strands_robots.policies import list_providers, policy_provider_error  # noqa: E402
from strands_robots.simulation.mujoco import simulation as mujoco_simulation  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

# A one-actuator arm needing no asset download: enough for the robot resolution
# that runs before these guards.
_ARM = """<mujoco><worldbody><body name="l1">
<joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/></body></worldbody>
<actuator><position name="a1" joint="j1" kp="30" ctrlrange="-1.5 1.5"/></actuator></mujoco>"""

# Names no spelling ``create_policy`` accepts can reach. "molmoact2" is the
# real guess from the report: it is a lerobot *policy type*, not a provider.
UNRESOLVABLE: list[str] = ["nope", "molmoact2", "", "lerobot-local"]

# The built-in providers, read from the shipped registry rather than from
# ``list_providers()``: that accessor also returns ``register_policy`` additions,
# and other test modules register their own, which leak into the session. The
# refusal message names the built-ins, so this is the set to compare against.
BUILTIN_PROVIDERS: list[str] = sorted(
    json.loads((pathlib.Path(strands_robots.__file__).parent / "registry" / "policies.json").read_text())["providers"]
)

# Every spelling that must keep working: the built-in providers, plus the smart
# strings ``resolve_policy`` accepts -- a HuggingFace model ID (including the
# SO-arm checkpoint the report was about), a transport URL, a host:port.
RESOLVABLE: list[str] = [
    *BUILTIN_PROVIDERS,
    "lerobot/act_aloha_sim",
    "allenai/MolmoAct2-SO100_101",
    "zmq://localhost:5555",
    "ws://localhost:8765",
]

# Wrong *type* rather than wrong name: resolution indexes the registry with the
# value, so each of these reached it as a bare ``TypeError`` past the envelope.
NOT_A_STRING: list[Any] = [None, 3, ["mock"], {"provider": "mock"}, True]

# The blocking facades take ``duration``; the async one is bounded the same way.
SURFACES: list[tuple[str, dict[str, Any]]] = [
    ("run_policy", {"duration": 0.05}),
    ("eval_policy", {"n_episodes": 1, "max_steps": 2}),
    ("start_policy", {"duration": 0.2}),
]


def _text(result: dict[str, Any]) -> str:
    """The human-readable half of an agent-tool envelope."""
    return " ".join(c["text"] for c in result.get("content", []) if "text" in c)


@pytest.fixture
def sim(tmp_path):
    """A live world holding one arm, torn down after each case."""
    engine = MuJoCoSimEngine(tool_name="provider_sim", mesh=False)
    engine.create_world()
    xml = tmp_path / "arm.xml"
    xml.write_text(_ARM)
    engine.add_robot(name="arm", urdf_path=str(xml))
    try:
        yield engine
    finally:
        engine.cleanup()


class TestEveryRolloutSurfaceReportsIt:
    """The refusal is returned as an envelope, on all three surfaces."""

    @pytest.mark.parametrize("provider", UNRESOLVABLE)
    @pytest.mark.parametrize("action,extra", SURFACES, ids=[s[0] for s in SURFACES])
    def test_the_refusal_is_an_envelope_not_a_raise(self, sim, action, extra, provider):
        result = getattr(sim, action)(robot_name="arm", policy_provider=provider, **extra)
        assert result["status"] == "error", result
        assert provider is not None

    @pytest.mark.parametrize("provider", UNRESOLVABLE)
    @pytest.mark.parametrize("action,extra", SURFACES, ids=[s[0] for s in SURFACES])
    def test_the_refusal_is_the_shared_verdict_verbatim(self, sim, action, extra, provider):
        reason = policy_provider_error(provider)
        assert reason is not None, "probe value must be outside the resolvable set"
        result = getattr(sim, action)(robot_name="arm", policy_provider=provider, **extra)
        assert _text(result) == reason


class TestTheDiscoverySignalReachesTheCaller:
    """The refusal carries the available set, which is what a guess needs back."""

    @pytest.mark.parametrize("action,extra", SURFACES, ids=[s[0] for s in SURFACES])
    def test_the_refusal_names_every_builtin_provider(self, sim, action, extra):
        result = getattr(sim, action)(robot_name="arm", policy_provider="molmoact2", **extra)
        message = _text(result)
        missing = [p for p in BUILTIN_PROVIDERS if p not in message]
        assert not missing, f"refusal does not name {missing}: {message}"

    def test_the_builtin_set_is_what_the_live_accessor_reports(self):
        """The compared set is real: every built-in is a provider at runtime.

        Guards the JSON read above -- a renamed key or an empty file would make
        the assertion vacuous rather than failing. ``list_providers()`` is a
        superset because it also reports ``register_policy`` additions, which
        the refusal message does not name.
        """
        assert BUILTIN_PROVIDERS, "no built-in providers were read from the registry"
        assert set(BUILTIN_PROVIDERS) <= set(list_providers())

    def test_it_names_the_value_the_caller_supplied(self, sim):
        result = sim.run_policy(robot_name="arm", policy_provider="molmoact2", duration=0.05)
        assert "'molmoact2'" in _text(result)


class TestStartPolicyRefusesBeforeItSubmits:
    """The async surface must not report a rollout it could never build."""

    def test_no_worker_is_submitted(self, sim):
        result = sim.start_policy(robot_name="arm", policy_provider="nope", duration=0.2)
        assert result["status"] == "error"
        assert sim._policy_threads.get("arm") is None
        assert sim._policy_rates.get("arm") is None

    def test_the_robot_is_left_free_for_a_usable_retry(self, sim):
        refused = sim.start_policy(robot_name="arm", policy_provider="nope", duration=0.2)
        assert refused["status"] == "error"
        # The refusal must not consume the per-robot slot: the obvious retry
        # with a usable provider has to be admitted.
        started = sim.start_policy(robot_name="arm", policy_provider="mock", duration=0.2)
        assert started["status"] == "success", _text(started)
        sim.stop_policy(robot_name="arm")

    def test_a_usable_provider_still_starts(self, sim):
        started = sim.start_policy(robot_name="arm", policy_provider="mock", duration=0.2)
        assert started["status"] == "success", _text(started)
        assert sim._policy_threads.get("arm") is not None
        sim.stop_policy(robot_name="arm")


class TestNoUsableSpellingIsRefused:
    """The guard reuses ``create_policy``'s resolution, so it cannot over-reach."""

    @pytest.mark.parametrize("provider", RESOLVABLE)
    def test_every_resolvable_spelling_passes_the_probe(self, provider):
        assert policy_provider_error(provider) is None, provider

    @pytest.mark.parametrize("provider", UNRESOLVABLE)
    def test_every_unresolvable_spelling_yields_a_reason(self, provider):
        reason = policy_provider_error(provider)
        assert reason is not None and provider is not None
        assert "Unknown policy provider" in reason

    def test_a_resolvable_provider_still_completes_a_rollout(self, sim):
        result = sim.run_policy(robot_name="arm", policy_provider="mock", duration=0.05)
        assert result["status"] == "success", _text(result)


class TestANonStringProviderIsRefusedToo:
    """The same escape, reached by type rather than by name."""

    @pytest.mark.parametrize("provider", NOT_A_STRING, ids=[type(v).__name__ for v in NOT_A_STRING])
    def test_the_probe_names_the_parameter_and_the_type(self, provider):
        reason = policy_provider_error(provider)
        assert reason is not None
        assert "policy_provider must be a string" in reason
        assert type(provider).__name__ in reason
        # The bare TypeError named neither, so its wording must not survive.
        assert "is not iterable" not in reason
        assert "unhashable" not in reason

    @pytest.mark.parametrize("provider", NOT_A_STRING, ids=[type(v).__name__ for v in NOT_A_STRING])
    @pytest.mark.parametrize("action,extra", SURFACES, ids=[s[0] for s in SURFACES])
    def test_every_surface_returns_it_as_an_envelope(self, sim, action, extra, provider):
        result = getattr(sim, action)(robot_name="arm", policy_provider=provider, **extra)
        assert result["status"] == "error", result
        assert _text(result) == policy_provider_error(provider)

    def test_a_string_provider_is_untouched_by_the_type_check(self):
        assert policy_provider_error("mock") is None
        reason = policy_provider_error("nope")
        assert reason is not None and "must be a string" not in reason


class TestTheProbeCostsNoConstruction:
    """Resolution must not instantiate the policy or download weights."""

    def test_the_probe_does_not_construct(self, monkeypatch):
        import strands_robots.policies.factory as factory

        def fatal(*_a: Any, **_kw: Any) -> Any:
            raise AssertionError("the probe constructed a policy")

        monkeypatch.setattr(factory, "create_policy", fatal)
        assert policy_provider_error("mock") is None
        assert policy_provider_error("nope") is not None


class TestTheGuardPrecedesTheSubmit:
    """Pinned structurally: a guard below the submit is a false success again."""

    def test_start_policy_checks_the_provider_before_submitting(self):
        src = inspect.getsource(mujoco_simulation)
        tree = ast.parse(src)
        checked = submitted = None
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "start_policy"):
                continue
            for index, statement in enumerate(node.body):
                if index == 0:
                    continue  # the docstring names these symbols in prose
                segment = ast.get_source_segment(src, statement) or ""
                if "_unresolvable_policy_provider_error" in segment and checked is None:
                    checked = index
                if "self._executor.submit(" in segment and submitted is None:
                    submitted = index
        assert checked is not None, "start_policy does not check the provider"
        assert submitted is not None, "start_policy does not submit"
        assert checked < submitted, f"guard body[{checked}] must precede submit body[{submitted}]"


class TestTheTrustGateIsOutOfScope:
    """The remote-code gate keeps raising: a security control, decided elsewhere.

    ``allenai/MolmoAct2-SO100_101`` resolves, so this guard passes it through to
    ``create_policy``, whose trust-remote-code gate raises. Whether that gate
    should report through the envelope instead is a separate question about a
    security control, and this module deliberately does not answer it.
    """

    def test_a_resolvable_checkpoint_is_not_refused_by_this_guard(self):
        assert policy_provider_error("allenai/MolmoAct2-SO100_101") is None

    def test_the_gate_still_raises(self, monkeypatch):
        from strands_robots.policies import UntrustedRemoteCodeError, create_policy

        monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
        with pytest.raises(UntrustedRemoteCodeError):
            create_policy("allenai/MolmoAct2-SO100_101")
