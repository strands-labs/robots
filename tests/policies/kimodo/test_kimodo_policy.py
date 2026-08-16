"""Unit tests for KimodoPolicy using an injected motion agent.

These tests exercise the frame -> action-dict mapping, prompt-change
resampling, cursor advancement, and end-of-buffer hold semantics WITHOUT
requiring torch/diffusers/CUDA/checkpoints. The real diffusers-backed sampler
is covered by an integration test gated on the ``[kimodo]`` extra + HF access.
"""

from __future__ import annotations

import asyncio
import hashlib

import numpy as np
import pytest

from strands_robots.policies.kimodo import (
    KIMODO_G1_JOINTS,
    KimodoConfig,
    KimodoPolicy,
)


class _StubAgent:
    """Deterministic stub returning a linear ramp per joint."""

    def __init__(self, num_joints: int = 29) -> None:
        self.calls = 0
        self.num_joints = num_joints

    def sample(self, prompt, num_frames, diffusion_steps, guidance_scale, seed):
        self.calls += 1
        out = np.zeros((num_frames, 7 + self.num_joints), dtype=np.float32)
        out[:, 6] = 1.0  # identity quaternion
        for t in range(num_frames):
            out[t, 7:] = np.linspace(0.0, 1.0, self.num_joints) * (t / max(num_frames - 1, 1))
        return out


def _make_policy(**cfg_kwargs):
    cfg = KimodoConfig(**cfg_kwargs)
    stub = _StubAgent()
    return KimodoPolicy(config=cfg, motion_agent=stub), stub


def _first_action(policy, instruction="walk forward", **kw):
    """Convenience: run the async get_actions and return the single dict."""
    return asyncio.run(policy.get_actions({}, instruction, **kw))[0]


def test_get_actions_returns_all_g1_joints():
    policy, _ = _make_policy(num_frames=10, native_fps=30, tracker_fps=30)
    action = _first_action(policy)
    assert set(action.keys()) == set(KIMODO_G1_JOINTS)
    assert all(isinstance(v, float) for v in action.values())


def test_get_actions_returns_single_element_list():
    policy, _ = _make_policy(num_frames=10, native_fps=30, tracker_fps=30)
    result = asyncio.run(policy.get_actions({}, "walk"))
    assert isinstance(result, list) and len(result) == 1


def test_prompt_change_triggers_resample():
    policy, stub = _make_policy(num_frames=8, native_fps=30, tracker_fps=30)
    _first_action(policy, "walk")
    _first_action(policy, "walk")
    assert stub.calls == 1
    _first_action(policy, "run")
    assert stub.calls == 2


def test_cursor_advances_frame_by_frame():
    policy, _ = _make_policy(num_frames=4, native_fps=30, tracker_fps=30)
    first = _first_action(policy)
    second = _first_action(policy)
    changed = sum(1 for k in KIMODO_G1_JOINTS if first[k] != second[k])
    assert changed >= len(KIMODO_G1_JOINTS) - 1  # index 0 stays 0


def test_end_of_buffer_holds_last_frame():
    policy, _ = _make_policy(num_frames=3, native_fps=30, tracker_fps=30)
    for _ in range(3):
        _first_action(policy)
    a = _first_action(policy)
    b = _first_action(policy)
    assert a == b  # last frame held


def test_empty_prompt_raises():
    policy, _ = _make_policy()
    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(policy.get_actions({}, ""))


def test_reset_rewinds_cursor():
    policy, _ = _make_policy(num_frames=4, native_fps=30, tracker_fps=30)
    first = _first_action(policy)
    _first_action(policy)
    policy.reset()
    again = _first_action(policy)
    assert again == first


def test_slerp_upsample_widens_buffer():
    policy, _ = _make_policy(num_frames=30, native_fps=30, tracker_fps=50)
    _first_action(policy)
    assert policy._motion_buffer is not None
    assert policy._motion_buffer.shape[0] > 30


def test_requires_images_false():
    policy, _ = _make_policy()
    assert policy.requires_images is False


def test_provider_name():
    policy, _ = _make_policy()
    assert policy.provider_name == "kimodo"


def test_set_robot_state_keys_ok():
    policy, _ = _make_policy()
    policy.set_robot_state_keys(list(KIMODO_G1_JOINTS))
    assert policy._joint_names == KIMODO_G1_JOINTS


def test_set_robot_state_keys_missing_joint_raises():
    policy, _ = _make_policy()
    bad = list(KIMODO_G1_JOINTS)[:-1]  # missing last joint
    with pytest.raises(ValueError, match="missing expected G1 joints"):
        policy.set_robot_state_keys(bad)


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError, match="diffusion_steps"):
        KimodoConfig(diffusion_steps=0)
    with pytest.raises(ValueError, match="guidance_scale"):
        KimodoConfig(guidance_scale=-1)
    with pytest.raises(ValueError, match="num_frames"):
        KimodoConfig(num_frames=1000)
    with pytest.raises(ValueError, match="dtype"):
        KimodoConfig(dtype="int8")


def test_config_from_dict_drops_unknown_keys():
    cfg = KimodoConfig.from_dict({"diffusion_steps": 50, "unknown": "x"})
    assert cfg.diffusion_steps == 50


# --------------------------------------------------------------------------
# The registry surface has to reach the constructor
# --------------------------------------------------------------------------
# policies.json advertises every KimodoConfig field in the provider's
# ``config_keys``, and create_policy splats those keys straight into the class.
# They were not parameters of __init__, so each one landed in a ``**kwargs``
# that forwarded to ``object.__init__`` - measured, every advertised knob raised
# ``TypeError: object.__init__() takes exactly one argument``, so the whole
# documented configuration surface was unusable through the factory that the
# registry exists to serve.


def test_each_advertised_config_key_reaches_the_config_through_the_factory():
    """Every config_keys entry configures the sampler, rather than raising."""
    from strands_robots.policies.kimodo import KimodoPolicy

    policy = KimodoPolicy(
        model_id="me/custom",
        diffusion_steps=25,
        guidance_scale=1.5,
        num_frames=48,
        native_fps=24,
        tracker_fps=60,
        device="cpu",
        dtype="fp32",
        seed=7,
    )
    assert policy.config.model_id == "me/custom"
    assert policy.config.diffusion_steps == 25
    assert policy.config.guidance_scale == 1.5
    assert policy.config.num_frames == 48
    assert policy.config.native_fps == 24
    assert policy.config.tracker_fps == 60
    assert policy.config.device == "cpu"
    assert policy.config.dtype == "fp32"
    assert policy.config.seed == 7


def test_every_advertised_config_key_is_a_constructor_parameter():
    """The registry list and the signature agree, so neither can drift alone."""
    import inspect

    from strands_robots.policies.kimodo import KimodoPolicy
    from strands_robots.registry.policies import get_policy_provider

    advertised = set(get_policy_provider("kimodo")["config_keys"])
    accepted = set(inspect.signature(KimodoPolicy.__init__).parameters) - {"self"}
    assert advertised <= accepted, f"advertised but not accepted: {sorted(advertised - accepted)}"


def test_an_unknown_knob_raises_rather_than_being_swallowed():
    """No ``**kwargs``: a mistyped parameter is refused at construction.

    The name is derived from the real signature rather than hardcoded, so the
    case cannot silently stop being a typo if a parameter is renamed later.
    """
    import inspect

    from strands_robots.policies.kimodo import KimodoPolicy

    accepted = set(inspect.signature(KimodoPolicy.__init__).parameters)
    unsupported = "diffusion_step"  # 'diffusion_steps' without the plural 's'
    assert unsupported not in accepted, "this name has to be one the constructor rejects"

    with pytest.raises(TypeError, match=unsupported):
        KimodoPolicy(**dict.fromkeys([unsupported], 25))


def test_an_override_is_validated_like_a_directly_constructed_field():
    """The merge re-enters KimodoConfig, so the domain still applies."""
    from strands_robots.policies.kimodo import KimodoPolicy

    with pytest.raises(ValueError, match="diffusion_steps"):
        KimodoPolicy(diffusion_steps=0)
    with pytest.raises(ValueError, match="num_frames"):
        KimodoPolicy(config=KimodoConfig(), num_frames=1000)


def test_an_override_wins_over_the_supplied_config_object():
    """Precedence: the more explicit per-field value beats the config it overlays."""
    from strands_robots.policies.kimodo import KimodoPolicy

    policy = KimodoPolicy(config=KimodoConfig(diffusion_steps=40), diffusion_steps=12)
    assert policy.config.diffusion_steps == 12


def test_a_config_object_survives_the_registry_default_merge(monkeypatch):
    """A caller's config must not be silently overwritten on the factory path.

    ``build_policy_kwargs`` forwards a provider's registry ``defaults``
    unconditionally - it only skips a key the caller passed itself - so a
    default that merely restates a dataclass default still arrives as a flat
    override and, with per-field precedence, would replace the caller's value.
    Measured before this change: a config carrying steps=25/guidance=1.5/
    model='me/custom' reached the policy as 100/7.5/'nvidia/Kimodo-G1-RP-v1'.
    The three redundant defaults are gone from policies.json for that reason.
    """
    from strands_robots.policies.factory import create_policy
    from strands_robots.registry.policies import build_policy_kwargs

    # The provider loads a custom HF sampler class, so the factory gates it.
    monkeypatch.setenv("STRANDS_TRUST_REMOTE_CODE", "1")

    mine = KimodoConfig(diffusion_steps=25, guidance_scale=1.5, model_id="me/custom")
    kwargs = build_policy_kwargs("kimodo", config=mine)
    policy = create_policy("kimodo", **kwargs, motion_agent=_StubAgent())

    assert policy.config.diffusion_steps == 25
    assert policy.config.guidance_scale == 1.5
    assert policy.config.model_id == "me/custom"


def test_config_rejects_a_bool_and_a_non_finite_number():
    """The shared numeric domains reject what a hand-rolled check let through.

    `True` is an `int` subclass, so an isinstance-only check accepted it as a
    silent 1; `nan`/`inf` poison the comparisons these knobs feed. Both are
    refused by the shared domain this config resolves through.
    """
    for bad in (True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="diffusion_steps"):
            KimodoConfig(diffusion_steps=bad)
    for bad in (True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="guidance_scale"):
            KimodoConfig(guidance_scale=bad)


def test_the_prompt_is_logged_by_digest_and_never_echoed(caplog):
    """A caller-supplied prompt must not be able to forge a second log record.

    The sampler identifies the prompt by digest rather than interpolating it, so
    neither the text nor a newline it carries reaches the log line.
    """
    import hashlib
    import logging

    prompt = "walk forward\nINFO:root:forged record"
    policy, _ = _make_policy(num_frames=4, native_fps=30, tracker_fps=30)
    with caplog.at_level(logging.INFO, logger="strands_robots.policies.kimodo.policy"):
        _first_action(policy, prompt)

    logged = [r.getMessage() for r in caplog.records]
    assert logged, "expected the sampler to log one record"
    assert not any("\n" in message for message in logged)
    assert not any("forged record" in message for message in logged)

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    assert any(digest in message for message in logged), "the digest identifies the prompt"


class _SeedAwareAgent:
    """Stub whose frames depend on every sampler input, and which records them.

    The real sampler is stochastic in ``seed`` and sensitive to
    ``diffusion_steps`` / ``guidance_scale``; a stub that ignores them cannot
    show whether an override reached it, so this one derives its frames from all
    of them.
    """

    def __init__(self, num_joints: int = 29) -> None:
        self.calls: list[dict[str, object]] = []
        self.num_joints = num_joints

    def sample(self, prompt, num_frames, diffusion_steps, guidance_scale, seed):
        self.calls.append(
            {
                "prompt": prompt,
                "diffusion_steps": diffusion_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
            }
        )
        # A stable digest, not builtin hash(): PYTHONHASHSEED salts str hashing
        # per process, which would make the reproducibility assertion below hold
        # only within a single interpreter run.
        material = repr((prompt, diffusion_steps, round(float(guidance_scale), 6), seed)).encode("utf-8")
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest()[:4], "big"))
        out = np.zeros((num_frames, 7 + self.num_joints), dtype=np.float32)
        out[:, 6] = 1.0  # identity quaternion
        out[:, 7:] = rng.uniform(-1.0, 1.0, size=(num_frames, self.num_joints)).astype(np.float32)
        return out


def _seeded_policy(**cfg_kwargs):
    agent = _SeedAwareAgent()
    return KimodoPolicy(config=KimodoConfig(**cfg_kwargs), motion_agent=agent), agent


def test_a_new_episode_seed_re_samples_the_motion():
    """``reset(seed=...)`` must reach the sampler, not just rewind the cursor.

    ``PolicyRunner.evaluate`` derives one seed per episode and forwards it here;
    a seed that is recorded but never sampled leaves every episode replaying the
    first episode's motion.
    """
    policy, agent = _seeded_policy(num_frames=8, native_fps=30, tracker_fps=30)
    first = _first_action(policy, "walk forward")

    policy.reset(seed=1234)
    after = _first_action(policy, "walk forward")

    assert [call["seed"] for call in agent.calls] == [None, 1234]
    assert any(first[joint] != after[joint] for joint in KIMODO_G1_JOINTS)


def test_reset_without_a_seed_replays_the_motion_without_re_sampling():
    """A plain rewind must stay cheap - no seed means no new sampler run."""
    policy, agent = _seeded_policy(num_frames=8, native_fps=30, tracker_fps=30)
    first = _first_action(policy, "walk forward")
    _first_action(policy, "walk forward")

    policy.reset()
    again = _first_action(policy, "walk forward")

    assert len(agent.calls) == 1
    assert again == first


def test_repeating_an_episode_seed_replays_instead_of_re_running_the_sampler():
    """The same inputs identify the same motion, so the buffer is reused."""
    policy, agent = _seeded_policy(num_frames=8, native_fps=30, tracker_fps=30)
    policy.reset(seed=99)
    first = _first_action(policy, "walk forward")

    policy.reset(seed=99)
    again = _first_action(policy, "walk forward")

    assert len(agent.calls) == 1
    assert again == first


def test_each_episode_of_a_seeded_eval_samples_at_its_own_seed():
    """Distinct per-episode seeds give distinct motions, reproducibly.

    Mirrors the per-episode ``set_eval_seed`` + ``policy.reset(seed=...)`` loop
    in ``PolicyRunner.evaluate``: episode N must not inherit episode N-1's
    motion, and re-running the same seed sequence must reproduce it.
    """
    episode_seeds = [11, 22, 33]

    def run_episodes():
        policy, agent = _seeded_policy(num_frames=6, native_fps=30, tracker_fps=30)
        motions = []
        for episode_seed in episode_seeds:
            policy.reset(seed=episode_seed)
            motions.append(tuple(_first_action(policy, "walk forward")[j] for j in KIMODO_G1_JOINTS))
        return motions, agent

    motions, agent = run_episodes()

    assert [call["seed"] for call in agent.calls] == episode_seeds
    assert len(set(motions)) == len(episode_seeds), "an episode replayed another episode's motion"
    assert run_episodes()[0] == motions, "the same seed sequence did not reproduce"


def test_a_changed_sampler_knob_is_honoured_after_the_first_call():
    """``diffusion_steps`` / ``guidance_scale`` are documented per-call overrides."""
    policy, agent = _seeded_policy(num_frames=6, native_fps=30, tracker_fps=30)
    _first_action(policy, "walk forward")

    _first_action(policy, "walk forward", diffusion_steps=25)
    _first_action(policy, "walk forward", guidance_scale=2.5)

    assert [call["diffusion_steps"] for call in agent.calls] == [100, 25, 100]
    assert [call["guidance_scale"] for call in agent.calls] == [7.5, 7.5, 2.5]
