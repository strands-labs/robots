"""Tests for ``strands_robots.policies.factory.create_policy``.

* provider resolution (mock / groot / lerobot_local)
* ``trust_remote_code`` security gate for HF-backed providers
* kwargs forwarding to the chosen provider
"""

import pytest

from strands_robots.policies import (
    MockPolicy,
    Policy,
    UntrustedRemoteCodeError,
    create_policy,
    list_providers,
    policy_overrides_preflight,
    preflight_policy,
    register_policy,
)

# Detect groot-service availability for conditional test grouping.
try:
    import msgpack  # noqa: F401
    import zmq  # noqa: F401

    _groot_available = True
except ImportError:
    _groot_available = False


class TestCreatePolicy:
    """create_policy() should resolve shorthands, URLs, and custom registrations."""

    def test_register_and_create_custom_provider(self):
        """Runtime-registered providers should be creatable by name and alias."""
        register_policy("custom_test", loader=lambda: MockPolicy, aliases=["ct"])
        p1 = create_policy("custom_test")
        assert isinstance(p1, MockPolicy)
        p2 = create_policy("ct")
        assert isinstance(p2, MockPolicy)

    def test_list_providers_includes_json_and_runtime(self):
        """list_providers() should include both JSON-defined and runtime providers."""
        register_policy("runtime_only_provider", loader=lambda: MockPolicy)
        providers = list_providers()
        assert "mock" in providers
        assert "groot" in providers
        assert "runtime_only_provider" in providers

    def test_unknown_provider_raises(self):
        """Unknown provider should raise, not silently fail."""
        with pytest.raises(Exception):
            create_policy("nonexistent_provider_xyz_123")

    def test_create_mock_by_shorthand(self):
        """All mock shorthands should produce a MockPolicy instance."""
        for name in ("mock", "random", "test"):
            p = create_policy(name)
            assert isinstance(p, MockPolicy), f"'{name}' did not create MockPolicy"

    def test_create_passes_kwargs_to_policy(self):
        """kwargs given to create_policy should reach the Policy constructor."""
        register_policy("kwarg_test", loader=lambda: _KwargCapture, aliases=[])
        p = create_policy("kwarg_test", some_key="some_val")
        assert p.captured == {"some_key": "some_val"}

    def test_create_via_hf_model_id_triggers_smart_resolution(self):
        """An org/model string should trigger smart-string resolution."""
        with pytest.raises(Exception):
            create_policy("unknownorg/somemodel")

    def test_create_via_grpc_url_triggers_smart_resolution(self):
        """A grpc:// URL should trigger smart-string resolution."""
        with pytest.raises(Exception):
            create_policy("grpc://localhost:50051")

    def test_create_via_ws_url_resolves_to_remote_policy(self):
        """A ws:// URL resolves to the remote-inference provider (RemotePolicy).

        Construction is lazy (the WebSocket connects on first use), so this
        succeeds without a running server and preserves the full endpoint URL.
        """
        from strands_robots.inference import RemotePolicy

        policy = create_policy("ws://localhost:8080")
        assert isinstance(policy, RemotePolicy)
        assert policy.uri == "ws://localhost:8080"


@pytest.mark.skipif(not _groot_available, reason="groot-service extras not installed")
class TestFactoryGrootIntegration:
    """Factory tests that require groot-service extras (zmq, msgpack).

    Grouped into a single class with a class-level skip marker so future
    contributors don't need to remember per-test decorators.
    """

    def test_create_via_zmq_url_resolves_to_groot(self):
        """A zmq:// URL should resolve to a Gr00tPolicy via smart-string resolution."""
        from strands_robots.policies.groot import Gr00tPolicy

        p = create_policy("zmq://localhost:5555")
        assert isinstance(p, Gr00tPolicy)

    def test_groot_strict_and_api_token_passthrough(self):
        """strict and api_token kwargs should reach Gr00tPolicy constructor."""
        from strands_robots.policies.groot import Gr00tPolicy

        p = create_policy("zmq://localhost:5555", strict=True, api_token="test-token")
        assert isinstance(p, Gr00tPolicy)
        assert p._strict is True
        assert p._client.api_token == "test-token"

    def test_groot_defaults_strict_false(self):
        """strict should default to False for production use."""
        p = create_policy("zmq://localhost:5555")
        assert p._strict is False

    def test_groot_direct_construction_with_new_params(self):
        """Direct Gr00tPolicy() should accept strict and api_token."""
        from strands_robots.policies.groot import Gr00tPolicy

        p = Gr00tPolicy(host="localhost", port=5555, strict=True, api_token="s3cret")
        assert p._strict is True
        assert p._mode == "service"
        assert p._client.api_token == "s3cret"


class TestTrustRemoteCodeGate:
    """STRANDS_TRUST_REMOTE_CODE gate should block lerobot_local without opt-in."""

    def test_lerobot_local_blocked_without_env(self, monkeypatch):
        """create_policy('lerobot_local') should raise without STRANDS_TRUST_REMOTE_CODE."""
        monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
        with pytest.raises(UntrustedRemoteCodeError):
            create_policy("lerobot_local")

    def test_lerobot_local_allowed_with_env(self, monkeypatch):
        """create_policy('lerobot_local') should succeed with STRANDS_TRUST_REMOTE_CODE=1."""
        monkeypatch.setenv("STRANDS_TRUST_REMOTE_CODE", "1")
        p = create_policy("lerobot_local")
        assert p.provider_name == "lerobot_local"

    def test_mock_never_gated(self, monkeypatch):
        """Mock provider should never be blocked by trust gate."""
        monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
        p = create_policy("mock")
        assert isinstance(p, MockPolicy)

    def test_runtime_registered_not_gated(self, monkeypatch):
        """Runtime-registered providers (not in HF list) should not be gated."""
        monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
        register_policy("safe_custom", loader=lambda: MockPolicy, aliases=["sc"])
        p = create_policy("safe_custom")
        assert isinstance(p, MockPolicy)


class TestSmartResolutionFallThrough:
    """When smart-string resolution fails, create_policy must fall through to
    the standard policies.json lookup rather than aborting on the resolver.

    A smart string (HF id, ws://, grpc://, zmq:// ...) first goes through
    ``resolve_policy``. Two failure modes are contractually swallowed so a
    resolver hiccup never masks a provider that the static registry can still
    serve:

    * ``ImportError`` -- an optional resolver backend is missing; resolution is
      skipped silently and the provider is tried as a plain registry name.
    * any other ``Exception`` -- logged as a warning, then the same fall-through.

    These paths are pure error-handling glue with no happy-path test, so a
    regression (e.g. letting the resolver error escape) would silently break
    provider creation for everyone whose provider is resolvable only via the
    static registry. The tests below pin both fall-through branches by forcing
    ``resolve_policy`` to raise and asserting the static lookup still wins.
    """

    def test_resolution_importerror_is_swallowed_not_propagated(self, monkeypatch):
        """An ImportError from resolve_policy must not escape: create_policy
        falls through to the static lookup, which raises its own (different)
        error for an unknown provider. The surfaced error therefore comes from
        the static lookup, proving the resolver ImportError was swallowed."""

        def boom(provider, **kwargs):
            raise ImportError("optional resolver backend missing -- do not surface")

        monkeypatch.setattr("strands_robots.policies.factory.resolve_policy", boom)
        # "/" trips _needs_resolution; resolver raises ImportError; fall-through
        # hits the static lookup for an unknown provider and raises from there.
        with pytest.raises(Exception) as ei:
            create_policy("unknownorg/doesnotexist")
        assert "do not surface" not in str(ei.value), "resolver ImportError leaked instead of falling through to lookup"

    def test_resolution_generic_error_falls_through_and_warns(self, monkeypatch, caplog):
        """A non-ImportError from resolve_policy is logged at WARNING and then
        swallowed, after which the static registry lookup proceeds."""
        import logging

        def boom(provider, **kwargs):
            raise RuntimeError("resolver exploded")

        monkeypatch.setattr("strands_robots.policies.factory.resolve_policy", boom)
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.factory"):
            with pytest.raises(Exception):
                # "/" triggers resolution; resolver raises; fall-through then
                # fails to find an unknown provider -> raises from static lookup,
                # proving the resolver error itself did not escape.
                create_policy("unknownorg/doesnotexist")
        assert any(
            "resolution failed" in r.getMessage().lower() or "resolver exploded" in r.getMessage()
            for r in caplog.records
        ), "expected a WARNING log for the swallowed resolver error"

    def test_resolution_importerror_does_not_warn(self, monkeypatch, caplog):
        """The ImportError branch is silent (optional backend absent is normal):
        it must NOT emit the generic 'resolution failed' warning."""
        import logging

        def boom(provider, **kwargs):
            raise ImportError("backend absent")

        monkeypatch.setattr("strands_robots.policies.factory.resolve_policy", boom)
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.factory"):
            with pytest.raises(Exception):
                create_policy("unknownorg/doesnotexist")
        assert not any("resolution failed" in r.getMessage().lower() for r in caplog.records), (
            "ImportError branch must stay silent, not log 'resolution failed'"
        )


class _KwargCapture(Policy):
    """Test helper -- captures kwargs for verification."""

    def __init__(self, **kwargs):
        self.captured = kwargs

    async def get_actions(self, observation_dict, instruction, **kwargs):
        return []

    def set_robot_state_keys(self, robot_state_keys):
        pass

    @property
    def provider_name(self):
        return "kwarg_test"


class _PreflightPolicy(MockPolicy):
    """Test helper -- a provider that overrides the class-level ``preflight``
    hook to reject a runtime observation missing a required camera key.

    Records every ``preflight`` invocation so tests can assert the observation
    keys and provider kwargs were forwarded verbatim.
    """

    preflight_calls: list[tuple[set[str], dict]] = []

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: object) -> None:
        cls.preflight_calls.append((set(observation_keys), dict(policy_config)))
        if "camera_top" not in observation_keys:
            raise ValueError("preflight: required image source 'camera_top' is absent")


class TestPreflightPolicy:
    """``preflight_policy`` is the fail-fast seam ``run_policy`` / ``eval_policy``
    call BEFORE ``create_policy`` downloads weights. It must: resolve a provider
    without instantiating it, invoke the class ``preflight`` hook only when the
    provider overrides it, propagate that hook's ``ValueError``, and swallow
    resolution failures (the authoritative error comes later from
    ``create_policy``).
    """

    @pytest.fixture(autouse=True)
    def _reset_preflight_calls(self):
        _PreflightPolicy.preflight_calls.clear()
        yield
        _PreflightPolicy.preflight_calls.clear()

    def test_noop_when_provider_does_not_override_preflight(self):
        """A provider using the default no-op ``preflight`` (e.g. ``mock``) must
        pass silently -- no exception, no hook side effects."""
        assert preflight_policy("mock", {"joint_0", "joint_1"}) is None

    def test_propagates_valueerror_from_overridden_preflight(self):
        """When the resolved provider overrides ``preflight`` and rejects the
        observation keys, ``preflight_policy`` must surface that ``ValueError``
        (this is the whole point of the fail-fast seam)."""
        register_policy("preflight_reject", loader=lambda: _PreflightPolicy)
        with pytest.raises(ValueError, match="camera_top"):
            preflight_policy("preflight_reject", {"joint_0"})

    def test_passes_when_overridden_preflight_accepts_keys(self):
        """A satisfied override returns ``None`` -- and was actually invoked."""
        register_policy("preflight_accept", loader=lambda: _PreflightPolicy)
        assert preflight_policy("preflight_accept", {"joint_0", "camera_top"}) is None
        assert len(_PreflightPolicy.preflight_calls) == 1

    def test_forwards_observation_keys_and_policy_config(self):
        """The runtime observation keys and provider kwargs must reach the hook
        unchanged -- a dropped kwarg would defeat config-dependent validation."""
        register_policy("preflight_capture", loader=lambda: _PreflightPolicy)
        preflight_policy(
            "preflight_capture",
            {"camera_top", "joint_0"},
            image_key="camera_top",
            temperature=0.5,
        )
        assert len(_PreflightPolicy.preflight_calls) == 1
        keys, config = _PreflightPolicy.preflight_calls[0]
        assert keys == {"camera_top", "joint_0"}
        assert config == {"image_key": "camera_top", "temperature": 0.5}

    def test_swallows_resolution_failure(self):
        """An unresolvable provider must NOT raise here: resolution errors are
        surfaced authoritatively by the subsequent ``create_policy`` call, so
        the preflight seam degrades to a no-op instead of masking that error."""
        assert preflight_policy("nonexistent_provider_xyz_123", {"joint_0"}) is None


class TestPolicyOverridesPreflight:
    """``policy_overrides_preflight`` answers whether ``preflight_policy`` will
    read its ``observation_keys`` argument, so a caller can decide whether to
    pay for producing it. The simulation's preflight sources those keys from a
    ``get_observation`` that renders every camera in the scene.
    """

    @pytest.mark.parametrize(
        ("provider", "overrides"),
        [
            ("mock", False),
            ("lerobot_local", True),
            ("nonexistent_provider_xyz_123", False),
        ],
    )
    def test_the_shipped_verdicts(self, provider, overrides):
        """``lerobot_local`` is the one shipped provider with a real hook. An
        unresolvable name reports no hook, matching ``preflight_policy``, which
        degrades to a no-op for a name it cannot resolve.
        """
        assert policy_overrides_preflight(provider) is overrides

    def test_a_registered_override_is_reported(self):
        register_policy("preflight_overrides_probe", loader=lambda: _PreflightPolicy)
        assert policy_overrides_preflight("preflight_overrides_probe") is True

    def test_the_answer_agrees_with_whether_the_hook_runs(self):
        """The two functions must never disagree: a provider reported as having
        no override must also leave ``_PreflightPolicy``'s recorder untouched.
        """
        register_policy("preflight_agreement_probe", loader=lambda: _PreflightPolicy)
        _PreflightPolicy.preflight_calls.clear()

        assert policy_overrides_preflight("preflight_agreement_probe") is True
        preflight_policy("preflight_agreement_probe", {"joint_0", "camera_top"})
        assert len(_PreflightPolicy.preflight_calls) == 1

        assert policy_overrides_preflight("mock") is False
        preflight_policy("mock", {"joint_0", "camera_top"})
        assert len(_PreflightPolicy.preflight_calls) == 1
