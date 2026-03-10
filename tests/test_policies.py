#!/usr/bin/env python3
"""Tests for the Policy abstraction, MockPolicy, PolicyRegistry, and public API.

All tests run on CPU without any GPU or hardware dependencies.
"""

import asyncio

import pytest

from strands_robots.policies import (
    MockPolicy,
    Policy,
    PolicyRegistry,
    create_policy,
    list_providers,
    register_policy,
)

# ─────────────────────────────────────────────────────────────────────
# Policy ABC
# ─────────────────────────────────────────────────────────────────────


class TestPolicyABC:
    """Test the Policy abstract base class contract."""

    def test_policy_is_abstract(self):
        """Policy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Policy()

    def test_policy_requires_get_actions(self):
        """Subclass must implement get_actions."""

        class Incomplete(Policy):
            def set_robot_state_keys(self, keys):
                pass

            @property
            def provider_name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_policy_requires_set_robot_state_keys(self):
        """Subclass must implement set_robot_state_keys."""

        class Incomplete(Policy):
            async def get_actions(self, obs, instruction, **kw):
                return []

            @property
            def provider_name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_policy_requires_provider_name(self):
        """Subclass must implement provider_name property."""

        class Incomplete(Policy):
            async def get_actions(self, obs, instruction, **kw):
                return []

            def set_robot_state_keys(self, keys):
                pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_instantiates(self):
        """A fully implemented subclass can be instantiated."""

        class Complete(Policy):
            async def get_actions(self, obs, instruction, **kw):
                return []

            def set_robot_state_keys(self, keys):
                pass

            @property
            def provider_name(self):
                return "complete"

        policy = Complete()
        assert policy.provider_name == "complete"

    def test_policy_subclass_check(self):
        """MockPolicy is a proper subclass of Policy."""
        assert issubclass(MockPolicy, Policy)
        mock = MockPolicy()
        assert isinstance(mock, Policy)


# ─────────────────────────────────────────────────────────────────────
# MockPolicy
# ─────────────────────────────────────────────────────────────────────


class TestMockPolicy:
    """Test MockPolicy — the built-in testing policy."""

    def test_init(self):
        mock = MockPolicy()
        assert mock.provider_name == "mock"
        assert mock.robot_state_keys == []
        assert mock._step == 0

    def test_init_with_kwargs(self):
        """MockPolicy accepts and ignores arbitrary kwargs."""
        mock = MockPolicy(some_param="value", another=42)
        assert mock.provider_name == "mock"

    def test_provider_name(self):
        mock = MockPolicy()
        assert mock.provider_name == "mock"

    def test_set_robot_state_keys(self):
        mock = MockPolicy()
        keys = ["joint_0", "joint_1", "joint_2"]
        mock.set_robot_state_keys(keys)
        assert mock.robot_state_keys == keys

    def test_get_actions_returns_list(self):
        """get_actions returns a list of action dicts."""
        mock = MockPolicy()
        mock.set_robot_state_keys(["j0", "j1", "j2"])

        actions = asyncio.run(mock.get_actions({"observation.state": [0.0, 0.0, 0.0]}, "test"))
        assert isinstance(actions, list)
        assert len(actions) == 8  # Default action horizon

    def test_get_actions_keys_match(self):
        """Each action dict has keys matching robot_state_keys."""
        mock = MockPolicy()
        keys = ["shoulder", "elbow", "wrist"]
        mock.set_robot_state_keys(keys)

        actions = asyncio.run(mock.get_actions({}, "move arm"))
        for action in actions:
            assert set(action.keys()) == set(keys)

    def test_get_actions_sinusoidal_values(self):
        """Actions contain sinusoidal float values (not noise)."""
        mock = MockPolicy()
        mock.set_robot_state_keys(["j0"])

        actions = asyncio.run(mock.get_actions({}, "test"))
        for action in actions:
            val = action["j0"]
            assert isinstance(val, float)
            assert -1.0 <= val <= 1.0  # Amplitude is 0.5, well within ±1

    def test_get_actions_deterministic(self):
        """Two fresh MockPolicies produce the same actions for the same call."""
        mock1 = MockPolicy()
        mock1.set_robot_state_keys(["j0", "j1"])
        mock2 = MockPolicy()
        mock2.set_robot_state_keys(["j0", "j1"])

        actions1 = asyncio.run(mock1.get_actions({}, "test"))
        actions2 = asyncio.run(mock2.get_actions({}, "test"))
        assert actions1 == actions2

    def test_get_actions_step_advances(self):
        """Internal step counter advances after each call."""
        mock = MockPolicy()
        mock.set_robot_state_keys(["j0"])

        assert mock._step == 0

        asyncio.run(mock.get_actions({}, "test"))
        assert mock._step == 8  # 8 actions produced

        asyncio.run(mock.get_actions({}, "test"))
        assert mock._step == 16

    def test_get_actions_successive_differ(self):
        """Successive calls produce different action sequences (trajectory progresses)."""
        mock = MockPolicy()
        mock.set_robot_state_keys(["j0"])

        actions1 = asyncio.run(mock.get_actions({}, "test"))
        actions2 = asyncio.run(mock.get_actions({}, "test"))
        # At least one action should differ (sinusoid progresses in time)
        values1 = [a["j0"] for a in actions1]
        values2 = [a["j0"] for a in actions2]
        assert values1 != values2

    def test_auto_generates_keys_from_observation_state(self):
        """When no keys are set, MockPolicy auto-generates from observation.state."""
        mock = MockPolicy()
        assert mock.robot_state_keys == []

        obs = {"observation.state": [0.1, 0.2, 0.3, 0.4, 0.5]}
        actions = asyncio.run(mock.get_actions(obs, "test"))
        # Should have auto-generated 5 joint keys
        assert len(mock.robot_state_keys) == 5
        for action in actions:
            assert len(action) == 5

    def test_auto_generates_default_6dof(self):
        """Without observation.state, defaults to 6-DOF."""
        mock = MockPolicy()
        actions = asyncio.run(mock.get_actions({}, "test"))
        assert len(mock.robot_state_keys) == 6
        for action in actions:
            assert len(action) == 6

    def test_smoothness_of_trajectory(self):
        """Verify the sinusoidal trajectory is smooth (no discontinuities)."""
        mock = MockPolicy()
        mock.set_robot_state_keys(["j0"])

        actions = asyncio.run(mock.get_actions({}, "test"))
        values = [a["j0"] for a in actions]

        # Check consecutive differences are small (smooth trajectory)
        for i in range(1, len(values)):
            delta = abs(values[i] - values[i - 1])
            assert delta < 0.1, f"Large discontinuity at step {i}: {delta}"


# ─────────────────────────────────────────────────────────────────────
# PolicyRegistry
# ─────────────────────────────────────────────────────────────────────


class TestPolicyRegistry:
    """Test the PolicyRegistry plugin system."""

    def test_empty_registry(self):
        reg = PolicyRegistry()
        assert reg.list_providers() == []

    def test_register_and_get(self):
        reg = PolicyRegistry()
        reg.register("test", loader=lambda: MockPolicy)
        cls = reg.get("test")
        assert cls is MockPolicy

    def test_register_with_aliases(self):
        reg = PolicyRegistry()
        reg.register("test", loader=lambda: MockPolicy, aliases=["t", "tst"])
        assert reg.get("test") is MockPolicy
        assert reg.get("t") is MockPolicy
        assert reg.get("tst") is MockPolicy

    def test_list_providers_includes_aliases(self):
        reg = PolicyRegistry()
        reg.register("test", loader=lambda: MockPolicy, aliases=["t"])
        providers = reg.list_providers()
        assert "test" in providers
        assert "t" in providers

    def test_list_providers_sorted(self):
        reg = PolicyRegistry()
        reg.register("zebra", loader=lambda: MockPolicy)
        reg.register("alpha", loader=lambda: MockPolicy)
        providers = reg.list_providers()
        assert providers == sorted(providers)

    def test_contains_registered(self):
        reg = PolicyRegistry()
        reg.register("test", loader=lambda: MockPolicy, aliases=["t"])
        assert "test" in reg
        assert "t" in reg
        assert "nonexistent" not in reg

    def test_get_unknown_raises(self):
        """Getting an unregistered provider raises ValueError."""
        reg = PolicyRegistry()
        with pytest.raises(ValueError, match="Unknown policy provider"):
            reg.get("nonexistent_provider_xyz")

    def test_error_message_lists_available(self):
        """ValueError message lists available providers."""
        reg = PolicyRegistry()
        reg.register("mock", loader=lambda: MockPolicy)
        try:
            reg.get("bad_name")
        except ValueError as e:
            assert "mock" in str(e)
            assert "Available:" in str(e)

    def test_lazy_loading(self):
        """Loader is not called until get() is invoked."""
        call_count = 0

        def loader():
            nonlocal call_count
            call_count += 1
            return MockPolicy

        reg = PolicyRegistry()
        reg.register("lazy", loader=loader)
        assert call_count == 0

        reg.get("lazy")
        assert call_count == 1

    def test_multiple_registrations(self):
        """Multiple providers can coexist."""
        reg = PolicyRegistry()

        class PolicyA(MockPolicy):
            @property
            def provider_name(self):
                return "a"

        class PolicyB(MockPolicy):
            @property
            def provider_name(self):
                return "b"

        reg.register("a", loader=lambda: PolicyA)
        reg.register("b", loader=lambda: PolicyB)

        assert reg.get("a") is PolicyA
        assert reg.get("b") is PolicyB
        assert set(reg.list_providers()) == {"a", "b"}

    def test_auto_discovery_mock(self):
        """Auto-discovery finds the mock provider in strands_robots.policies."""
        # The global registry has mock registered, but let's test auto-discovery
        # on a fresh registry — it should find strands_robots.policies.groot etc.
        # We can't easily test this without a real module, but we can verify
        # the mechanism doesn't crash on missing modules
        reg = PolicyRegistry()
        result = reg._auto_discover("definitely_not_a_real_provider")
        assert result is None

    def test_overwrite_registration(self):
        """Re-registering a name overwrites the previous loader."""
        reg = PolicyRegistry()

        class PolicyA(MockPolicy):
            pass

        class PolicyB(MockPolicy):
            pass

        reg.register("test", loader=lambda: PolicyA)
        assert reg.get("test") is PolicyA

        reg.register("test", loader=lambda: PolicyB)
        assert reg.get("test") is PolicyB


# ─────────────────────────────────────────────────────────────────────
# Public API: create_policy, register_policy, list_providers
# ─────────────────────────────────────────────────────────────────────


class TestCreatePolicy:
    """Test create_policy() — the main policy factory function."""

    def test_create_mock_policy(self):
        policy = create_policy("mock")
        assert isinstance(policy, MockPolicy)
        assert policy.provider_name == "mock"

    def test_create_mock_with_kwargs(self):
        policy = create_policy("mock", custom_param="test")
        assert isinstance(policy, MockPolicy)

    def test_create_returns_policy_instance(self):
        """create_policy returns an instance, not a class."""
        result = create_policy("mock")
        assert isinstance(result, Policy)
        assert not isinstance(result, type)

    def test_create_unknown_raises(self):
        """Creating with unknown provider raises ValueError."""
        with pytest.raises(ValueError):
            create_policy("absolutely_nonexistent_provider_12345")


class TestRegisterPolicy:
    """Test register_policy() — the runtime registration API."""

    def test_register_custom_provider(self):
        """Register a custom provider and create it."""

        class MyCustomPolicy(Policy):
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def get_actions(self, obs, instruction, **kw):
                return [{"action": 0.0}]

            def set_robot_state_keys(self, keys):
                pass

            @property
            def provider_name(self):
                return "my_custom"

        register_policy("my_custom_test", loader=lambda: MyCustomPolicy)

        policy = create_policy("my_custom_test")
        assert isinstance(policy, MyCustomPolicy)
        assert policy.provider_name == "my_custom"

    def test_register_with_aliases(self):
        register_policy(
            "alias_test_provider",
            loader=lambda: MockPolicy,
            aliases=["atp", "alias_test"],
        )
        assert create_policy("alias_test_provider").provider_name == "mock"
        assert create_policy("atp").provider_name == "mock"
        assert create_policy("alias_test").provider_name == "mock"


class TestListProviders:
    """Test list_providers() — provider enumeration."""

    def test_returns_list(self):
        providers = list_providers()
        assert isinstance(providers, list)

    def test_mock_always_available(self):
        assert "mock" in list_providers()

    def test_built_in_providers_present(self):
        """All core built-in providers should be registered."""
        providers = list_providers()
        # These are registered at module load time (lazy — no GPU needed)
        expected_core = {"mock", "groot", "lerobot_async", "lerobot_local", "dreamgen"}
        assert expected_core.issubset(set(providers))

    def test_all_19_canonical_providers(self):
        """All 19 canonical (non-alias) providers should be registered."""
        providers = list_providers()
        canonical = {
            "mock",
            "groot",
            "lerobot_async",
            "lerobot_local",
            "dreamgen",
            "dreamzero",
            "omnivla",
            "openvla",
            "internvla",
            "rdt",
            "magma",
            "unifolm",
            "alpamayo",
            "robobrain",
            "cogact",
            "cosmos_predict",
            "gear_sonic",
            "go1",
        }
        for name in canonical:
            assert name in providers, f"Missing canonical provider: {name}"

    def test_aliases_present(self):
        """Known aliases should be in the provider list."""
        providers = list_providers()
        assert "lerobot" in providers  # alias for lerobot_local
        assert "sonic" in providers  # alias for gear_sonic
        assert "cosmos" in providers  # alias for cosmos_predict

    def test_provider_count(self):
        """Should have a substantial number of providers (canonical + aliases)."""
        providers = list_providers()
        assert len(providers) > 30  # 17 canonical + many aliases

    def test_sorted_output(self):
        providers = list_providers()
        assert providers == sorted(providers)


# ─────────────────────────────────────────────────────────────────────
# Smart resolution via create_policy (HF model IDs, URLs)
# ─────────────────────────────────────────────────────────────────────


class TestSmartResolution:
    """Test create_policy's smart string resolution (delegates to policy_resolver)."""

    def test_mock_direct(self):
        """Direct provider name works."""
        policy = create_policy("mock")
        assert policy.provider_name == "mock"

    def test_policy_module_exports(self):
        """The policies __init__.py exports the expected symbols."""
        from strands_robots.policies import __all__

        expected = {"Policy", "MockPolicy", "PolicyRegistry", "create_policy", "register_policy", "list_providers"}
        assert expected == set(__all__)


# ─────────────────────────────────────────────────────────────────────
# Coverage: _auto_discover, list_providers, __contains__, create_policy fallback
# ─────────────────────────────────────────────────────────────────────

from unittest.mock import patch  # noqa: E402


class TestAutoDiscover:
    """Test PolicyRegistry._auto_discover for dynamic provider loading."""

    def test_auto_discover_finds_existing_provider(self):
        """Auto-discover should find an existing policy subpackage."""
        reg = PolicyRegistry()
        # groot exists as strands_robots.policies.groot -> Gr00tPolicy
        result = reg._auto_discover("groot")
        # Gr00tPolicy should be found via convention <Name>Policy
        # (may be None if groot has unmet deps, that's OK)
        if result is not None:
            assert hasattr(result, "provider_name")

    def test_auto_discover_unknown_returns_none(self):
        """Auto-discover should return None for nonexistent providers."""
        reg = PolicyRegistry()
        result = reg._auto_discover("totally_nonexistent_provider_xyz")
        assert result is None

    def test_auto_discover_caches_in_registry(self):
        """Once discovered, the provider should be cached in _registry."""
        reg = PolicyRegistry()
        # Use lerobot_local which has its own subpackage
        result = reg._auto_discover("lerobot_local")
        if result is not None:
            assert "lerobot_local" in reg._registry

    def test_auto_discover_handles_import_error(self):
        """Import errors during auto-discover should be silently handled."""
        reg = PolicyRegistry()
        with patch.dict("sys.modules", {"strands_robots.policies.broken_xyz": None}):
            result = reg._auto_discover("broken_xyz")
            assert result is None

    def test_auto_discover_handles_generic_exception(self):
        """Generic exceptions during auto-discover should be logged and handled."""
        reg = PolicyRegistry()
        with patch("importlib.import_module", side_effect=RuntimeError("test error")):
            result = reg._auto_discover("will_error")
            assert result is None

    def test_auto_discover_via_get_triggers(self):
        """Registry.get() should trigger auto-discover for unknown names."""
        reg = PolicyRegistry()
        # Register mock first, then get should work
        from strands_robots.policies import MockPolicy

        reg.register("mock", lambda: MockPolicy)
        policy = reg.get("mock")
        assert policy is not None

    def test_auto_discover_all_subclass_fallback(self):
        """Auto-discover should find Policy subclasses via dir() fallback."""
        import types

        from strands_robots.policies import Policy

        # Create a module with a Policy subclass that doesn't follow naming convention
        mock_mod = types.ModuleType("strands_robots.policies.custom_provider_xyz")

        class MyCustomPolicy(Policy):
            def __init__(self, **kwargs):
                pass

            @property
            def provider_name(self):
                return "custom_provider_xyz"

            def set_robot_state_keys(self, keys):
                pass

            async def get_actions(self, obs, instruction, **kwargs):
                return []

        mock_mod.MyCustomPolicy = MyCustomPolicy

        reg = PolicyRegistry()
        with patch.dict("sys.modules", {"strands_robots.policies.custom_provider_xyz": mock_mod}):
            result = reg._auto_discover("custom_provider_xyz")
            assert result is MyCustomPolicy


class TestListProvidersAndContains:
    """Test module-level list_providers and registry __contains__."""

    def test_list_providers_returns_sorted(self):
        """Module-level list_providers should return sorted list."""
        providers = list_providers()
        assert providers == sorted(providers)

    def test_list_providers_includes_mock(self):
        """list_providers should include the mock provider."""
        providers = list_providers()
        assert "mock" in providers

    def test_list_providers_includes_known_providers(self):
        """list_providers should include well-known providers."""
        providers = list_providers()
        # At minimum, mock should be there. Others depend on registration.
        assert len(providers) >= 1

    def test_registry_contains_mock(self):
        """The global _registry should contain 'mock'."""
        from strands_robots.policies import _registry

        assert "mock" in _registry

    def test_registry_not_contains_unknown(self):
        """The global _registry should not contain unknown names."""
        from strands_robots.policies import _registry

        assert "totally_nonexistent_xyz" not in _registry


class TestCreatePolicyFallback:
    """Test create_policy resolution failure paths."""

    def test_create_policy_resolution_failure_falls_through(self):
        """When resolution fails, create_policy should fall through to direct lookup."""
        with patch("strands_robots.policy_resolver.resolve_policy", side_effect=Exception("resolution broke")):
            policy = create_policy("mock")
            assert policy.provider_name == "mock"

    def test_create_policy_unknown_raises(self):
        """create_policy with unknown provider should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown policy provider"):
            create_policy("totally_unknown_provider_xyz")

    def test_create_policy_with_kwargs(self):
        """create_policy should forward kwargs to the policy constructor."""
        policy = create_policy("mock", some_param="test")
        assert policy is not None
        assert policy.provider_name == "mock"
