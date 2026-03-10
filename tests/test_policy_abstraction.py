"""Tests for the policy abstraction layer and factory."""

import pytest


class TestPolicyRegistry:
    """Test the plugin-based policy registry."""

    def test_list_providers(self):
        from strands_robots.policies import list_providers

        providers = list_providers()
        assert "mock" in providers
        assert "groot" in providers
        assert "lerobot_local" in providers
        assert "dreamgen" in providers
        assert len(providers) >= 10  # We have 18+ providers

    def test_create_mock_policy(self):
        from strands_robots.policies import create_policy

        policy = create_policy("mock")
        assert policy.provider_name == "mock"

    def test_mock_policy_sync(self):
        from strands_robots.policies import create_policy

        policy = create_policy("mock")
        policy.set_robot_state_keys(["joint_0", "joint_1", "joint_2"])
        actions = policy.get_actions_sync({"observation.state": [0, 0, 0]}, "test")
        assert len(actions) == 8
        assert "joint_0" in actions[0]

    def test_register_custom_policy(self):
        from strands_robots.policies import Policy, create_policy, register_policy

        class MyPolicy(Policy):
            @property
            def provider_name(self):
                return "my_custom"

            async def get_actions(self, obs, instr, **kw):
                return [{"a": 1}]

            def set_robot_state_keys(self, keys):
                pass

        register_policy("my_custom", lambda: MyPolicy)
        policy = create_policy("my_custom")
        assert policy.provider_name == "my_custom"

    def test_unknown_provider_raises(self):
        from strands_robots.policies import create_policy

        with pytest.raises(ValueError, match="Unknown policy provider"):
            create_policy("nonexistent_provider_xyz")

    def test_smart_string_resolution(self):
        """Test that create_policy handles HuggingFace model IDs."""
        from strands_robots.policy_resolver import resolve_policy

        # Mock shorthand
        provider, kwargs = resolve_policy("mock")
        assert provider == "mock"

        # Host:port → lerobot_async
        provider, kwargs = resolve_policy("localhost:8080")
        assert provider == "lerobot_async"
        assert kwargs["server_address"] == "localhost:8080"

        # ws:// → dreamzero
        provider, kwargs = resolve_policy("ws://gpu:9000")
        assert provider == "dreamzero"
        assert kwargs["host"] == "gpu"
        assert kwargs["port"] == 9000

        # HuggingFace org → auto-resolve
        provider, kwargs = resolve_policy("lerobot/act_aloha_sim")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "lerobot/act_aloha_sim"

        provider, kwargs = resolve_policy("openvla/openvla-7b")
        assert provider == "openvla"

        provider, kwargs = resolve_policy("microsoft/Magma-8B")
        assert provider == "magma"


class TestFactory:
    """Test the Robot() factory."""

    def test_list_robots(self):
        from strands_robots.factory import list_robots

        robots = list_robots()
        assert len(robots) >= 30  # We have 35+ robots
        names = [r["name"] for r in robots]
        assert "so100" in names
        assert "unitree_g1" in names
        assert "panda" in names
        assert "aloha" in names

    def test_list_robots_filter_sim(self):
        from strands_robots.factory import list_robots

        sim_robots = list_robots(mode="sim")
        for r in sim_robots:
            assert r["has_sim"]

    def test_list_robots_filter_real(self):
        from strands_robots.factory import list_robots

        real_robots = list_robots(mode="real")
        for r in real_robots:
            assert r["has_real"]

    def test_alias_resolution(self):
        from strands_robots.factory import _resolve_name

        assert _resolve_name("g1") == "unitree_g1"
        assert _resolve_name("franka") == "panda"
        assert _resolve_name("h1") == "unitree_h1"
        assert _resolve_name("go2") == "unitree_go2"
        assert _resolve_name("so100_follower") == "so100"


class TestPolicyResolver:
    """Test the policy resolver module."""

    def test_resolve_zmq(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("zmq://myhost:5555")
        assert provider == "groot"
        assert kwargs["host"] == "myhost"
        assert kwargs["port"] == 5555

    def test_resolve_grpc(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("grpc://server:50051")
        assert provider == "lerobot_async"
        assert kwargs["server_address"] == "server:50051"

    def test_resolve_nvidia_models(self):
        from strands_robots.policy_resolver import resolve_policy

        p, _ = resolve_policy("nvidia/cosmos-predict2.5-2b")
        assert p == "cosmos_predict"

        p, _ = resolve_policy("nvidia/alpamayo")
        assert p == "alpamayo"

    def test_resolve_agibot(self):
        from strands_robots.policy_resolver import resolve_policy

        p, _ = resolve_policy("agibot-world/go-1")
        assert p == "go1"
