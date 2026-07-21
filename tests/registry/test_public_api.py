"""Tests for strands_robots.registry - tests for loader, policies, and robots modules."""

import pytest

from strands_robots.registry import get_policy_provider, list_policy_providers, resolve_policy
from strands_robots.registry.loader import _load, _validate, reload
from strands_robots.registry.policies import build_policy_kwargs, import_policy_class
from strands_robots.registry.robots import (
    LIST_ROBOTS_MODES,
    format_robot_table,
    get_hardware_type,
    get_robot,
    has_hardware,
    has_sim,
    list_aliases,
    list_robots,
    list_robots_by_category,
    resolve_name,
)

# Loader tests


class TestLoader:
    """loader.py - JSON loading, caching, hot-reload, and validation."""

    def test_load_caches_and_returns_same_object(self):
        """Consecutive loads without file change should return cached data."""
        first = _load("policies")
        second = _load("policies")
        assert first is second  # same object identity = cache hit

    def test_reload_clears_cache(self):
        """reload() should force re-read on next _load()."""
        first = _load("policies")
        reload()
        after = _load("policies")
        assert first == after

    def test_load_missing_file_returns_empty(self):
        """Missing JSON file should return {} without crashing."""
        result = _load("nonexistent_file_xyz")
        assert result == {}

    def test_validate_duplicate_robot_alias_raises(self):
        """Duplicate robot aliases across entries should raise ValueError."""
        bad_data = {
            "robots": {
                "robot_a": {"aliases": ["shared"]},
                "robot_b": {"aliases": ["shared"]},
            }
        }
        with pytest.raises(ValueError, match="Duplicate robot alias"):
            _validate("robots", bad_data)

    def test_validate_alias_collides_with_canonical_name(self):
        """An alias that matches another robot's canonical name should raise."""
        bad_data = {
            "robots": {
                "robot_a": {"aliases": ["robot_b"]},
                "robot_b": {"aliases": []},
            }
        }
        with pytest.raises(ValueError, match="collides with a canonical robot name"):
            _validate("robots", bad_data)

    def test_validate_duplicate_policy_alias_raises(self):
        """Duplicate policy aliases should raise ValueError."""
        bad_data = {
            "providers": {
                "prov_a": {"aliases": ["dup"], "shorthands": [], "url_patterns": []},
                "prov_b": {"aliases": ["dup"], "shorthands": [], "url_patterns": []},
            }
        }
        with pytest.raises(ValueError, match="Duplicate policy alias"):
            _validate("policies", bad_data)

    def test_validate_duplicate_policy_shorthand_raises(self):
        """Duplicate shorthands across providers should raise."""
        bad_data = {
            "providers": {
                "prov_a": {"aliases": ["sh"], "shorthands": [], "url_patterns": []},
                "prov_b": {"aliases": [], "shorthands": ["sh"], "url_patterns": []},
            }
        }
        with pytest.raises(ValueError, match="Duplicate policy shorthand"):
            _validate("policies", bad_data)

    def test_validate_duplicate_url_pattern_raises(self):
        """Duplicate URL patterns across providers should raise."""
        bad_data = {
            "providers": {
                "prov_a": {"aliases": [], "shorthands": [], "url_patterns": ["^zmq://"]},
                "prov_b": {"aliases": [], "shorthands": [], "url_patterns": ["^zmq://"]},
            }
        }
        with pytest.raises(ValueError, match="Duplicate URL pattern"):
            _validate("policies", bad_data)

    def test_validate_policy_alias_collides_with_canonical_name_raises(self):
        """An alias matching a different provider's canonical name should raise.

        get_policy_provider resolves through the alias map before the canonical
        name, so an alias equal to another provider's name silently shadows it.
        """
        bad_data = {
            "providers": {
                "prov_a": {"aliases": ["prov_b"], "shorthands": [], "url_patterns": []},
                "prov_b": {"aliases": [], "shorthands": [], "url_patterns": []},
            }
        }
        with pytest.raises(ValueError, match="collides with a canonical provider name"):
            _validate("policies", bad_data)

    def test_validate_policy_shorthand_collides_with_canonical_name_raises(self):
        """A shorthand matching a different provider's canonical name should raise."""
        bad_data = {
            "providers": {
                "prov_a": {"aliases": [], "shorthands": ["prov_b"], "url_patterns": []},
                "prov_b": {"aliases": [], "shorthands": [], "url_patterns": []},
            }
        }
        with pytest.raises(ValueError, match="collides with a canonical provider name"):
            _validate("policies", bad_data)

    def test_validate_policy_self_shorthand_allowed(self):
        """A provider naming itself in its shorthands is idiomatic and must pass.

        Every real provider lists its canonical name as a shorthand so the bare
        name resolves; the canonical-collision guard must not reject that.
        """
        clean_data = {
            "providers": {
                "prov_a": {"aliases": [], "shorthands": ["prov_a"], "url_patterns": []},
            }
        }
        _validate("policies", clean_data)

    def test_validate_clean_data_passes(self):
        """Well-formed data should pass validation without error."""
        clean_robots = {
            "robots": {
                "r1": {"aliases": ["alias1"]},
                "r2": {"aliases": ["alias2"]},
            }
        }
        _validate("robots", clean_robots)

        clean_policies = {
            "providers": {
                "p1": {"aliases": ["a1"], "shorthands": ["s1"], "url_patterns": ["^ws://"]},
                "p2": {"aliases": ["a2"], "shorthands": ["s2"], "url_patterns": ["^zmq://"]},
            }
        }
        _validate("policies", clean_policies)


# Policy resolution tests


class TestResolvePolicy:
    """resolve_policy() should handle shorthands, HF model IDs, and server URLs."""

    def test_shorthand_aliases(self):
        """All shorthand aliases for mock should resolve to 'mock'."""
        for alias in ("mock", "random", "test"):
            provider, _ = resolve_policy(alias)
            assert provider == "mock", f"'{alias}' should resolve to 'mock'"

    def test_huggingface_model_id_nvidia(self):
        """NVIDIA model IDs should resolve to groot via hf_orgs."""
        provider, kwargs = resolve_policy("nvidia/gr00t-n1.5-3b")
        assert provider == "groot"
        assert kwargs["pretrained_name_or_path"] == "nvidia/gr00t-n1.5-3b"

    def test_huggingface_model_id_override(self):
        """model_id_overrides should match before hf_orgs."""
        provider, kwargs = resolve_policy("nvidia/groot-something-new")
        assert provider == "groot"
        assert kwargs["pretrained_name_or_path"] == "nvidia/groot-something-new"

    def test_unknown_hf_org_falls_back_to_lerobot_local(self):
        """Unknown HF org should fall back to lerobot_local."""
        provider, kwargs = resolve_policy("unknownorg/somemodel")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "unknownorg/somemodel"

    def test_zmq_url_extracts_host_and_port(self):
        """ZMQ URLs should resolve to groot with parsed host/port."""
        provider, kwargs = resolve_policy("zmq://myhost:9999")
        assert provider == "groot"
        assert kwargs["host"] == "myhost"
        assert kwargs["port"] == 9999

    def test_extra_kwargs_forwarded_on_shorthand(self):
        """Extra kwargs should pass through on shorthand resolution."""
        _, kwargs = resolve_policy("mock", custom_param="hello")
        assert kwargs["custom_param"] == "hello"

    def test_extra_kwargs_forwarded_on_hf_model(self):
        """Extra kwargs should pass through on HF model resolution."""
        _, kwargs = resolve_policy("nvidia/gr00t-n1.5-3b", batch_size=4)
        assert kwargs["batch_size"] == 4

    def test_extra_kwargs_forwarded_on_zmq_url(self):
        """Extra kwargs should pass through on URL resolution."""
        _, kwargs = resolve_policy("zmq://host:1234", data_config="abc")
        assert kwargs["data_config"] == "abc"

    def test_unrecognised_string_falls_back(self):
        """A random string should fall back to lerobot_local."""
        provider, kwargs = resolve_policy("totally_unknown_string_xyz")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "totally_unknown_string_xyz"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped."""
        provider, _ = resolve_policy("  mock  ")
        assert provider == "mock"

    def test_registered_provider_name_resolves(self):
        """A canonical provider name should resolve directly."""
        provider, _ = resolve_policy("groot")
        assert provider == "groot"

    def test_case_insensitive_shorthand(self):
        """Shorthands should match case-insensitively."""
        provider, _ = resolve_policy("Mock")
        assert provider == "mock"

        provider, _ = resolve_policy("GROOT")
        assert provider == "groot"


# Provider lookup tests


class TestProviderLookup:
    """JSON-based provider config should be queryable."""

    def test_known_provider_returns_config(self):
        config = get_policy_provider("groot")
        assert config is not None
        assert "port" in config["config_keys"]
        assert config["class"] == "Gr00tPolicy"

    def test_unknown_provider_returns_none(self):
        assert get_policy_provider("nonexistent_xyz") is None

    def test_list_providers_includes_all_json_entries(self):
        providers = list_policy_providers()
        assert "mock" in providers
        assert "groot" in providers

    def test_provider_has_required_keys(self):
        """Every provider entry should have module, class, and config_keys."""
        for name in list_policy_providers():
            config = get_policy_provider(name)
            assert "module" in config, f"{name} missing 'module'"
            assert "class" in config, f"{name} missing 'class'"
            assert "config_keys" in config, f"{name} missing 'config_keys'"

    def test_get_provider_by_alias(self):
        """get_policy_provider should resolve aliases to the canonical config."""
        # "random" is an alias/shorthand for "mock"
        config = get_policy_provider("random")
        assert config is not None
        assert config["class"] == "MockPolicy"


# import_policy_class tests


class TestImportPolicyClass:
    """import_policy_class() should dynamically load the right class."""

    def test_import_mock(self):
        """Importing 'mock' should return MockPolicy."""
        from strands_robots.policies import MockPolicy

        cls = import_policy_class("mock")
        assert cls is MockPolicy

    def test_import_unknown_raises(self):
        """Unknown provider should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown policy provider"):
            import_policy_class("nonexistent_provider_xyz_999")

    def test_import_via_alias(self):
        """Importing via alias should return the same class."""
        from strands_robots.policies import MockPolicy

        cls = import_policy_class("random")
        assert cls is MockPolicy


# build_policy_kwargs tests


class TestBuildPolicyKwargs:
    """build_policy_kwargs() should map generic params to provider-specific keys."""

    def test_groot_port_and_host(self):
        """groot provider should accept port and host."""
        kwargs = build_policy_kwargs("groot", policy_port=5555, policy_host="gpu-box")
        assert kwargs["port"] == 5555
        assert kwargs["host"] == "gpu-box"

    def test_groot_defaults_host(self):
        """groot should default host to 'localhost' when not provided."""
        kwargs = build_policy_kwargs("groot", policy_port=5555)
        assert kwargs["host"] == "localhost"

    def test_groot_data_config(self):
        """groot should accept data_config when provided."""
        kwargs = build_policy_kwargs("groot", data_config={"key": "val"})
        assert kwargs["data_config"] == {"key": "val"}

    def test_unknown_provider_returns_empty(self):
        """Unknown provider should return empty kwargs."""
        kwargs = build_policy_kwargs("nonexistent_xyz")
        assert kwargs == {}

    def test_extra_kwargs_for_allowed_keys(self):
        """Extra kwargs matching config_keys should be included."""
        kwargs = build_policy_kwargs("groot", data_config={"some": "config"})
        assert kwargs["data_config"] == {"some": "config"}

    def test_extra_kwargs_not_in_allowed_keys_ignored(self):
        """Extra kwargs NOT in config_keys should be ignored."""
        kwargs = build_policy_kwargs("groot", not_a_real_key="ignored")
        assert "not_a_real_key" not in kwargs

    def test_groot_only_port_no_host_gets_default(self):
        """When only port is given, host should default from JSON defaults."""
        kwargs = build_policy_kwargs("groot", policy_port=9999)
        assert kwargs["port"] == 9999
        assert kwargs["host"] == "localhost"  # from defaults


# Robot registry tests


class TestRobotRegistry:
    """robots.py - resolve, query, filter, and format robot definitions."""

    def test_resolve_name_canonical(self):
        assert resolve_name("so100") == "so100"
        assert resolve_name("panda") == "panda"

    def test_resolve_name_alias(self):
        assert resolve_name("franka") == "panda"
        assert resolve_name("g1") == "unitree_g1"
        assert resolve_name("go2") == "unitree_go2"
        assert resolve_name("so100_follower") == "so100"

    def test_resolve_name_case_insensitive(self):
        assert resolve_name("FRANKA") == "panda"
        assert resolve_name("Go2") == "unitree_go2"

    def test_resolve_name_normalizes_hyphens(self):
        assert resolve_name("reachy-mini") == "reachy_mini"

    def test_resolve_unknown_returns_input(self):
        assert resolve_name("nonexistent_bot") == "nonexistent_bot"

    def test_get_robot_returns_full_definition(self):
        robot = get_robot("so100")
        assert robot is not None
        assert robot["category"] == "arm"
        assert robot["joints"] == 6  # so100 is a 6-DOF arm (5 joints + gripper)
        assert "asset" in robot
        assert "hardware" in robot

    def test_get_robot_via_alias(self):
        robot = get_robot("franka")
        assert robot is not None
        assert "Franka" in robot["description"] or "Panda" in robot["description"]

    def test_get_robot_unknown_returns_none(self):
        assert get_robot("nonexistent_xyz") is None

    def test_has_sim(self):
        assert has_sim("so100") is True
        assert has_sim("panda") is True

    def test_has_sim_false_for_real_only(self):
        assert has_sim("reachy2") is False

    def test_has_hardware(self):
        assert has_hardware("so100") is True
        assert has_hardware("lekiwi") is True

    def test_has_hardware_false_for_sim_only(self):
        assert has_hardware("ur5e") is False

    def test_get_hardware_type(self):
        assert get_hardware_type("so100") == "so100_follower"
        assert get_hardware_type("lekiwi") == "lekiwi"

    def test_get_hardware_type_none_for_sim_only(self):
        assert get_hardware_type("ur5e") is None

    def test_get_hardware_type_none_for_unknown(self):
        assert get_hardware_type("nonexistent_xyz") is None

    def test_list_robots_all(self):
        robots = list_robots("all")
        names = [r["name"] for r in robots]
        assert "so100" in names
        assert "panda" in names
        assert "lekiwi" in names
        assert len(robots) > 20

    def test_list_robots_sim_only(self):
        robots = list_robots("sim")
        for r in robots:
            assert r["has_sim"] is True
        assert "reachy2" not in [r["name"] for r in robots]

    def test_list_robots_real_only(self):
        robots = list_robots("real")
        for r in robots:
            assert r["has_real"] is True
        assert "ur5e" not in [r["name"] for r in robots]

    def test_list_robots_both(self):
        robots = list_robots("both")
        for r in robots:
            assert r["has_sim"] is True
            assert r["has_real"] is True
        names = [r["name"] for r in robots]
        assert "so100" in names
        assert "lekiwi" in names  # sim + real
        assert "reachy2" not in names
        assert "ur5e" not in names

    def test_list_robots_unknown_mode_raises(self):
        # A plausible-but-unsupported filter (hardware-capable robots) must not
        # silently return the full, unfiltered list - that would mislead a
        # caller into thinking every robot is hardware-backed. It raises instead.
        with pytest.raises(ValueError, match="Unknown list_robots mode 'hardware'"):
            list_robots("hardware")

    def test_list_robots_typo_mode_raises_and_lists_valid_modes(self):
        # A typo is rejected loudly, and the message enumerates the valid modes
        # so the caller can self-correct without reading the source.
        with pytest.raises(ValueError) as excinfo:
            list_robots("sims")
        msg = str(excinfo.value)
        for valid in ("all", "sim", "real", "both"):
            assert valid in msg

    def test_list_robots_all_documented_modes_accepted(self):
        # Every value advertised in LIST_ROBOTS_MODES must be accepted without
        # raising, guarding the constant and the validation set against drift.
        for mode in LIST_ROBOTS_MODES:
            assert isinstance(list_robots(mode), list)

    def test_list_robots_by_category(self):
        by_cat = list_robots_by_category()
        assert "arm" in by_cat
        assert "humanoid" in by_cat
        arm_names = [r["name"] for r in by_cat["arm"]]
        assert "so100" in arm_names

    def test_list_aliases_returns_mapping(self):
        aliases = list_aliases()
        assert isinstance(aliases, dict)
        assert len(aliases) > 10
        assert aliases["franka"] == "panda"
        assert aliases["g1"] == "unitree_g1"

    def test_format_robot_table_readable(self):
        table = format_robot_table()
        assert "so100" in table
        assert "panda" in table
        assert "Total:" in table
        assert len(table.strip().split("\n")) > 10

    def test_has_sim_unknown_returns_false(self):
        assert has_sim("nonexistent_xyz") is False

    def test_has_hardware_unknown_returns_false(self):
        assert has_hardware("nonexistent_xyz") is False
