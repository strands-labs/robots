"""Tests for ``strands_robots.simulation.benchmark_spec`` (declarative YAML/JSON loader).

Covers:

* :meth:`DeclarativeBenchmark.from_dict` schema validation (good / bad specs).
* :func:`register_benchmark_from_file` end-to-end with JSON + YAML.
* The sandboxed contract: unknown predicates / unknown top-level keys /
  non-dict predicate entries produce clear errors, not ``eval`` side-effects.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.benchmark import (
    _BENCHMARK_REGISTRY,
    get_benchmark,
)
from strands_robots.simulation.benchmark_spec import (
    DeclarativeBenchmark,
    register_benchmark_from_file,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


class _BodyStateSim:
    def __init__(self, positions: dict[str, list[float]]):
        self._pos = positions

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._pos:
            return {"status": "error", "content": [{"text": "missing"}]}
        return {
            "status": "success",
            "content": [
                {"text": body_name},
                {"json": {"position": self._pos[body_name]}},
            ],
        }

    def get_observation(self, *_, **__) -> dict[str, Any]:
        return {}


# Schema validation


class TestFromDictValidation:
    def test_minimal_valid_spec(self):
        spec = {
            "name": "minimal",
            "default_robot": "so100",
            "supported_robots": ["so100"],
        }
        bench = DeclarativeBenchmark.from_dict(spec)
        assert bench.name == "minimal"
        assert bench.default_robot == "so100"
        assert bench.supported_robots == ["so100"]
        assert bench.max_steps == 300  # default

    def test_rejects_non_dict_spec(self):
        with pytest.raises(ValueError):
            DeclarativeBenchmark.from_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_unknown_top_level_keys(self):
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(
                {"name": "x", "default_robot": "y", "supported_robots": ["y"], "weird_key": 1}
            )
        assert "weird_key" in str(exc.value)

    def test_rejects_missing_name(self):
        with pytest.raises(ValueError):
            DeclarativeBenchmark.from_dict({"default_robot": "y", "supported_robots": []})

    def test_rejects_missing_default_robot(self):
        with pytest.raises(ValueError):
            DeclarativeBenchmark.from_dict({"name": "x"})

    def test_rejects_default_not_in_supported(self):
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict({"name": "x", "default_robot": "ghost", "supported_robots": ["a", "b"]})
        assert "not in supported_robots" in str(exc.value)

    def test_allows_default_outside_supported_when_empty(self):
        """Empty supported_robots means "any" - default outside makes sense."""
        bench = DeclarativeBenchmark.from_dict({"name": "x", "default_robot": "anything", "supported_robots": []})
        assert bench.default_robot == "anything"

    def test_accepts_instruction(self):
        """A spec may declare a natural-language ``instruction`` and it is
        surfaced on the compiled benchmark. Before this was supported the key
        was rejected as an unknown top-level key, so a spec-authored benchmark
        could not carry a task command for a language-conditioned policy."""
        bench = DeclarativeBenchmark.from_dict(
            {
                "name": "x",
                "default_robot": "so100",
                "supported_robots": ["so100"],
                "instruction": "walk forward at 1 m/s",
            }
        )
        assert bench.instruction == "walk forward at 1 m/s"

    def test_instruction_defaults_to_empty(self):
        """A spec that omits ``instruction`` compiles to the empty-string
        default (backward compatible with every pre-existing spec)."""
        bench = DeclarativeBenchmark.from_dict({"name": "x", "default_robot": "so100", "supported_robots": ["so100"]})
        assert bench.instruction == ""

    def test_rejects_non_string_instruction(self):
        """A non-string ``instruction`` is a clear error, not a silent coerce."""
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(
                {
                    "name": "x",
                    "default_robot": "so100",
                    "supported_robots": ["so100"],
                    "instruction": 123,
                }
            )
        assert "instruction" in str(exc.value)

    def test_rejects_non_positive_max_steps(self):
        for bad in (-1, 0, "300", True):
            with pytest.raises(ValueError):
                DeclarativeBenchmark.from_dict(
                    {
                        "name": "x",
                        "default_robot": "y",
                        "supported_robots": ["y"],
                        "max_steps": bad,
                    }
                )


# Predicate compilation


class TestPredicateCompilation:
    def _base_spec(self, **overrides: Any) -> dict[str, Any]:
        spec = {
            "name": "t",
            "default_robot": "so100",
            "supported_robots": ["so100"],
            "max_steps": 10,
        }
        spec.update(overrides)
        return spec

    def test_success_all_true(self):
        spec = self._base_spec(
            success={
                "all": [
                    {"predicate": "body_above_z", "body": "cube", "z": 0.1},
                ]
            }
        )
        bench = DeclarativeBenchmark.from_dict(spec)
        sim_hit = _BodyStateSim({"cube": [0, 0, 0.2]})
        sim_miss = _BodyStateSim({"cube": [0, 0, 0.05]})
        assert bench.is_success(sim_hit) is True
        assert bench.is_success(sim_miss) is False

    def test_success_all_any_combined(self):
        """When both 'all' and 'any' are provided, both must hold."""
        spec = self._base_spec(
            success={
                "all": [{"predicate": "body_above_z", "body": "cube", "z": 0.0}],
                "any": [
                    {"predicate": "body_above_z", "body": "cube", "z": 10.0},
                    {"predicate": "body_above_z", "body": "cube", "z": 0.05},
                ],
            }
        )
        bench = DeclarativeBenchmark.from_dict(spec)
        sim = _BodyStateSim({"cube": [0, 0, 0.1]})
        # all: z>0.0 true. any: z>10 false OR z>0.05 true → any true. Combined: true.
        assert bench.is_success(sim) is True

    def test_failure_any(self):
        spec = self._base_spec(failure={"any": [{"predicate": "body_below_z", "body": "cube", "z": 0.0}]})
        bench = DeclarativeBenchmark.from_dict(spec)
        assert bench.is_failure(_BodyStateSim({"cube": [0, 0, -0.01]})) is True
        assert bench.is_failure(_BodyStateSim({"cube": [0, 0, 0.5]})) is False

    def test_dense_reward_sums_terms(self):
        spec = self._base_spec(
            dense_reward=[
                {"predicate": "constant", "value": 1.0},
                {"predicate": "constant", "value": -0.5},
            ]
        )
        bench = DeclarativeBenchmark.from_dict(spec)
        info = bench.on_step(None, {}, {})  # type: ignore[arg-type]
        assert info.reward == pytest.approx(0.5)

    def test_rejects_unknown_predicate(self):
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(self._base_spec(success={"all": [{"predicate": "totally_made_up"}]}))
        assert "Unknown predicate" in str(exc.value)

    def test_rejects_non_dict_predicate_entry(self):
        with pytest.raises(ValueError):
            DeclarativeBenchmark.from_dict(self._base_spec(success={"all": ["just a string"]}))

    def test_rejects_missing_predicate_key(self):
        with pytest.raises(ValueError):
            DeclarativeBenchmark.from_dict(self._base_spec(success={"all": [{"body": "cube", "z": 0.1}]}))

    def test_rejects_bad_clause_keys(self):
        """success/failure only allow 'all' / 'any'."""
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(self._base_spec(success={"all": [], "other": []}))
        assert "other" in str(exc.value)

    def test_rejects_unknown_predicate_in_dense_reward(self):
        """A typo'd predicate name in a ``dense_reward`` term surfaces the same
        clear ``Unknown predicate ... Valid: [...]`` error as a success clause.

        Success / failure clauses reject unknown names early via
        ``predicate_kind`` kind-checking, but ``dense_reward`` terms compile
        through a different path with no kind requirement and rely on
        ``make_predicate`` to reject the name. This pins that a typo in a
        reward-term predicate still fails loudly at compile time with a
        discoverable list of valid predicates, rather than a cryptic
        downstream crash at evaluation.
        """
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(
                self._base_spec(dense_reward=[{"predicate": "totally_made_up", "value": 1.0}])
            )
        msg = str(exc.value)
        assert "Unknown predicate" in msg
        assert "totally_made_up" in msg

    def test_predicate_bad_kwargs_surface_compile_error(self):
        """Bad predicate kwargs (wrong types, missing required) surface as a
        compile-time error, not a runtime predicate crash."""
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(
                self._base_spec(success={"all": [{"predicate": "inside_region", "body": "x"}]})
            )
        # Should mention the predicate name in the error for discoverability.
        assert "inside_region" in str(exc.value)


# Empty / default clauses


class TestEmptyClauses:
    def test_success_absent_defaults_to_false(self):
        bench = DeclarativeBenchmark.from_dict({"name": "x", "default_robot": "so100", "supported_robots": ["so100"]})
        assert bench.is_success(_BodyStateSim({})) is False

    def test_failure_absent_defaults_to_false(self):
        bench = DeclarativeBenchmark.from_dict({"name": "x", "default_robot": "so100", "supported_robots": ["so100"]})
        assert bench.is_failure(_BodyStateSim({})) is False

    def test_empty_success_returns_false(self):
        """Non-None but empty success clause must not default to "always true"."""
        bench = DeclarativeBenchmark.from_dict(
            {
                "name": "x",
                "default_robot": "so100",
                "supported_robots": ["so100"],
                "success": {"all": [], "any": []},
            }
        )
        assert bench.is_success(_BodyStateSim({})) is False


# File loading


class TestRegisterBenchmarkFromFile:
    def test_register_from_json(self, tmp_path):
        spec_path = tmp_path / "drawer.json"
        spec_path.write_text(
            json.dumps(
                {
                    "name": "drawer",
                    "default_robot": "so100",
                    "supported_robots": ["so100"],
                    "max_steps": 50,
                    "success": {
                        "all": [
                            {"predicate": "body_above_z", "body": "cube", "z": 0.1},
                        ]
                    },
                }
            )
        )
        bench = register_benchmark_from_file("drawer", str(spec_path))
        assert get_benchmark("drawer") is bench
        assert bench.max_steps == 50
        assert bench.is_success(_BodyStateSim({"cube": [0, 0, 0.5]})) is True

    def test_register_from_yaml(self, tmp_path):
        """YAML support is opt-in; skip if pyyaml isn't available in this env."""
        pytest.importorskip("yaml")
        spec_path = tmp_path / "y.yaml"
        spec_path.write_text(
            """
name: yml-task
default_robot: so100
supported_robots: [so100]
max_steps: 99
success:
  all:
    - {predicate: body_above_z, body: cube, z: 0.5}
"""
        )
        bench = register_benchmark_from_file("yml-task", str(spec_path))
        assert bench.max_steps == 99

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            register_benchmark_from_file("missing", str(tmp_path / "nope.json"))

    def test_rejects_unsupported_extension(self, tmp_path):
        p = tmp_path / "spec.toml"
        p.write_text("")
        with pytest.raises(ValueError) as exc:
            register_benchmark_from_file("x", str(p))
        assert ".toml" in str(exc.value) or "extension" in str(exc.value)

    def test_spec_name_internal_overridden_by_registry_name(self, tmp_path):
        """Registry name wins over any ``name`` declared inside the spec file.

        The override applies to the instance's ``.name`` too, not just the
        registry key - the documented contract is that the registry name wins,
        and ``DeclarativeBenchmark.name`` is what error/log messages report.
        """
        p = tmp_path / "s.json"
        p.write_text(
            json.dumps(
                {
                    "name": "internal-name",
                    "default_robot": "so100",
                    "supported_robots": ["so100"],
                }
            )
        )
        bench = register_benchmark_from_file("external-name", str(p))
        assert get_benchmark("external-name") is bench
        # The spec's internal name doesn't end up in the registry.
        assert get_benchmark("internal-name") is None
        # The instance reports the registry name, not the stale spec-internal one.
        assert bench.name == "external-name"

    def test_same_spec_registered_under_multiple_names(self, tmp_path):
        """The documented use case: one spec file, many registry names.

        Each registration must yield an instance whose ``.name`` matches its own
        registry key so the two are distinguishable - a spec that declares its
        own ``name`` must not make every registration report that one name.
        """
        p = tmp_path / "s.json"
        p.write_text(
            json.dumps(
                {
                    "name": "declared-in-spec",
                    "default_robot": "so100",
                    "supported_robots": ["so100"],
                }
            )
        )
        first = register_benchmark_from_file("task-a", str(p))
        second = register_benchmark_from_file("task-b", str(p))
        assert first.name == "task-a"
        assert second.name == "task-b"
        assert get_benchmark("task-a").name == "task-a"
        assert get_benchmark("task-b").name == "task-b"

    def test_rejects_empty_name(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text('{"name": "x", "default_robot": "y", "supported_robots": []}')
        with pytest.raises(ValueError):
            register_benchmark_from_file("", str(p))

    def test_bad_json_propagates(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json}")
        with pytest.raises(json.JSONDecodeError):
            register_benchmark_from_file("x", str(p))


# DeclarativeBenchmark lifecycle


class TestDeclarativeBenchmarkLifecycle:
    def test_on_episode_start_delegates_to_base(self):
        """Default on_episode_start loads the default_robot when sim is empty."""
        spec = {
            "name": "x",
            "default_robot": "so100",
            "supported_robots": ["so100"],
        }
        bench = DeclarativeBenchmark.from_dict(spec)

        class FakeSim:
            def __init__(self):
                self.added: list[dict[str, Any]] = []

            def list_robots(self):
                return []

            def add_robot(self, *, name, data_config):
                self.added.append({"name": name, "data_config": data_config})

        sim = FakeSim()
        bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]
        assert len(sim.added) == 1
        assert sim.added[0]["data_config"] == "so100"

    def test_scene_load_error_raises(self, tmp_path: Path):
        """If the sim's load_scene returns an error dict, the benchmark must surface it."""
        spec = {
            "name": "x",
            "default_robot": "so100",
            "supported_robots": ["so100"],
            "scene": str(tmp_path / "missing.xml"),
        }
        bench = DeclarativeBenchmark.from_dict(spec)

        class FakeSim:
            def load_scene(self, path):
                return {"status": "error", "content": [{"text": f"no such file: {path}"}]}

            def list_robots(self):
                return ["preloaded"]

        sim = FakeSim()
        with pytest.raises(RuntimeError) as exc:
            bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]
        assert "load_scene" in str(exc.value)


# Per-episode reward-term reset (stateful phase machines must not leak state)


class TestDeclarativeBenchmarkRewardTermReset:
    """Pin ``on_episode_start``'s per-episode reward-term reset contract.

    A ``staged_reward`` compiles to a stateful phase machine whose current
    stage advances monotonically as the episode progresses. Across a
    multi-episode eval each episode must start from stage 0 -- otherwise a
    later episode inherits the phase reached by an earlier one and the dense
    reward signal is silently wrong. ``on_episode_start`` enforces this by
    calling ``reset()`` on every reward term that exposes one; stateless
    (plain-callable) terms have no ``reset`` and must be skipped without error.
    """

    class _BodyZSim:
        """Minimal sim exposing the surface both predicate lookups and the
        base ``on_episode_start`` compatibility check need.

        ``get_body_state`` feeds ``body_above_z`` (the stage gate); a
        non-empty ``list_robots`` keeps the base impl from trying to add a
        default robot (there is no ``add_robot`` here on purpose).
        """

        def __init__(self, cube_z: float) -> None:
            self._cube_z = cube_z

        def get_body_state(self, body_name: str) -> dict[str, Any]:
            return {
                "status": "success",
                "content": [
                    {"text": body_name},
                    {"json": {"position": [0.0, 0.0, self._cube_z]}},
                ],
            }

        def list_robots(self) -> list[str]:
            return ["robot"]

    @staticmethod
    def _staged_benchmark() -> DeclarativeBenchmark:
        from strands_robots.simulation.predicates import make_predicate

        staged = make_predicate(
            "staged_reward",
            stages=[
                {
                    "reward": {"predicate": "constant", "value": 1.0},
                    "advance_when": {"predicate": "body_above_z", "body": "cube", "z": 0.5},
                },
                {"reward": {"predicate": "constant", "value": 2.0}},
            ],
        )
        return DeclarativeBenchmark(
            name="staged",
            supported_robots=[],
            default_robot="so100",
            max_steps=10,
            success_fn=lambda _sim: False,
            failure_fn=lambda _sim: False,
            reward_terms=[staged],
            scene=None,
        )

    def test_stateful_reward_term_phase_reset_between_episodes(self):
        """A staged reward driven into a later phase is reset to stage 0 when
        the next episode starts, so phase state never leaks across episodes."""
        bench = self._staged_benchmark()
        (term,) = bench._reward_terms
        sim = self._BodyZSim(cube_z=1.0)  # cube above the stage gate

        # Drive the phase machine forward: the gate fires so it advances.
        term(sim)
        assert term.phase == 1

        # Next episode: on_episode_start must reset the term back to stage 0.
        bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]
        assert term.phase == 0

        # And the reset is per-episode: advance again, reset again.
        term(sim)
        assert term.phase == 1
        bench.on_episode_start(sim, random.Random(1))  # type: ignore[arg-type]
        assert term.phase == 0

    def test_stateless_reward_terms_are_skipped_without_error(self):
        """Plain-callable reward terms expose no ``reset`` and must be left
        untouched by the reset loop -- on_episode_start still completes."""

        def stateless_term(_sim: Any) -> float:
            return 0.0

        bench = DeclarativeBenchmark(
            name="stateless",
            supported_robots=[],
            default_robot="so100",
            max_steps=10,
            success_fn=lambda _sim: False,
            failure_fn=lambda _sim: False,
            reward_terms=[stateless_term],
            scene=None,
        )
        sim = self._BodyZSim(cube_z=0.0)

        # No ``reset`` attribute on the term -> the loop must skip it silently.
        bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]
        assert bench._reward_terms == [stateless_term]
        assert stateless_term(sim) == 0.0


# Malformed-shape rejection (defensive validation paths)


class TestSpecShapeRejection:
    """Cover the error branches that guard against malformed spec dicts.

    These reject bad input at compile time with a clear ValueError rather than
    letting a wrong-typed field reach the runtime evaluation loop.
    """

    def _base_spec(self, **overrides: Any) -> dict[str, Any]:
        spec = {"name": "t", "default_robot": "so100", "supported_robots": ["so100"]}
        spec.update(overrides)
        return spec

    def test_rejects_non_dict_success_clause(self):
        """A success clause must be an {'all'/'any'} dict, not a bare list."""
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(self._base_spec(success=["body_above_z"]))
        assert "expected a dict" in str(exc.value)

    def test_rejects_non_list_dense_reward(self):
        """dense_reward must be a list of predicate entries, not a dict."""
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(self._base_spec(dense_reward={"predicate": "constant", "value": 1.0}))
        assert "dense_reward" in str(exc.value)
        assert "expected a list" in str(exc.value)

    def test_rejects_supported_robots_with_non_string_entry(self):
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(self._base_spec(supported_robots=["so100", 123]))
        assert "supported_robots" in str(exc.value)

    def test_rejects_supported_robots_not_a_list(self):
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(self._base_spec(supported_robots="so100"))
        assert "supported_robots" in str(exc.value)

    def test_rejects_non_string_scene(self):
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(self._base_spec(scene=42))
        assert "scene" in str(exc.value)


# Defensive runtime behaviour (one bad term / missing capability must not crash)


class TestDeclarativeBenchmarkResilience:
    def test_reward_term_failure_is_swallowed(self, caplog):
        """A reward term that raises is logged and skipped; sibling terms still sum."""
        import logging

        def boom(_sim):
            raise RuntimeError("term exploded")

        def good(_sim):
            return 2.0

        bench = DeclarativeBenchmark(
            name="resilient",
            supported_robots=["so100"],
            default_robot="so100",
            max_steps=10,
            success_fn=lambda _s: False,
            failure_fn=lambda _s: False,
            reward_terms=[boom, good],
        )
        with caplog.at_level(logging.WARNING):
            info = bench.on_step(_BodyStateSim({}), {}, {})  # type: ignore[arg-type]
        # The good term still contributes; the broken one contributes nothing.
        assert info.reward == pytest.approx(2.0)
        assert info.done is False
        assert any("reward term failed" in r.message for r in caplog.records)

    def test_scene_declared_but_sim_lacks_load_scene_warns(self, caplog):
        """A scene-declared benchmark on a sim without load_scene() warns and
        falls through to the base episode setup rather than raising."""
        import logging

        spec = {
            "name": "scene-task",
            "default_robot": "so100",
            "supported_robots": ["so100"],
            "scene": "some/scene.xml",
        }
        bench = DeclarativeBenchmark.from_dict(spec)

        class NoSceneSim:
            def __init__(self):
                self.added: list[dict[str, Any]] = []

            def list_robots(self):
                return []

            def add_robot(self, *, name, data_config):
                self.added.append({"name": name, "data_config": data_config})

        sim = NoSceneSim()
        with caplog.at_level(logging.WARNING):
            bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]
        assert any("no load_scene" in r.message for r in caplog.records)
        # Base setup still ran: the default robot was added.
        assert sim.added and sim.added[0]["data_config"] == "so100"


# File-loader edge cases


class TestSpecFileLoaderEdges:
    def test_directory_path_rejected(self, tmp_path):
        """A path that exists but is a directory (not a file) is rejected."""
        d = tmp_path / "a.json"
        d.mkdir()
        with pytest.raises(ValueError) as exc:
            register_benchmark_from_file("x", str(d))
        assert "not a file" in str(exc.value)

    def test_non_dict_json_payload_rejected(self, tmp_path):
        """A JSON file that parses to a list (not a dict) is rejected."""
        p = tmp_path / "list.json"
        p.write_text(json.dumps(["not", "a", "dict"]))
        with pytest.raises(ValueError) as exc:
            register_benchmark_from_file("x", str(p))
        assert "must parse to a dict" in str(exc.value)
