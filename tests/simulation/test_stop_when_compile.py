"""Unit tests for ``compile_stop_when`` - the ``run_policy(stop_when=...)`` compiler.

Pure DSL-compilation behaviour against fake sims; no physics backend needed.
The end-to-end ``run_policy`` integration (early return, ``stopped_reason``
telemetry, recording interplay, tool dispatch) lives in
``tests/simulation/test_run_policy_stop_when.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.simulation.benchmark_spec import compile_stop_when


class _BodyStateSim:
    """Minimal sim exposing ``get_body_state`` for body_* predicates."""

    def __init__(self, positions: dict[str, list[float]]):
        self._pos = positions

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._pos:
            return {"status": "error", "content": [{"text": f"Body '{body_name}' not found."}]}
        return {
            "status": "success",
            "content": [
                {"text": f"body {body_name}"},
                {"json": {"position": self._pos[body_name], "quaternion": [1, 0, 0, 0], "mass": 1.0}},
            ],
        }

    def get_observation(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {}


class TestCompileSingleCall:
    def test_single_predicate_call_compiles_and_evaluates(self):
        fn = compile_stop_when({"predicate": "body_above_z", "body": "cube", "z": 0.2})
        assert fn(_BodyStateSim({"cube": [0, 0, 0.5]})) is True
        assert fn(_BodyStateSim({"cube": [0, 0, 0.1]})) is False

    def test_all_group_requires_every_predicate(self):
        fn = compile_stop_when(
            {
                "all": [
                    {"predicate": "body_above_z", "body": "cube", "z": 0.2},
                    {"predicate": "body_below_z", "body": "cube", "z": 0.8},
                ]
            }
        )
        assert fn(_BodyStateSim({"cube": [0, 0, 0.5]})) is True
        assert fn(_BodyStateSim({"cube": [0, 0, 0.9]})) is False

    def test_any_group_requires_one_predicate(self):
        fn = compile_stop_when(
            {
                "any": [
                    {"predicate": "body_above_z", "body": "cube", "z": 0.8},
                    {"predicate": "body_below_z", "body": "cube", "z": 0.2},
                ]
            }
        )
        assert fn(_BodyStateSim({"cube": [0, 0, 0.1]})) is True
        assert fn(_BodyStateSim({"cube": [0, 0, 0.5]})) is False


class TestCompileRejections:
    """Every rejected shape gets an actionable ValueError - a clause that
    silently never fires would run the rollout to its full budget while the
    caller believes the gate is armed."""

    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="stop_when"):
            compile_stop_when("grasped")  # type: ignore[arg-type]

    def test_empty_dict_rejected(self):
        with pytest.raises(ValueError, match="never fire"):
            compile_stop_when({})

    def test_empty_all_any_lists_rejected(self):
        with pytest.raises(ValueError, match="never fire"):
            compile_stop_when({"all": []})
        with pytest.raises(ValueError, match="never fire"):
            compile_stop_when({"any": []})

    def test_mixed_single_call_and_group_rejected(self):
        with pytest.raises(ValueError, match="not both"):
            compile_stop_when(
                {
                    "predicate": "body_above_z",
                    "body": "cube",
                    "z": 0.2,
                    "all": [{"predicate": "body_upright", "body": "cube"}],
                }
            )

    def test_unknown_top_level_keys_rejected(self):
        with pytest.raises(ValueError, match="unknown keys"):
            compile_stop_when({"when": [{"predicate": "body_above_z", "body": "cube", "z": 0.2}]})

    def test_unknown_predicate_name_lists_valid_set(self):
        with pytest.raises(ValueError, match="Unknown predicate 'levitated'.*grasped"):
            compile_stop_when({"predicate": "levitated", "body": "cube"})

    def test_float_reward_term_rejected(self):
        # A float-valued term reads as bool(<nonzero float>) == always-True and
        # would stop the rollout on step 1 - the same guard the benchmark
        # success clause applies.
        with pytest.raises(ValueError, match="reward term"):
            compile_stop_when({"predicate": "distance_neg", "body_a": "a", "body_b": "b"})

    def test_bad_predicate_kwargs_rejected(self):
        with pytest.raises(ValueError, match="body_above_z"):
            compile_stop_when({"predicate": "body_above_z", "altitude": 0.2})


class TestReferencedEntities:
    """``stop_when_referenced_entities`` - the walker behind the pre-rollout
    scene probe (a typo'd body must not compile into a clause that silently
    never fires)."""

    def test_single_call_body(self):
        from strands_robots.simulation.benchmark_spec import stop_when_referenced_entities

        bodies, joints = stop_when_referenced_entities({"predicate": "body_above_z", "body": "cube", "z": 0.2})
        assert bodies == ["cube"]
        assert joints == []

    def test_group_collects_all_name_kwargs_deduplicated(self):
        from strands_robots.simulation.benchmark_spec import stop_when_referenced_entities

        clause = {
            "all": [
                {"predicate": "distance_less_than", "body_a": "cube", "body_b": "tray", "threshold": 0.1},
                {"predicate": "body_inside", "body": "cube", "container": "bin"},
                {"predicate": "joint_above", "joint": "so100/Jaw", "value": 0.5},
            ]
        }
        bodies, joints = stop_when_referenced_entities(clause)
        assert bodies == ["cube", "tray", "bin"]
        assert joints == ["so100/Jaw"]

    def test_non_name_kwargs_ignored(self):
        from strands_robots.simulation.benchmark_spec import stop_when_referenced_entities

        # gripper_prefix / z / geoms are not probeable entity names.
        bodies, joints = stop_when_referenced_entities(
            {
                "any": [
                    {"predicate": "grasped", "body": "cube", "gripper_prefix": "so100"},
                    {"predicate": "contact_between", "geom_a": "g1", "geom_b": "g2"},
                ]
            }
        )
        assert bodies == ["cube"]
        assert joints == []
