"""Tests for :meth:`LiberoAdapter.ensure_scene` - the public scene-resolution API.

Driver scripts (``examples/libero/run.py`` /
``run_mujoco_agent.py``) need the LIBERO scene - and the cameras it
supplies - available *before* ``evaluate_benchmark`` runs, so
``start_cameras_recording`` can resolve the camera names. Pre-fix they
had to call the private ``_generate_scene_from_bddl()``; ``ensure_scene``
is the public equivalent. Contract pinned here:

* idempotent: an already-set ``scene_path`` is returned unchanged and
  the generator is never invoked;
* generates + stores + returns the path when ``scene_path`` is unset
  and ``auto_generate_scene=True``;
* returns ``None`` without side effects when ``auto_generate_scene=False``
  or the generator can't recover a BDDL source;
* propagates generation failures (unlike the lazy warn-and-fall-back
  path inside ``on_episode_start``).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from strands_robots.benchmarks.libero import LiberoAdapter

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the red cube and place it on the plate")
  (:objects cube_1 plate_1 table_1 - object)
  (:init (on cube_1 table_1))
  (:goal (on cube_1 plate_1)))
"""


def _adapter(**kwargs) -> LiberoAdapter:
    kwargs.setdefault("init_jitter", 0.0)
    kwargs.setdefault("install_cameras", False)
    return LiberoAdapter.from_text(PICK_CUBE_BDDL, **kwargs)


def test_ensure_scene_is_public_api() -> None:
    """The method drivers call must be public (no leading underscore)."""
    assert hasattr(LiberoAdapter, "ensure_scene")
    assert not LiberoAdapter.ensure_scene.__name__.startswith("_")


def test_existing_scene_path_returned_unchanged_without_generation(tmp_path) -> None:
    explicit = str(tmp_path / "explicit.xml")
    adapter = _adapter(scene_path=explicit)

    with patch.object(adapter, "_generate_scene_from_bddl") as mock_gen:
        assert adapter.ensure_scene() == explicit

    mock_gen.assert_not_called()
    assert adapter.scene_path == explicit


def test_generates_stores_and_returns_path_when_unset(tmp_path) -> None:
    generated = str(tmp_path / "generated.xml")
    adapter = _adapter()
    assert adapter.scene_path is None

    with patch.object(adapter, "_generate_scene_from_bddl", return_value=generated) as mock_gen:
        assert adapter.ensure_scene() == generated

    mock_gen.assert_called_once()
    assert adapter.scene_path == generated


def test_idempotent_second_call_skips_generator(tmp_path) -> None:
    generated = str(tmp_path / "generated.xml")
    adapter = _adapter()

    with patch.object(adapter, "_generate_scene_from_bddl", return_value=generated) as mock_gen:
        adapter.ensure_scene()
        adapter.ensure_scene()

    mock_gen.assert_called_once()


def test_auto_generate_scene_false_returns_none_without_side_effects() -> None:
    adapter = _adapter(auto_generate_scene=False)

    with patch.object(adapter, "_generate_scene_from_bddl") as mock_gen:
        assert adapter.ensure_scene() is None

    mock_gen.assert_not_called()
    assert adapter.scene_path is None


def test_unrecoverable_bddl_source_returns_none() -> None:
    adapter = _adapter()

    with patch.object(adapter, "_generate_scene_from_bddl", return_value=None):
        assert adapter.ensure_scene() is None

    assert adapter.scene_path is None


def test_generation_failure_propagates_to_caller() -> None:
    """Unlike ``on_episode_start``'s lazy path, an explicit ``ensure_scene``
    call must fail loudly when generation raises."""
    adapter = _adapter()

    with patch.object(adapter, "_generate_scene_from_bddl", side_effect=ImportError("'libero' is required")):
        with pytest.raises(ImportError, match="libero"):
            adapter.ensure_scene()

    assert adapter.scene_path is None
