"""Direct-unit coverage of ``LerobotLocalPolicy._resolve_camera_targets``.

The camera-name -> policy-image-key router encodes a documented precedence:

  1. an explicit ``camera_key_map`` ctor param wins for any name it lists,
  2. an exact name match (``top`` -> ``observation.images.top`` OR a bare
     ``top`` that the policy declares directly),
  3. positional fallback into the remaining declared slots (loud WARN), and
  4. a hard ``ValueError`` when the robot under-supplies cameras.

The method is exercised indirectly through the torch-batch build path, but its
bare-declared-key branch (a policy that declares ``base`` / ``wrist`` instead of
``observation.images.<cam>``, e.g. MolmoAct2) had no direct test. These pin the
full contract so a routing regression is caught at the method boundary rather
than only through a higher-level rollout.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


class _VisualFeature:
    """Minimal stand-in for a declared VISUAL ``PolicyFeature``."""

    class _T:
        name = "VISUAL"

    type = _T()


def _make_policy(
    features: dict[str, object],
    *,
    camera_key_map: dict[str, str] | None = None,
    strict_keys: bool = False,
) -> LerobotLocalPolicy:
    """Build a policy with ``_load_model`` patched and declared input features."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        p = LerobotLocalPolicy(
            pretrained_name_or_path="test/model",
            camera_key_map=camera_key_map,
            strict_keys=strict_keys,
        )
    p._input_features = features
    return p


def _prefixed_two_cam() -> dict[str, object]:
    return {
        "observation.images.top": _VisualFeature(),
        "observation.images.wrist": _VisualFeature(),
        "observation.state": object(),
    }


def _bare_two_cam() -> dict[str, object]:
    return {
        "base": _VisualFeature(),
        "wrist": _VisualFeature(),
        "observation.state": object(),
    }


def test_bare_declared_key_binds_by_exact_name():
    """A camera named exactly like a BARE declared key binds by name, not positionally."""
    p = _make_policy(_bare_two_cam())
    result = p._resolve_camera_targets(["base", "wrist"])
    assert result == {"base": "base", "wrist": "wrist"}
    assert p.positional_fallback_used is False


def test_prefixed_declared_key_binds_by_short_name():
    """``top`` binds to the declared ``observation.images.top`` slot."""
    p = _make_policy(_prefixed_two_cam())
    result = p._resolve_camera_targets(["top", "wrist"])
    assert result == {
        "top": "observation.images.top",
        "wrist": "observation.images.wrist",
    }
    assert p.positional_fallback_used is False


def test_explicit_camera_key_map_wins():
    """An explicit mapping routes a mismatched camera name onto a declared key."""
    p = _make_policy(
        _prefixed_two_cam(),
        camera_key_map={"cam_left": "observation.images.top"},
    )
    result = p._resolve_camera_targets(["cam_left", "wrist"])
    assert result["cam_left"] == "observation.images.top"
    assert result["wrist"] == "observation.images.wrist"
    assert p.positional_fallback_used is False


def test_explicit_map_to_undeclared_key_raises():
    """A camera_key_map entry targeting an undeclared image key is rejected."""
    p = _make_policy(
        _prefixed_two_cam(),
        camera_key_map={"top": "observation.images.nonexistent"},
    )
    with pytest.raises(ValueError, match="does not declare"):
        p._resolve_camera_targets(["top"])


def test_positional_fallback_when_names_do_not_match(caplog):
    """Unmatched names fill remaining declared slots positionally with a WARN."""
    p = _make_policy(_prefixed_two_cam())
    with caplog.at_level(logging.WARNING):
        result = p._resolve_camera_targets(["cam_a", "cam_b"])
    assert set(result.values()) == {
        "observation.images.top",
        "observation.images.wrist",
    }
    assert p.positional_fallback_used is True
    assert "does not match any declared policy image key" in caplog.text


def test_strict_keys_raises_instead_of_positional_fallback():
    """strict_keys=True turns an unmatched-name fallback into an actionable error."""
    p = _make_policy(_prefixed_two_cam(), strict_keys=True)
    with pytest.raises(ValueError, match="strict_keys=True"):
        p._resolve_camera_targets(["cam_a", "cam_b"])
    assert p.positional_fallback_used is False


def test_extra_cameras_beyond_policy_slots_are_dropped():
    """Cameras past what the policy consumes are omitted, not force-fed."""
    p = _make_policy(_bare_two_cam())
    result = p._resolve_camera_targets(["base", "wrist", "extra"])
    assert result == {"base": "base", "wrist": "wrist"}
    assert "extra" not in result


def test_under_supplied_cameras_raises():
    """Fewer cameras than the policy's declared image slots is a hard error."""
    p = _make_policy(_prefixed_two_cam())
    with pytest.raises(ValueError, match="requires image input"):
        p._resolve_camera_targets(["top"])
