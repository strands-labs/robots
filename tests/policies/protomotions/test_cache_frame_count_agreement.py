"""A reference-motion cache states its frame count several times, and they must agree.

``MotionPlayer`` accepts a hand-built cache dict and an ``.npz`` as first-class
input modes, and the documented format gives every state channel a leading frame
axis - ``dof_pos`` is ``[num_frames, num_dofs]``, ``body_rot`` is
``[num_frames, num_bodies, 4]``, and so on - alongside a scalar ``num_frames``.
So one clip states its length seven times over, and nothing made those
statements agree.

The count is load-bearing rather than informational.
:meth:`MotionPlayer.get_state_at_frame` clamps a query into
``[0, num_frames - 1]`` so that an out-of-range frame is safe, and
:meth:`MotionPlayer.get_future_references` - the call the ONNX tracker makes
every control tick - asks for frames *ahead* of the playhead. That clamp is
therefore the only thing standing between the future window and the end of the
arrays, and it is calibrated against the declared count. A declared count the
channels cannot serve turns the guard into an out-of-range read a whole window
early: on a 120-frame cache still declaring 120 after its channels were trimmed
to 100, playback serves 92 frames and then raises a bare ``IndexError`` naming
neither the cache nor the mismatch. A declared count *below* the row count is
worse, because nothing is raised at all - the tail of the clip is simply
unreachable, and :meth:`MotionPlayer.save_cache_npz` then writes the wrong count
beside the full-length channels, so the truncation becomes a durable property of
a file that gets shared between machines.

Both directions arrive the same way: editing a cache by hand. Trimming or
concatenating the channels and leaving ``num_frames`` behind is the ordinary
slip, and ``qpos_to_motion_data`` sets the two consistently, so a mismatch is
always caller-side and always worth naming.

What is pinned here:

* A consistent cache is untouched - the count, the playback and ``control_dt``
  are what they were, including the omitted-``num_frames`` fallback to the rows.
* Every frame the player claims to have can actually be served through the
  tracker's future window.
* A declared count that contradicts the channels is refused in either direction,
  naming both numbers and a remedy.
* Channels that disagree with *each other* are refused too - the count is one
  number, not ``dof_pos``'s number.
* ``save_cache_npz`` can no longer emit a file whose count contradicts its rows.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.protomotions import MotionPlayer

_NUM_DOFS = 29
_NUM_BODIES = 33
_CONTROL_DT = 0.02
_CHANNELS = ("dof_pos", "dof_vel", "body_rot", "body_pos", "body_vel", "body_ang_vel")

# The GTP tracker's own future-reference offsets, in control steps.
_FUTURE_STEPS = [1, 2, 4, 8]


def _cache(num_rows: int, *, declared: int | None = None, control_dt: float = _CONTROL_DT) -> dict[str, Any]:
    """A well-formed cache of ``num_rows`` frames, declaring ``declared``.

    ``dof_pos`` counts up so a served frame can be identified by its value,
    which is what makes an unreachable tail observable rather than merely
    suspected.
    """
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    cache: dict[str, Any] = {
        "dof_pos": np.arange(num_rows * _NUM_DOFS, dtype=np.float32).reshape(num_rows, _NUM_DOFS),
        "dof_vel": np.zeros((num_rows, _NUM_DOFS), dtype=np.float32),
        "body_rot": np.tile(identity, (num_rows, _NUM_BODIES, 1)),
        "body_pos": np.zeros((num_rows, _NUM_BODIES, 3), dtype=np.float32),
        "body_vel": np.zeros((num_rows, _NUM_BODIES, 3), dtype=np.float32),
        "body_ang_vel": np.zeros((num_rows, _NUM_BODIES, 3), dtype=np.float32),
        "control_dt": control_dt,
    }
    if declared is not None:
        cache["num_frames"] = declared
    return cache


def _frames_servable(player: MotionPlayer) -> int:
    """Count the frames the player can serve through the tracker's future window."""
    served = 0
    for frame in range(player.total_frames):
        try:
            player.get_future_references(frame, _FUTURE_STEPS)
        except IndexError:
            break
        served += 1
    return served


class TestAConsistentCacheIsUntouched:
    """The check must cost a well-formed cache nothing - this is the boundary."""

    @pytest.mark.parametrize("declared", [120, None])
    def test_the_frame_count_is_the_row_count(self, declared: int | None) -> None:
        player = MotionPlayer(_cache(120, declared=declared))
        assert player.total_frames == 120

    def test_every_claimed_frame_is_servable_through_the_future_window(self) -> None:
        player = MotionPlayer(_cache(120, declared=120))
        assert _frames_servable(player) == 120

    def test_the_cache_still_carries_its_own_control_dt(self) -> None:
        player = MotionPlayer(_cache(60, declared=60, control_dt=1.0 / 30.0))
        assert player.control_dt == pytest.approx(1.0 / 30.0)

    def test_the_channels_are_bound_verbatim(self) -> None:
        cache = _cache(12, declared=12)
        player = MotionPlayer(cache)
        bound = player.as_cache()
        for key in _CHANNELS:
            assert np.array_equal(bound[key], cache[key]), key


class TestADeclaredCountTheChannelsCannotServeIsRefused:
    """The clamp is calibrated against the declared count, so it has to be true."""

    def test_the_player_never_claims_a_frame_it_cannot_serve(self) -> None:
        """The consequence, stated without reference to how it is prevented.

        Either the cache is refused on load, or every frame the player counts
        has to survive the tracker's future window. Nothing else is acceptable,
        because the overrun lands mid-rollout as a bare ``IndexError``.
        """
        cache = _cache(120, declared=120)
        for key in _CHANNELS:
            cache[key] = cache[key][:100]

        try:
            player = MotionPlayer(cache)
        except ValueError:
            return  # refused before playback, so the claim is never made

        servable = _frames_servable(player)
        assert servable == player.total_frames, (
            f"the player claims {player.total_frames} frames but the future window "
            f"overruns the channels after {servable}"
        )

    def test_a_trimmed_cache_still_declaring_the_old_count_is_refused(self) -> None:
        cache = _cache(120, declared=120)
        for key in _CHANNELS:
            cache[key] = cache[key][:100]

        with pytest.raises(ValueError) as excinfo:
            MotionPlayer(cache)

        message = str(excinfo.value)
        assert "120" in message and "100" in message, message
        assert "num_frames" in message, message

    def test_the_refusal_offers_a_remedy_that_works(self) -> None:
        cache = _cache(120, declared=120)
        for key in _CHANNELS:
            cache[key] = cache[key][:100]

        with pytest.raises(ValueError) as excinfo:
            MotionPlayer(cache)

        # The message tells the caller to drop num_frames or resize; do the
        # first and the same cache has to load, or the remedy is a dead end.
        assert re.search(r"[Dd]rop num_frames", str(excinfo.value)), str(excinfo.value)
        del cache["num_frames"]
        player = MotionPlayer(cache)
        assert player.total_frames == 100
        assert _frames_servable(player) == 100

    def test_a_count_below_the_row_count_is_refused_rather_than_hiding_the_tail(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            MotionPlayer(_cache(120, declared=50))
        assert "50" in str(excinfo.value) and "120" in str(excinfo.value)

    def test_channels_that_disagree_with_each_other_are_refused(self) -> None:
        cache = _cache(120)
        cache["body_rot"] = cache["body_rot"][:60]

        with pytest.raises(ValueError) as excinfo:
            MotionPlayer(cache)

        message = str(excinfo.value)
        assert "body_rot" in message, message
        assert "60" in message and "120" in message, message

    def test_a_scalar_channel_carries_no_frame_axis_and_is_refused(self) -> None:
        cache = _cache(120)
        cache["dof_vel"] = np.float32(0.0)

        with pytest.raises(ValueError) as excinfo:
            MotionPlayer(cache)
        assert "dof_vel" in str(excinfo.value)


class TestASavedCacheCannotContradictItself:
    """A cache file travels, so a wrong count in one is durable."""

    def test_save_cache_npz_writes_a_count_its_channels_can_serve(self, tmp_path: Any) -> None:
        path = str(tmp_path / "clip.npz")
        MotionPlayer(_cache(120, declared=120)).save_cache_npz(path)

        written = dict(np.load(path))
        assert int(written["num_frames"]) == written["dof_pos"].shape[0] == 120

    def test_a_saved_cache_round_trips_through_the_loader(self, tmp_path: Any) -> None:
        path = str(tmp_path / "clip.npz")
        MotionPlayer(_cache(64, declared=64)).save_cache_npz(path)

        reloaded = MotionPlayer(path)
        assert reloaded.total_frames == 64
        assert _frames_servable(reloaded) == 64
