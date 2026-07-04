"""Resilience + clean-shutdown contract for the mesh camera-publish loop.

``Mesh._camera_loop`` is the background thread that publishes camera frames on
the mesh at a fixed rate. Two guarantees matter and are pinned here:

  1. A transient error from a single ``_publish_cameras_once`` tick (a camera
     that momentarily fails to render or JPEG-encode) MUST NOT kill the loop -
     it is logged and the loop keeps publishing on the next tick.
  2. The loop shuts down promptly when ``_stop_event`` is signalled, and paces
     itself at ``period = 1 / hz`` via ``_stop_event.wait(period)`` (so stop is
     observed immediately rather than after a full sleep).

The loop only touches ``_running``, ``_publish_cameras_once``, ``_stop_event``
and ``peer_id``, so it is exercised on a bare instance built with
``Mesh.__new__`` (the same construction pattern used by the other mesh unit
tests) - no zenoh transport or live robot required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from strands_robots.mesh.core import Mesh


def _bare_mesh(stop_waits, publish):
    """A Mesh with just the attributes ``_camera_loop`` reads.

    Args:
        stop_waits: return values for successive ``_stop_event.wait(period)``
            calls; the loop breaks on the first truthy one.
        publish: the ``_publish_cameras_once`` callable (a mock).
    """
    mesh = Mesh.__new__(Mesh)
    mesh.peer_id = "test__arm"
    mesh._running = True
    mesh._publish_cameras_once = publish
    mesh._stop_event = MagicMock()
    mesh._stop_event.wait.side_effect = list(stop_waits)
    return mesh


def test_camera_loop_publishes_each_tick_and_stops_on_event():
    publish = MagicMock()
    # Two ticks proceed, the third wait signals stop.
    mesh = _bare_mesh([False, False, True], publish)

    mesh._camera_loop(10.0)

    assert publish.call_count == 3
    # Paces at period = 1 / hz so a stop is observed within one interval.
    assert mesh._stop_event.wait.call_args_list[0].args[0] == 0.1


def test_camera_loop_swallows_tick_error_and_keeps_going():
    # Every tick raises; the loop must log and continue rather than die on the
    # first failure. Stop is signalled after the second tick.
    publish = MagicMock(side_effect=RuntimeError("camera render blipped"))
    mesh = _bare_mesh([False, True], publish)

    # No exception escapes the loop.
    mesh._camera_loop(20.0)

    # It kept publishing after the first error (resilience), then stopped.
    assert publish.call_count == 2
    assert mesh._stop_event.wait.call_args_list[0].args[0] == 0.05
