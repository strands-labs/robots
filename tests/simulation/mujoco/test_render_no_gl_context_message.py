"""Every renderer consumer surfaces a clean, actionable no-OpenGL-context error.

When the offscreen renderer cannot be built (no EGL/OSMesa GL context on the
host), ``_get_renderer`` returns ``None`` and each of its four consumers has to
answer for itself. They deliberately do *not* share one channel:

* :meth:`render` / :meth:`render_depth` are agent tools, so they short-circuit
  with a structured ``status=error`` dict. That text is read verbatim by an LLM
  caller, so it must be a clean sentence: no stray leading or trailing
  whitespace, naming both the failure and the actionable fix.
* :meth:`get_frame` returns raw ``(rgb, depth)`` ndarrays to in-process
  consumers, so it **raises** ``RuntimeError`` with the same actionable text.
  Mirroring its siblings would be silently unsafe: a two-key envelope unpacks
  without complaint at a ``rgb, depth = sim.get_frame(...)`` call site, handing
  the consumer the strings ``"status"`` and ``"content"`` instead of failing.
* ``_get_sim_observation`` skips the camera and keeps the rest of the
  observation (pinned next to the other camera-failure paths, in
  ``test_observation_camera_failure_resilience``).

Both documented in-process consumers of :meth:`get_frame` depend on the raise
for their own message to name GL:
:class:`~strands_robots.rendering.HybridCompositor` propagates it, and
``get_world_point`` catches it and carries the text into its tool envelope.

The renderer-None branch is exercised deterministically by monkeypatching
``_get_renderer`` to return ``None``, so these tests run on every runner
regardless of the local GL driver -- no test here needs a GL context.
"""

from __future__ import annotations

import ast
import inspect

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import rendering  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="no_gl_context_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _error_text(result: dict) -> str:
    assert result["status"] == "error", result
    return next(block["text"] for block in result["content"] if "text" in block)


def test_render_reports_clean_message_when_no_gl_context(sim, monkeypatch):
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: None)

    text = _error_text(sim.render(camera_name="default", width=4, height=3))

    # No orphaned whitespace (a leading space here is a real cosmetic defect
    # left over from stripping a prefix glyph); actionable and self-describing.
    assert text == text.strip()
    assert "Rendering unavailable" in text
    assert "OpenGL" in text
    assert ("EGL" in text) or ("OSMesa" in text)


def test_render_depth_reports_clean_message_when_no_gl_context(sim, monkeypatch):
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: None)

    text = _error_text(sim.render_depth(camera_name="default", width=4, height=3))

    assert text == text.strip()
    assert "Depth rendering unavailable" in text
    assert "OpenGL" in text
    assert ("EGL" in text) or ("OSMesa" in text)


# ---------------------------------------------------------------------------
# get_frame: the raw-frame surface, whose channel is a raise
# ---------------------------------------------------------------------------


def test_get_frame_raises_a_clean_message_when_no_gl_context(sim, monkeypatch):
    """The third renderer consumer answers with the same actionable sentence.

    ``get_frame`` documents ``RuntimeError`` for "no GL context available"; this
    pins that the text it raises is as actionable as the two agent-tool
    surfaces' envelopes, so a caller on a GL-free host learns the same fix from
    whichever surface they reached.
    """
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: None)

    with pytest.raises(RuntimeError) as excinfo:
        sim.get_frame(camera_name="default", width=4, height=3)

    text = str(excinfo.value)
    assert text == text.strip()
    assert "Rendering unavailable" in text
    assert "OpenGL" in text
    assert ("EGL" in text) or ("OSMesa" in text)


def test_get_frame_raises_rather_than_returning_an_envelope(sim, monkeypatch):
    """The raw-frame surface must not be "harmonised" onto the envelope channel.

    ``get_frame``'s consumers unpack its return value as a two-tuple, and a
    two-key envelope dict unpacks without complaint -- so returning one here
    would hand a consumer the strings ``"status"`` and ``"content"`` and fail
    much later, far from the missing GL context. The raise is what keeps the
    failure at its cause.
    """
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: None)

    with pytest.raises(RuntimeError):
        sim.get_frame(camera_name="default", width=4, height=3)

    # The premise that makes the two channels genuinely different: this is what
    # a consumer's ``rgb, depth = ...`` would silently bind if the envelope were
    # returned instead of raised.
    envelope = {"status": "error", "content": [{"text": "Rendering unavailable ..."}]}
    first, second = envelope
    assert (first, second) == ("status", "content")


# ---------------------------------------------------------------------------
# The two documented in-process consumers of get_frame
# ---------------------------------------------------------------------------


def test_the_compositor_surfaces_the_actionable_message(sim, monkeypatch):
    """``HybridCompositor`` -- named in ``get_frame``'s own docstring -- propagates it."""
    hybrid = pytest.importorskip("strands_robots.rendering")
    sim.add_camera(name="look", position=[0.6, -0.5, 0.4], target=[0.0, 0.0, 0.1])
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: None)

    with pytest.raises(RuntimeError) as excinfo:
        hybrid.HybridCompositor(sim).render("look")

    text = str(excinfo.value)
    assert "OpenGL" in text
    assert ("EGL" in text) or ("OSMesa" in text)


def test_get_world_point_carries_the_actionable_message_into_its_envelope(sim, monkeypatch):
    """The pixel-grounding tool converts the raise into its own error envelope.

    ``get_world_point`` is an agent tool, so it must not raise -- but the text a
    caller reads only names GL because ``get_frame`` raised something
    actionable rather than returning an empty frame.
    """
    sim.add_camera(name="look", position=[0.6, -0.5, 0.4], target=[0.0, 0.0, 0.1])
    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: None)

    result = sim.get_world_point(camera_name="look", pixels=[[2, 1]], width=4, height=3)

    text = _error_text(result)
    assert "OpenGL" in text
    assert ("EGL" in text) or ("OSMesa" in text)


# ---------------------------------------------------------------------------
# Drift guard: a fifth consumer must decide and pin its own channel
# ---------------------------------------------------------------------------

#: Every consumer of ``_get_renderer``, and where its ``renderer is None``
#: channel is pinned.  Keyed on the method name so a consumer added later fails
#: this test until someone decides what it does without a GL context.
_NO_GL_CHANNEL = {
    "render": "error envelope (this module)",
    "render_depth": "error envelope (this module)",
    "get_frame": "raises RuntimeError (this module)",
    "_get_sim_observation": "skips the camera (test_observation_camera_failure_resilience)",
}


def _renderer_consumers() -> set[str]:
    """Method names that ask ``_get_renderer`` for an offscreen renderer."""
    source = inspect.getsource(rendering)
    tree = ast.parse(source)
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name == "_get_renderer":
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "self._get_renderer(" in segment:
            found.add(node.name)
    return found


def test_every_renderer_consumer_has_a_pinned_no_gl_channel():
    """The no-GL matrix is complete.

    ``_get_renderer`` returning ``None`` is not an error in itself -- each
    consumer chooses how to answer, and the choices genuinely differ (envelope,
    raise, skip).  A new consumer that never decides would silently inherit
    whatever falls out of its own control flow, so this pins the set.
    """
    consumers = _renderer_consumers()

    assert consumers, "no _get_renderer consumers found -- the scan is looking in the wrong module"
    assert consumers == set(_NO_GL_CHANNEL), (
        "the set of _get_renderer consumers changed: "
        f"unpinned={sorted(consumers - set(_NO_GL_CHANNEL))} "
        f"stale={sorted(set(_NO_GL_CHANNEL) - consumers)}. "
        "Decide what the new consumer does without a GL context, pin it, and record it here."
    )


# ---------------------------------------------------------------------------
# The advice has to be true on the host that is reading it
# ---------------------------------------------------------------------------


class TestTheFixIsPlatformCorrect:
    """``apt-get install libosmesa6-dev`` is right on Linux and false on macOS.

    macOS ships neither EGL nor OSMesa (MuJoCo renders through CGL there), so
    the old single-sentence advice sent a Mac operator to install a package that
    does not exist for their machine - and while they chased it, the real cause
    kept its cover. This pins both halves: Linux keeps the packages, darwin gets
    a cause it can actually act on.

    ``platform=`` is injected rather than read from the host, so both halves are
    graded on every runner - nothing here skips on a Linux CI box.
    """

    def test_linux_keeps_the_package_advice(self):
        text = rendering.no_gl_context_message(platform="linux")
        assert "apt-get install libosmesa6-dev" in text
        assert "macOS" not in text

    def test_macos_is_not_told_to_apt_get(self):
        text = rendering.no_gl_context_message(platform="darwin")
        assert "apt-get" not in text, "there is no apt on macOS"
        assert "CGL" in text, "name the API that actually renders there"
        assert "launchd" in text, "the daemon-with-no-window-server case is the common one"

    def test_macos_names_the_lost_context_case_no_install_can_fix(self):
        """A context that worked and then vanished is not a missing install.

        The two macOS causes need opposite actions: no window-server session is
        fixed by where the process is started from, a context that worked
        EARLIER in this same process is fixed by starting a fresh one. Naming
        only the first sends the second case chasing a session it already had.
        """
        text = rendering.no_gl_context_message(platform="darwin")
        assert "EARLIER" in text and "fresh process" in text

    def test_both_platforms_stay_within_the_shared_contract(self):
        for plat in ("linux", "darwin", "win32"):
            for depth in (False, True):
                text = rendering.no_gl_context_message(depth=depth, platform=plat)
                assert text == text.strip()
                assert "OpenGL" in text
                # The existing consumers' assertions: EGL/OSMesa is named either
                # as the fix (Linux) or as the thing that does not exist (macOS).
                assert ("EGL" in text) or ("OSMesa" in text)
                head = "Depth rendering unavailable" if depth else "Rendering unavailable"
                assert text.startswith(head)

    def test_the_running_platform_is_the_default(self):
        """Every consumer calls the helper with no ``platform=``.

        The injected argument exists for the tests; the shipped call sites pass
        nothing, so the default has to be the host actually reading the message.
        """
        import sys

        assert rendering.no_gl_context_message() == rendering.no_gl_context_message(platform=sys.platform)

    def test_every_consumer_goes_through_the_helper(self):
        """No consumer may keep its own hardcoded sentence and drift."""
        src = inspect.getsource(rendering)
        assert src.count("apt-get install libosmesa6-dev") == 1, (
            "the Linux advice lives in no_gl_context_message and nowhere else"
        )
        assert src.count("no_gl_context_message(") >= 4  # def + 3 call sites
