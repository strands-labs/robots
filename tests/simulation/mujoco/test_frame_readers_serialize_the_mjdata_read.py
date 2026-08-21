"""Every public frame-producing reader serialises its mjData read.

``Renderer.update_scene`` copies ``xpos``/``xquat``/``xmat`` and the geom poses
out of mjData, and ``Renderer.render`` dereferences ``data.contact``. Both race
a concurrent ``mj_step`` - from a ``PolicyRunner`` worker, from the ``step()``
loop, or from the camera-recorder daemon, none of which is covered by the
blanket dispatch lock - and the result is a torn frame or a native crash.

:meth:`~strands_robots.simulation.mujoco.rendering.RenderingMixin.render` and
:meth:`~strands_robots.simulation.mujoco.rendering.RenderingMixin.get_frame`
already serialise that read under ``self._lock``; ``render`` carries the comment
explaining why.
:meth:`~strands_robots.simulation.mujoco.rendering.RenderingMixin.render_depth`
performed the identical pair of operations on the same buffers without the lock,
while its own docstring says its content "mirrors :meth:`render`" and that the
depth map is pixel-aligned with it - so of the three readers a caller can reach,
two were serialised and one was not.

The hazard is not hypothetical on this surface: with a policy worker stepping on
its own thread, 3 of 60 depth reads landed inside a physics step while 0 of 60
RGB reads did.

Two things are pinned here:

* **Behaviour** - a reader must not be able to complete while another thread
  holds ``self._lock``. The writer here holds the real lock and never sleeps, so
  the serialised verdict is an ``Event`` that times out rather than a wall-clock
  measurement: a reader blocked on the lock *cannot* signal completion while the
  writer is inside its critical section, whatever the machine's load.
* **The root cause** - every public method that reads mjData into a renderer has
  that read inside a ``with self._lock`` block, derived from the module's own AST
  rather than from a hand-written list, so a fourth reader is graded when it is
  added. The one private helper that reads mjData the same way
  (``_get_sim_observation``) is graded through its caller instead: the assertion
  is that ``get_observation`` holds the lock across the delegation, which pins
  the arrangement rather than exempting it.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from strands_robots.simulation.mujoco import rendering as rendering_mod  # noqa: E402
from strands_robots.simulation.mujoco import simulation as simulation_mod  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Long enough that an unserialised reader (~1 ms against the stand-in renderer)
# finishes inside it by a wide margin, and irrelevant to the serialised verdict:
# a reader waiting on the lock can never signal completion while the writer
# holds it, so that direction cannot time out early under load.
_READER_BUDGET_S = 1.0

_PUBLIC_FRAME_READERS = ("render", "render_depth", "get_frame")


class _StandInRenderer:
    """Offscreen-renderer stand-in with the ``mujoco.Renderer`` read contract.

    Mirrors the two calls that touch mjData - ``update_scene`` then ``render``
    - so the serialisation question can be asked without a GL context. The
    depth toggle mirrors ``enable_depth_rendering``/``disable_depth_rendering``
    because ``render_depth`` drives the renderer through them.
    """

    def __init__(self) -> None:
        self.rgb = np.full((4, 6, 3), 128, dtype=np.uint8)
        self.depth = np.array([[0.75, 2.40], [1.50, 1.50]], dtype=np.float32)
        self._depth_mode = False
        self.update_scene_calls = 0

    def update_scene(self, data: Any, camera: Any = None, scene_option: Any = None) -> None:
        self.update_scene_calls += 1

    def enable_depth_rendering(self) -> None:
        self._depth_mode = True

    def disable_depth_rendering(self) -> None:
        self._depth_mode = False

    def render(self) -> Any:
        return self.depth if self._depth_mode else self.rgb


@pytest.fixture
def sim():
    s = Simulation(tool_name="frame_reader_lock", mesh=False)
    s.create_world()
    yield s
    s.cleanup(policy_stop_timeout=0.5)


def _call_reader(sim: Simulation, name: str) -> Any:
    """Invoke one public frame reader at a fixed small resolution."""
    return getattr(sim, name)(camera_name="default", width=2, height=2)


def _completes_while_a_writer_holds_the_lock(sim: Simulation, name: str) -> tuple[bool, dict[str, Any]]:
    """Does *name* finish while this thread holds ``sim._lock``?

    The writer (this thread) holds the real lock for as long as it takes the
    reader thread to either finish or fail to; it never sleeps, so a ``False``
    verdict means the reader was genuinely blocked rather than merely slow.
    """
    done = threading.Event()
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["result"] = _call_reader(sim, name)
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
            box["error"] = exc
            raise AssertionError(f"{name} raised while a writer held the lock: {exc!r}") from exc
        finally:
            done.set()

    with sim._lock:
        reader = threading.Thread(target=run, daemon=True, name=f"reader-{name}")
        reader.start()
        completed_under_the_lock = done.wait(_READER_BUDGET_S)

    reader.join(_READER_BUDGET_S * 5)
    return completed_under_the_lock, box


class TestEveryPublicFrameReaderSerialisesItsMjDataRead:
    """A reader must not observe mjData while another thread may be mutating it."""

    @pytest.mark.parametrize("reader", _PUBLIC_FRAME_READERS)
    def test_a_reader_cannot_complete_while_a_writer_holds_the_lock(self, sim, monkeypatch, reader):
        stand_in = _StandInRenderer()
        monkeypatch.setattr(sim, "_get_renderer", lambda w, h: stand_in)
        # Warm the one-time depth-warning path so its stderr capture is not the
        # thing being timed on the render_depth case.
        _call_reader(sim, "render_depth")
        assert getattr(sim, "_depth_warn_text", None) is not None, "premise: the warn path ran"
        reads_before = stand_in.update_scene_calls

        completed, box = _completes_while_a_writer_holds_the_lock(sim, reader)

        assert "error" not in box, box.get("error")
        assert not completed, (
            f"{reader} finished while another thread held sim._lock, so its "
            f"update_scene read mjData a concurrent mj_step was free to mutate "
            f"(update_scene calls during the window: {stand_in.update_scene_calls - reads_before})"
        )

    @pytest.mark.parametrize("reader", _PUBLIC_FRAME_READERS)
    def test_a_reader_still_returns_once_the_writer_releases(self, sim, monkeypatch, reader):
        """Serialising must not deadlock or drop the frame - the read still happens."""
        stand_in = _StandInRenderer()
        monkeypatch.setattr(sim, "_get_renderer", lambda w, h: stand_in)
        before = stand_in.update_scene_calls
        _call_reader(sim, reader)
        assert stand_in.update_scene_calls == before + 1


def _ids_under_the_lock(fn: ast.AST) -> set[int]:
    """``id()`` of every node lexically inside a ``with self._lock`` block."""
    under: set[int] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.With) and any("_lock" in ast.unparse(i.context_expr) for i in node.items):
            for inner in ast.walk(node):
                under.add(id(inner))
    return under


def _mjdata_reads_outside_the_lock() -> dict[str, list[int]]:
    """Map public reader name -> lines of its ``update_scene`` calls not under the lock.

    Derived from the module's AST, so a reader added later is graded without an
    edit here. Private helpers are excluded: they are reached through a public
    facade that holds the lock, which the sibling test below pins directly.
    """
    source = Path(str(inspect.getsourcefile(rendering_mod))).read_text(encoding="utf-8")
    tree = ast.parse(source)
    out: dict[str, list[int]] = {}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (m for m in cls.body if isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef)):
            if fn.name.startswith("_"):
                continue
            under = _ids_under_the_lock(fn)
            reads = [
                node
                for node in ast.walk(fn)
                if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == "update_scene"
            ]
            if reads:
                out[fn.name] = [node.lineno for node in reads if id(node) not in under]
    return out


class TestTheLockCoversTheMjDataReadStructurally:
    """The root cause: the read itself has to sit inside the critical section."""

    def test_the_scan_found_every_known_public_reader(self) -> None:
        """Non-vacuity: a scan that reaches nothing would report a clean module."""
        found = _mjdata_reads_outside_the_lock()
        assert set(_PUBLIC_FRAME_READERS) <= set(found), (
            f"the mjData-read scan reached {sorted(found)}; it must grade every public "
            f"reader in {list(_PUBLIC_FRAME_READERS)}"
        )

    def test_no_public_reader_reads_mjdata_outside_the_lock(self) -> None:
        offenders = {name: lines for name, lines in _mjdata_reads_outside_the_lock().items() if lines}
        assert not offenders, (
            "every public frame reader must copy mjData into the renderer under "
            "self._lock - render() carries the comment explaining why. Reads "
            f"outside the lock: {offenders}"
        )

    def test_the_private_observation_helper_is_covered_by_its_caller(self) -> None:
        """``_get_sim_observation`` reads mjData under ``get_observation``'s lock."""
        caller = textwrap.dedent(inspect.getsource(simulation_mod.MuJoCoSimEngine.get_observation))
        fn = next(n for n in ast.walk(ast.parse(caller)) if isinstance(n, ast.FunctionDef))
        under = _ids_under_the_lock(fn)
        delegations = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("_get_sim_observation")
        ]
        assert delegations, "premise: get_observation still delegates to _get_sim_observation"
        assert all(id(node) in under for node in delegations), (
            "get_observation must hold self._lock across the _get_sim_observation "
            "delegation - that is what keeps the helper's mjData read serialised"
        )


class TestTheDepthAnswerIsUnchanged:
    """Serialising the read must not move what render_depth reports."""

    def test_the_metric_bounds_and_png_still_come_back(self, sim, monkeypatch) -> None:
        monkeypatch.setattr(sim, "_get_renderer", lambda w, h: _StandInRenderer())
        result = sim.render_depth(camera_name="default", width=2, height=2)
        assert result["status"] == "success", result
        payload = next(block["json"] for block in result["content"] if "json" in block)
        assert payload["depth_min"] == pytest.approx(0.75, abs=1e-4)
        assert payload["depth_max"] == pytest.approx(2.40, abs=1e-4)
        image = next(block["image"] for block in result["content"] if "image" in block)
        assert image["format"] == "png"
        assert len(image["source"]["bytes"]) > 0

    def test_an_unknown_camera_is_still_refused_before_any_read(self, sim, monkeypatch) -> None:
        stand_in = _StandInRenderer()
        monkeypatch.setattr(sim, "_get_renderer", lambda w, h: stand_in)
        result = sim.render_depth(camera_name="no_such_camera", width=2, height=2)
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]
        assert stand_in.update_scene_calls == 0

    def test_the_one_time_warning_text_is_still_cached(self, sim, monkeypatch) -> None:
        monkeypatch.setattr(sim, "_get_renderer", lambda w, h: _StandInRenderer())
        assert getattr(sim, "_depth_warn_text", None) is None
        # Hoisted out of the assert so the failure carries the whole envelope,
        # matching the sibling depth tests that drive a stand-in renderer.
        first_result = sim.render_depth(camera_name="default", width=2, height=2)
        assert first_result["status"] == "success", first_result
        first = sim._depth_warn_text
        assert first is not None
        second_result = sim.render_depth(camera_name="default", width=2, height=2)
        assert second_result["status"] == "success", second_result
        assert sim._depth_warn_text == first
