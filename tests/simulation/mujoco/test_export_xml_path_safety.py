"""``export_xml(output_path=...)`` treats its destination as untrusted input.

``export_xml`` is one of the 77 actions the MuJoCo simulation exposes as an
agent tool, so ``output_path`` arrives from an LLM tool call exactly as
``render``'s does. :mod:`strands_robots.simulation.safe_output` exists for that
class of sink and its module docstring enumerates them; this module pins that
``export_xml`` is one of the enumerated sinks in behaviour and not only in prose.

Two halves, because a guard that refuses everything is as wrong as one that
refuses nothing:

* the arbitrary-write vectors (``..`` traversal, a symlinked target, shell
  metacharacters, backslash separators) are refused through the tool envelope
  and leave nothing on disk, and
* an ordinary absolute destination still exports. Confinement here is
  guards-only: unlike ``render``, whose ``output_path`` ``safe_output`` documents
  as "a newer, sandboxed-by-design feature", this sink has always accepted an
  absolute path and a sandbox would be a breaking change.

The structural test at the end is the drift net: it asserts no function in the
agent-callable simulation tree writes a caller-supplied path parameter directly
rather than the validated derivative, which is exactly the shape this fixed.
"""

import ast
import inspect
import pathlib

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Path parameters that reach a filesystem sink from an agent tool call.
_PATH_PARAMS = frozenset({"output_path", "output_dir", "path", "root", "spec_path", "scene_path"})
# Calls that write. The validated derivative is what these must receive.
_WRITE_CALLS = frozenset({"open", "write_text", "write_bytes"})
# Helpers that mean "this function routed its path through the shared guard".
_GUARDS = ("validate_output_path", "_validate_render_output_path", "atomic_write_bytes")


@pytest.fixture
def sim():
    """A live world whose ``_backend_state["spec"]`` export_xml serialises."""
    s = Simulation(tool_name="export_path_safety", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _text(result):
    """First text block of an agent-tool envelope."""
    for block in result.get("content", []):
        if "text" in block:
            return block["text"]
    return ""


class TestArbitraryWriteVectorsAreRefused:
    """Each vector is reported through the envelope and writes nothing."""

    def test_a_traversal_segment_is_refused(self, sim, tmp_path):
        escaped = tmp_path.parent / "escaped-by-traversal.xml"
        assert not escaped.exists(), "fixture: the escape target must not pre-exist"

        result = sim.export_xml(output_path=str(tmp_path / ".." / escaped.name))

        assert result["status"] == "error"
        assert "path traversal" in _text(result)
        assert not escaped.exists(), "a refused export still escaped the requested directory"

    def test_a_symlinked_target_is_not_followed(self, sim, tmp_path):
        victim = tmp_path / "victim.xml"
        victim.write_text("ORIGINAL")
        link = tmp_path / "link.xml"
        link.symlink_to(victim)

        result = sim.export_xml(output_path=str(link))

        assert result["status"] == "error"
        assert "symlink" in _text(result)
        assert victim.read_text() == "ORIGINAL", "the export followed a symlink and overwrote the target"

    @pytest.mark.parametrize("bad", [";", "|", "$", "`", ">", "<"])
    def test_shell_metacharacters_are_refused(self, sim, tmp_path, bad):
        target = tmp_path / f"a{bad}b.xml"

        result = sim.export_xml(output_path=str(target))

        assert result["status"] == "error"
        assert "shell metacharacters" in _text(result)
        assert not target.exists()

    def test_backslash_separators_are_refused(self, sim, tmp_path):
        result = sim.export_xml(output_path=str(tmp_path) + "\\nested\\scene.xml")

        assert result["status"] == "error"
        assert "backslash" in _text(result)

    def test_a_refusal_names_the_method(self, sim, tmp_path):
        """The envelope says which action refused, not just what was wrong."""
        result = sim.export_xml(output_path=str(tmp_path / ".." / "x.xml"))

        assert result["status"] == "error"
        assert _text(result).startswith("export_xml: ")


class TestTheHistoricContractStillHolds:
    """Over-reach controls: an ordinary destination must still export."""

    def test_an_absolute_destination_still_exports(self, sim, tmp_path):
        target = tmp_path / "exported.xml"

        result = sim.export_xml(output_path=str(target))

        assert result["status"] == "success"
        assert target.exists()
        assert "<mujoco" in target.read_text()

    def test_the_success_text_reports_the_resolved_path(self, sim, tmp_path):
        """The raw argument can name a different file than the one written."""
        nested = tmp_path / "sub"
        nested.mkdir()
        # No ".." segment, but a symlinked *directory* still resolves elsewhere.
        alias = tmp_path / "alias"
        alias.symlink_to(nested, target_is_directory=True)
        target = alias / "scene.xml"

        result = sim.export_xml(output_path=str(target))

        assert result["status"] == "success"
        written = nested / "scene.xml"
        assert written.exists(), "the export did not land in the resolved directory"
        assert str(written) in _text(result), (
            f"the text names the raw argument rather than the file written: {_text(result)!r}"
        )

    def test_inline_export_is_unaffected(self, sim):
        """Omitting output_path keeps returning the XML inline."""
        result = sim.export_xml()

        assert result["status"] == "success"
        assert "mujoco" in _text(result).lower()


class TestTheWriteIsAtomic:
    def test_an_existing_file_is_replaced_without_temp_residue(self, sim, tmp_path):
        target = tmp_path / "scene.xml"
        target.write_text("STALE")

        result = sim.export_xml(output_path=str(target))

        assert result["status"] == "success"
        assert "<mujoco" in target.read_text()
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == [], f"atomic write left temp residue: {leftovers}"


class TestTheWriteNeverRaisesPastTheEnvelope:
    """``export_xml`` returns a result dict; a bad destination must not escape it.

    The write used to sit outside the method's ``try``, so any ``OSError`` from
    the caller's path - a missing parent, a directory, an unwritable parent -
    propagated to the caller as a raw exception rather than as a result.
    """

    def test_a_missing_parent_directory_is_created(self, sim, tmp_path):
        target = tmp_path / "does" / "not" / "exist" / "scene.xml"

        result = sim.export_xml(output_path=str(target))

        assert result["status"] == "success"
        assert target.exists()
        assert oct(target.parent.stat().st_mode & 0o777) == "0o700", "a created parent must be owner-only"

    def test_a_destination_the_filesystem_rejects_is_reported_not_raised(self, sim, tmp_path):
        # A directory can never be written as a file.
        result = sim.export_xml(output_path=str(tmp_path))

        assert result["status"] == "error"
        text = _text(result)
        assert str(tmp_path) in text, f"the report does not name the destination: {text!r}"
        assert ".tmp" not in text, f"the report leaks the internal temp filename: {text!r}"


class TestNoAgentCallableSinkWritesAnUnvalidatedPath:
    """Drift net for the shape this module fixed.

    A well-behaved sink validates first and writes the *validated* path, so its
    write call never names the raw parameter. A write whose target expression
    names a caller-supplied path parameter directly is therefore the defect,
    and this scan needs no exemption list.
    """

    @staticmethod
    def _offenders():
        import strands_robots.simulation as sim_pkg

        root = pathlib.Path(inspect.getfile(sim_pkg)).parent
        found = []
        for py in sorted(root.rglob("*.py")):
            text = py.read_text()
            for fn in ast.walk(ast.parse(text)):
                if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                params = {a.arg for a in fn.args.args + fn.args.kwonlyargs} & _PATH_PARAMS
                if not params:
                    continue
                fn_src = ast.get_source_segment(text, fn) or ""
                if any(g in fn_src for g in _GUARDS):
                    continue
                for node in ast.walk(fn):
                    if not isinstance(node, ast.Call):
                        continue
                    name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                    if name not in _WRITE_CALLS:
                        continue
                    seg = ast.get_source_segment(text, node) or ""
                    if any(p in seg for p in params):
                        found.append(f"{py.name}::{fn.name} line {node.lineno} -> {name}()")
        return found, root

    def test_every_path_sink_routes_through_the_shared_guard(self):
        offenders, _ = self._offenders()
        assert offenders == [], (
            "these functions write a caller-supplied path without routing it through "
            f"strands_robots.simulation.safe_output: {offenders}"
        )

    def test_the_scan_reaches_the_simulation_tree(self):
        """Non-vacuity: an empty result must mean clean sources, not an empty scan."""
        _, root = self._offenders()
        modules = list(root.rglob("*.py"))
        assert len(modules) > 20, f"scan root {root} looks wrong: {len(modules)} modules"
        assert (root / "mujoco" / "physics.py").exists(), f"scan root {root} misses physics.py"

    def test_the_scan_detects_a_planted_offender(self, tmp_path, monkeypatch):
        """A planted raw write must be reported, or the scan proves nothing."""
        planted = tmp_path / "planted.py"
        planted.write_text('def save(output_path):\n    with open(output_path, "w") as f:\n        f.write("x")\n')

        text = planted.read_text()
        hits = []
        for fn in ast.walk(ast.parse(text)):
            if not isinstance(fn, ast.FunctionDef):
                continue
            params = {a.arg for a in fn.args.args} & _PATH_PARAMS
            fn_src = ast.get_source_segment(text, fn) or ""
            if not params or any(g in fn_src for g in _GUARDS):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) in _WRITE_CALLS:
                    seg = ast.get_source_segment(text, node) or ""
                    if any(p in seg for p in params):
                        hits.append(fn.name)
        assert hits == ["save"], f"the scan predicate missed a planted raw write: {hits}"
