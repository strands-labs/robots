"""The mesh discovery posture is gossip-only, and prose that says otherwise is a defect.

:func:`strands_robots.mesh._zenoh_config.scouting_block` emits
``scouting/multicast/enabled = false`` and ``scouting/gossip/enabled = true``
unless an operator sets ``STRANDS_MESH_MULTICAST=true``. Multicast on a shared
LAN is an enrollment surface - any host that joins ``224.0.0.224:7446`` sees and
can attract fleet robots without credentials - so the default is deliberate and
:mod:`strands_robots.mesh.core` logs a loud warning when it is turned on.

Prose that presents multicast as the standing posture therefore does not merely
read oddly: it tells an operator the fleet is discoverable on the LAN when it is
not (so cross-host peers silently never find each other and the reader looks for
the fault in the wrong place), and it describes the opt-in attack surface as
already accepted. Two places said exactly that - the layer-5 label of
``examples/lerobot/architecture.svg`` read "Zenoh multicast (default)", and the
:mod:`strands_robots.mesh.iot.camera_offload` module docstring called the Zenoh
path "LAN multicast" - so this guard fails on both.

Why the rule is judged per *block* rather than per sentence
-----------------------------------------------------------
The correct prose about multicast mentions it repeatedly while explaining why it
is off, and the correct *warning* text says "Multicast scouting is ON" outright.
Both would be flagged by any sentence-level match. The unit that carries a claim
is the whole block - one docstring, one contiguous ``#`` comment run, one Markdown
paragraph, one SVG ``<text>`` label - and a block is compliant when it either
names the ``STRANDS_MESH_MULTICAST`` opt-in knob or says multicast is off. That
granularity is what lets this guard ship with **no** exclusion list:
:mod:`strands_robots.mesh.core`'s fleet-takeover warning names the flag, so the
block containing the words "Multicast scouting is ON" passes on its own merits.

Scope: the prose surfaces that describe *this* transport
--------------------------------------------------------
Every module under :mod:`strands_robots.mesh`, ``README.md``, and the repository
diagrams. Device Connect's D2D pages are deliberately outside it, and that is a
measured boundary rather than an exemption:
:class:`strands_robots.device_connect.RobotDeviceDriver` and its siblings
configure no Zenoh session at all - the package reads no scouting key, imports no
config builder, and takes its runtime from the external ``device_connect_edge``
distribution - so its multicast prose describes a session this repository never
configures. ``test_device_connect_configures_no_scouting_key`` pins that, and
fails loudly if the two subsystems ever share a config path, at which point the
D2D pages belong in scope too.

The behavioural half of this contract - that the warning fires on opt-in and
stays silent by default - is pinned separately by
``tests/mesh/test_multicast_startup_warning.py``.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import strands_robots.mesh as mesh_pkg

_MESH_PKG = Path(inspect.getfile(mesh_pkg)).parent
_REPO_ROOT = _MESH_PKG.parents[1]

#: Any mention of the transport-level discovery mode this rule is about.
_MULTICAST = re.compile(r"multicast", re.I)

#: A block is compliant when it names the opt-in knob ...
_NAMES_THE_FLAG = re.compile(r"STRANDS_MESH_MULTICAST")

#: ... or states that multicast is not the standing posture.
_SAYS_IT_IS_OFF = re.compile(
    r"multicast[^.]{0,40}(?:off|disabled|opt-in|opt in|not the default)"
    r"|(?:off|disabled|opt-in|opt in)[^.]{0,40}multicast",
    re.I,
)


def states_the_posture(text: str) -> bool:
    """Whether a block that mentions multicast also says what the default is."""
    return bool(_NAMES_THE_FLAG.search(text) or _SAYS_IT_IS_OFF.search(text))


def _python_blocks(source: str) -> list[tuple[str, str]]:
    """Docstrings and contiguous ``#`` comment runs, each as one block."""
    blocks: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                blocks.append((f"docstring:{getattr(node, 'name', '<module>')}", doc))
    run: list[str] = []
    start = 0
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            if not run:
                start = lineno
            run.append(stripped.lstrip("#").strip())
        elif run:
            blocks.append((f"comment@L{start}", "\n".join(run)))
            run = []
    if run:
        blocks.append((f"comment@L{start}", "\n".join(run)))
    return blocks


def _markdown_blocks(source: str) -> list[tuple[str, str]]:
    """Blank-line separated paragraphs, each as one block."""
    blocks: list[tuple[str, str]] = []
    offset = 0
    for para in source.split("\n\n"):
        lineno = source[:offset].count("\n") + 1
        offset += len(para) + 2
        if para.strip():
            blocks.append((f"paragraph@L{lineno}", para))
    return blocks


def _svg_blocks(source: str) -> list[tuple[str, str]]:
    """Each ``<text>`` label, which stands alone in a diagram, as one block."""
    return [
        (f"label@L{source[: m.start()].count(chr(10)) + 1}", m.group(1))
        for m in re.finditer(r"<text[^>]*>(.*?)</text>", source, re.S)
    ]


def blocks_of(path: Path) -> list[tuple[str, str]]:
    """Split a prose surface into the blocks this rule judges."""
    source = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return _python_blocks(source)
    if path.suffix == ".md":
        return _markdown_blocks(source)
    if path.suffix == ".svg":
        return _svg_blocks(source)
    raise AssertionError(f"no block reader for {path.suffix}")


def unmarked_blocks(path: Path) -> list[tuple[str, str]]:
    """Blocks that mention multicast without saying it is off / opt-in."""
    return [
        (label, " ".join(text.split()))
        for label, text in blocks_of(path)
        if _MULTICAST.search(text) and not states_the_posture(text)
    ]


def _prose_surfaces() -> list[Path]:
    """The prose that describes the ``strands_robots.mesh`` transport."""
    candidates = [
        *_MESH_PKG.rglob("*.py"),
        _REPO_ROOT / "README.md",
        *(_REPO_ROOT / "examples").rglob("*.svg"),
        *(_REPO_ROOT / "docs").rglob("*.svg"),
    ]
    return sorted(p for p in candidates if p.is_file())


_SURFACES = _prose_surfaces()
_ARCHITECTURE_SVG = _REPO_ROOT / "examples" / "lerobot" / "architecture.svg"


@pytest.mark.parametrize("path", _SURFACES, ids=lambda p: str(p.relative_to(_REPO_ROOT)))
def test_no_prose_presents_multicast_as_the_standing_posture(path: Path) -> None:
    """Every block naming multicast must say it is off, or name the opt-in flag."""
    offenders = unmarked_blocks(path)
    assert not offenders, (
        f"{path.relative_to(_REPO_ROOT)} presents multicast as the discovery default. "
        "scouting_block() ships multicast OFF and gossip ON; say so, or name "
        f"STRANDS_MESH_MULTICAST as the opt-in: {offenders}"
    )


class TestTheCorpusIsRealRatherThanEmpty:
    """A rule over prose is worthless if the scan found no prose to judge."""

    def test_every_block_reader_has_live_prose_to_judge(self) -> None:
        """.py, .md and .svg each contribute a block that names multicast."""
        naming: dict[str, list[str]] = {".py": [], ".md": [], ".svg": []}
        for path in _SURFACES:
            for label, text in blocks_of(path):
                if _MULTICAST.search(text):
                    naming[path.suffix].append(f"{path.name}[{label}]")
        for suffix, found in naming.items():
            assert found, (
                f"no {suffix} block in scope mentions multicast, so the {suffix} "
                "reader is not exercised and this guard would pass vacuously"
            )

    def test_the_scope_holds_the_surfaces_that_set_and_describe_the_posture(self) -> None:
        """The config builder, the warning, the README and the diagram are all in scope."""
        relative = {str(p.relative_to(_REPO_ROOT)) for p in _SURFACES}
        for expected in (
            "strands_robots/mesh/_zenoh_config.py",
            "strands_robots/mesh/core.py",
            "strands_robots/mesh/session.py",
            "strands_robots/mesh/iot/camera_offload.py",
            "README.md",
            "examples/lerobot/architecture.svg",
        ):
            assert expected in relative, f"{expected} dropped out of the scanned scope"


class TestTheDiagramNamesTheRealPosture:
    """The diagram label is the one in-scope SVG that still mentions multicast."""

    def test_the_transport_label_states_multicast_is_opt_in(self) -> None:
        """Compliance is reachable in an SVG label, not only by deleting the word."""
        labels = [text for _, text in blocks_of(_ARCHITECTURE_SVG) if _MULTICAST.search(text)]
        assert labels, "the architecture diagram no longer names the LAN transport at all"
        for label in labels:
            assert states_the_posture(label), f"diagram label still claims multicast is the default: {label!r}"
            assert "multicast (default)" not in label.lower()


class TestAPlantedClaimIsCaughtInEveryFormat:
    """One planted stale claim per block reader, so no format is unguarded."""

    def test_a_planted_python_docstring_claim_is_caught(self, tmp_path: Path) -> None:
        path = tmp_path / "planted.py"
        path.write_text('"""Peers find each other over the LAN via multicast scouting."""\n', encoding="utf-8")
        assert unmarked_blocks(path)

    def test_a_planted_python_comment_claim_is_caught(self, tmp_path: Path) -> None:
        path = tmp_path / "planted.py"
        path.write_text("# LAN discovery is automatic: multicast scouting is on.\nX = 1\n", encoding="utf-8")
        assert unmarked_blocks(path)

    def test_a_planted_markdown_paragraph_claim_is_caught(self, tmp_path: Path) -> None:
        path = tmp_path / "planted.md"
        path.write_text("# Mesh\n\nPeers discover each other via multicast scouting.\n", encoding="utf-8")
        assert unmarked_blocks(path)

    def test_a_planted_svg_label_claim_is_caught(self, tmp_path: Path) -> None:
        path = tmp_path / "planted.svg"
        path.write_text('<svg><text x="1" y="2">Zenoh multicast (default)</text></svg>\n', encoding="utf-8")
        assert unmarked_blocks(path)


class TestCorrectProseIsNotFlagged:
    """The shapes that made a sentence-level rule need an exclusion list."""

    def test_the_opt_in_warning_passes_while_saying_multicast_is_on(self, tmp_path: Path) -> None:
        """The live warning text says "ON" and is correct, because it names the flag."""
        path = tmp_path / "warning.py"
        path.write_text(
            "# Multicast scouting is ON (STRANDS_MESH_MULTICAST=true). Any device on\n"
            "# the LAN can discover and attract fleet robots without credentials.\n"
            "X = 1\n",
            encoding="utf-8",
        )
        assert not unmarked_blocks(path)

    def test_off_by_default_passes_without_naming_the_flag(self, tmp_path: Path) -> None:
        """Either form satisfies the rule; a block need not cite the env var."""
        path = tmp_path / "prose.md"
        path.write_text(
            "Multicast scouting is off by default, so cross-host peers need\nexplicit endpoints.\n", encoding="utf-8"
        )
        assert not unmarked_blocks(path)

    def test_a_multi_sentence_rationale_naming_multicast_repeatedly_passes(self, tmp_path: Path) -> None:
        """Block granularity is what keeps a security rationale out of the offender list."""
        path = tmp_path / "rationale.py"
        path.write_text(
            '"""Multicast on a hostile LAN is a discovery attack surface.\n\n'
            "Any host that joins the multicast group sees every peer. Multicast is\n"
            'therefore disabled unless an operator opts in.\n"""\n',
            encoding="utf-8",
        )
        assert not unmarked_blocks(path)


class TestTheDeviceConnectBoundaryIsMeasuredNotAssumed:
    """D2D prose is out of scope because that package configures no Zenoh session."""

    def test_device_connect_configures_no_scouting_key(self) -> None:
        """No scouting key, no config-builder import: a different session entirely."""
        package = _REPO_ROOT / "strands_robots" / "device_connect"
        assert package.is_dir(), "device_connect package moved; re-derive this boundary"
        offenders: list[str] = []
        for path in sorted(package.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if re.search(r"scouting/multicast|STRANDS_MESH_MULTICAST|scouting_block|_zenoh_config", source):
                offenders.append(str(path.relative_to(_REPO_ROOT)))
        assert not offenders, (
            "device_connect now touches the mesh scouting config, so its D2D "
            f"discovery prose is no longer a different transport's default: {offenders}"
        )
