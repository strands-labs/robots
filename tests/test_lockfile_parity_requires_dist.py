"""Contract pins for the whole-set ``requires-dist`` parity check.

#2039 gated the lock with ``uv lock --check`` (advisory) and pinned two
properties of it offline in ``tests/test_lockfile_parity_gate.py`` (required):
every declared distribution present somewhere, and no locked version below its
declared floor. Against ``main``'s stale lock at ``ad8696b`` that pair reports 2
of the 5 rows that had actually drifted. The three it misses are the ones where
the drift is in the *shape* of the recorded set rather than in a property of a
pin:

* ``mink`` and ``qpsolvers`` were both locked -- reachable via ``[cosmos3-sim]``
  -- so a presence check passes while ``[sim-mujoco]`` is recorded as
  ``['imageio', 'imageio-ffmpeg', 'mujoco', 'robot-descriptions']``, a locked sim
  install with no IK stack behind ``move_to``. ``[rosbridge]`` was recorded
  **empty**.
* ``huggingface-hub`` is pinned at ``1.20.1``, which satisfies its ``>=1.5``
  floor, so only the recorded *specifier* was stale (``>=1.0``). No property of a
  correct pin can see that.

``scripts/check_lockfile_parity.py`` compares the whole set instead, in both
directions, against uv's own transcription of the manifest in the root package's
``[package.metadata] requires-dist``. See #2041.

What this module pins, and why each part is here
------------------------------------------------
The load-bearing assertion is one line -- the live pair agrees -- and on its own
it is indistinguishable from a checker that finds nothing because it reads
nothing. So it is surrounded by:

* **Non-vacuity guards.** Both sides must reconstruct a non-trivial number of
  rows. An empty-versus-empty comparison agrees perfectly.
* **The five repaired rows, individually.** Asserted present in *both*
  transcriptions, so this module fails if any of them regresses -- which is what
  makes it a regression pin for #2038 rather than a restatement of the checker.
* **The two encodings, as premises.** Self-reference expansion and
  several-specifiers-per-key are both load-bearing: getting either wrong produces
  a checker that fails a correct lock, which is worse than one that misses drift
  because it blocks every branch. Each is asserted to be *necessary* (the
  unexpanded comparison is shown to manufacture findings) rather than merely
  described.
* **Planted manifest/lock pairs.** Every finding class is provoked on a synthetic
  pair, so an empty finding list on the live pair means agreement rather than
  blindness. These carry the drift classes rather than a vendored copy of the
  5914-line stale lock: the classes are what must keep being caught, and a
  vendored artifact would pin one historical file forever.

The stale lock is not committed here. It is reachable as ``git show
ad8696b:uv.lock`` for anyone reproducing the 24 findings quoted in the script's
docstring.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_lockfile_parity.py"
_PYPROJECT = _ROOT / "pyproject.toml"
_LOCK = _ROOT / "uv.lock"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_lockfile_parity", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The script is annotated and mypy-clean on its own
# (mypy scripts/check_lockfile_parity.py); it is reached through importlib here
# because scripts/ is not an importable package.
mod = _load()


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> dict[str, Any]:
    return tomllib.loads(_LOCK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def declared(manifest: dict[str, Any]) -> set[Any]:
    return mod.declared_rows(manifest)


@pytest.fixture(scope="module")
def recorded(lock: dict[str, Any]) -> set[Any]:
    rows, marker_findings = mod.recorded_rows(lock)
    assert marker_findings == [], f"uv.lock carries a marker this checker cannot compare: {marker_findings}"
    return rows


# ---------------------------------------------------------------------------
# The live pair. One assertion, plus the guards that stop it passing vacuously.
# ---------------------------------------------------------------------------


def test_the_live_pair_agrees_in_both_directions(declared: set[Any], recorded: set[Any]) -> None:
    """``uv.lock``'s transcription of ``pyproject.toml`` matches it exactly.

    The remedy for a failure here is always ``uv lock``, committed. A dependency
    change and its relock belong in the same commit.
    """
    findings = mod.compare(declared, recorded)
    assert findings == [], "uv.lock no longer transcribes pyproject.toml:\n" + "\n".join(
        f"  {finding.kind}: {finding.render()}" for finding in findings
    )


def test_neither_side_reconstructs_an_empty_set(declared: set[Any], recorded: set[Any]) -> None:
    """A parser that returns nothing agrees with anything.

    The floor is deliberately far below the current 111 on each side, so this
    guards against a reader that breaks rather than against ordinary dependency
    churn -- a threshold near the real count would fail on the next extra added.
    """
    assert len(declared) > 80
    assert len(recorded) > 80


def test_the_two_transcriptions_are_the_same_size(declared: set[Any], recorded: set[Any]) -> None:
    """Equal sets have equal size; asserted separately so a size mismatch is legible on its own."""
    assert len(declared) == len(recorded)


def test_the_checker_agrees_with_itself_through_its_file_entry_point() -> None:
    """``check()`` reads the same pair from paths, which is the entry point the script uses."""
    assert mod.check(_PYPROJECT, _LOCK) == []


# ---------------------------------------------------------------------------
# The five rows #2038 repaired, asserted present in both transcriptions.
# ---------------------------------------------------------------------------

#: ``(extra, distribution, extras)`` for each row, with the floor that must be
#: recorded where the drift was in the specifier rather than in presence.
REPAIRED_ROWS = [
    pytest.param("rosbridge", "roslibpy", (), None, id="roslibpy-recorded-under-rosbridge"),
    pytest.param("sim-mujoco", "mink", (), None, id="mink-recorded-under-sim-mujoco"),
    pytest.param("sim-mujoco", "qpsolvers", ("daqp",), None, id="qpsolvers-recorded-under-sim-mujoco"),
    pytest.param("sim-newton", "mink", (), None, id="mink-recorded-under-sim-newton"),
    pytest.param("sim-newton", "qpsolvers", ("daqp",), None, id="qpsolvers-recorded-under-sim-newton"),
    pytest.param("lerobot", "lerobot", ("dataset", "feetech"), "0.6.1", id="lerobot-floor-recorded-as-0.6.1"),
    pytest.param("wbc", "huggingface-hub", (), "1.5", id="huggingface-hub-floor-recorded-as-1.5"),
]


@pytest.mark.parametrize(("extra", "name", "extras", "floor"), REPAIRED_ROWS)
def test_a_repaired_row_is_present_in_both_transcriptions(
    extra: str,
    name: str,
    extras: tuple[str, ...],
    floor: str | None,
    declared: set[Any],
    recorded: set[Any],
) -> None:
    """Each row that had drifted is now declared *and* recorded, with the floor it was given.

    ``roslibpy`` and the two IK packages were absent from the recorded set
    entirely; ``lerobot`` and ``huggingface-hub`` were recorded at a floor below
    the declared one. Both failure modes are asserted the same way here, because
    both are answered by the row being in both sets.
    """
    key = (extra, name, extras)
    declared_here = {row.specifier for row in declared if row.key == key}
    recorded_here = {row.specifier for row in recorded if row.key == key}
    assert declared_here, f"pyproject.toml no longer declares {name}{list(extras)} under [{extra}]"
    assert recorded_here, f"uv.lock no longer records {name}{list(extras)} under [{extra}] -- relock"
    assert declared_here == recorded_here
    if floor is not None:
        assert all(floor in specifier for specifier in recorded_here), (
            f"uv.lock records {name} under [{extra}] as {recorded_here}, which does not carry the {floor} floor"
        )


# ---------------------------------------------------------------------------
# The two encodings, as premises. Each is asserted to be necessary.
# ---------------------------------------------------------------------------


def test_the_manifest_uses_self_references(manifest: dict[str, Any]) -> None:
    """The premise behind expansion: the manifest declares itself with extras.

    If this ever stops being true the expansion becomes dead code, and the next
    reader should delete it rather than maintain it.
    """
    lines = [
        spec
        for specs in (manifest["project"].get("optional-dependencies") or {}).values()
        for spec in specs
        if mod.canonical_name(Requirement(spec).name) == mod.PROJECT_NAME
    ]
    assert lines, "no strands-robots[...] self-reference in pyproject.toml"


def test_expansion_is_load_bearing_not_cosmetic(manifest: dict[str, Any], recorded: set[Any]) -> None:
    """Comparing the literal declared lines manufactures findings on a correct lock.

    uv records the transitive closure of each self-reference, so the unexpanded
    declared set is strictly smaller than the recorded one. A checker written
    without the expansion does not merely miss drift -- it reports a correct lock
    as drifted, and would fail every branch.
    """
    project = manifest["project"]
    unexpanded: set[Any] = set()
    for spec in project.get("dependencies") or []:
        requirement = Requirement(str(spec))
        unexpanded.add(
            mod.Row(
                None,
                mod.canonical_name(requirement.name),
                tuple(sorted(requirement.extras)),
                str(requirement.specifier),
            )
        )
    for extra, specs in (project.get("optional-dependencies") or {}).items():
        for spec in specs:
            requirement = Requirement(str(spec))
            if mod.canonical_name(requirement.name) == mod.PROJECT_NAME:
                continue
            unexpanded.add(
                mod.Row(
                    extra,
                    mod.canonical_name(requirement.name),
                    tuple(sorted(requirement.extras)),
                    str(requirement.specifier),
                )
            )
    assert len(unexpanded) < len(recorded)
    assert mod.compare(unexpanded, recorded), "the unexpanded comparison agreed, so expansion cannot be load-bearing"


def test_a_key_recorded_at_two_specifiers_is_compared_as_a_set(tmp_path: Path) -> None:
    """The ``scipy`` case, as observed in ``ad8696b``'s lock.

    One key legitimately carries two specifiers on both sides. A checker that
    flattened either side into a dictionary keyed by ``(extra, name, extras)``
    would keep one specifier and report the other as drift -- a finding on a
    correct file, which blocks every branch until someone reads the checker.
    """
    pyproject, lock = _plant(
        tmp_path,
        "dependencies = []\n\n[project.optional-dependencies]\n"
        'sim-mujoco = ["scipy>=1.10.0"]\n'
        'wbc = ["scipy>=1.14.0,<2.0.0"]\n'
        'all = ["strands-robots[sim-mujoco]", "strands-robots[wbc]"]\n',
        '    { name = "scipy", marker = "extra == \'sim-mujoco\'", specifier = ">=1.10.0" },\n'
        '    { name = "scipy", marker = "extra == \'wbc\'", specifier = ">=1.14.0,<2.0.0" },\n'
        '    { name = "scipy", marker = "extra == \'all\'", specifier = ">=1.10.0" },\n'
        '    { name = "scipy", marker = "extra == \'all\'", specifier = ">=1.14.0,<2.0.0" },',
    )
    assert mod.check(pyproject, lock) == []


def test_dropping_one_of_two_specifiers_for_a_key_is_reported(tmp_path: Path) -> None:
    """The counterpart: the lock records only one of the two, which is real drift.

    Without this, the test above could pass on a checker that compares only
    whether the key is present.
    """
    pyproject, lock = _plant(
        tmp_path,
        "dependencies = []\n\n[project.optional-dependencies]\n"
        'sim-mujoco = ["scipy>=1.10.0"]\n'
        'wbc = ["scipy>=1.14.0,<2.0.0"]\n'
        'all = ["strands-robots[sim-mujoco]", "strands-robots[wbc]"]\n',
        '    { name = "scipy", marker = "extra == \'sim-mujoco\'", specifier = ">=1.10.0" },\n'
        '    { name = "scipy", marker = "extra == \'wbc\'", specifier = ">=1.14.0,<2.0.0" },\n'
        '    { name = "scipy", marker = "extra == \'all\'", specifier = ">=1.10.0" },',
    )
    findings = mod.check(pyproject, lock)
    assert [finding.kind for finding in findings] == [mod.SPECIFIER_DRIFT]
    assert findings[0].key == ("all", "scipy", ())
    # Rendered through ``SpecifierSet``, which sorts within a specifier, and the
    # tuple itself is sorted -- so the expectation is written normalised too.
    assert findings[0].declared == ("<2.0.0,>=1.14.0", ">=1.10.0")
    assert findings[0].recorded == (">=1.10.0",)


def test_every_recorded_marker_is_a_plain_extra_marker(lock: dict[str, Any]) -> None:
    """The comparison is exact only while markers say nothing but ``extra == '...'``."""
    root = next(p for p in lock["package"] if mod.canonical_name(p["name"]) == mod.PROJECT_NAME)
    markers = [entry["marker"] for entry in root["metadata"]["requires-dist"] if entry.get("marker")]
    assert markers, "no marker at all in requires-dist, which would make the extra comparison vacuous"
    assert all(mod._EXTRA_MARKER.fullmatch(marker.strip()) for marker in markers)


def test_no_declared_requirement_carries_an_environment_marker(manifest: dict[str, Any]) -> None:
    """The other half of the same premise, read off the manifest.

    Inert today. Pinned so that adding a marked requirement fails here -- naming
    the assumption -- rather than silently comparing fewer rows.
    """
    project = manifest["project"]
    specs = list(project.get("dependencies") or [])
    for extra_specs in (project.get("optional-dependencies") or {}).values():
        specs.extend(extra_specs)
    marked = [spec for spec in specs if Requirement(str(spec)).marker is not None]
    assert marked == []


# ---------------------------------------------------------------------------
# Planted pairs. Every finding class provoked on a synthetic manifest and lock,
# so an empty finding list on the live pair means agreement, not blindness.
# ---------------------------------------------------------------------------


def _plant(tmp_path: Path, manifest_body: str, requires_dist: str) -> tuple[Path, Path]:
    """Write a minimal manifest/lock pair and return the two paths.

    Planted manifests that use a ``strands-robots[...]`` self-reference name a
    **real declared extra** (``sim-mujoco``, ``wbc``) rather than a placeholder.
    That is not cosmetic: ``tests/test_dependency_audit.py`` sweeps the tree for
    ``strands-robots[<extra>]`` and refuses any that names an extra which does not
    exist, because pip exits 0 on an unknown extra and installs none of its
    dependencies -- so the mistake surfaces later and misattributed. A placeholder
    here fails that sweep. The dependencies inside these fixtures stay synthetic;
    only the extra names are real.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[project]\nname = "strands-robots"\nversion = "0.0.0"\n{manifest_body}', encoding="utf-8")
    lock = tmp_path / "uv.lock"
    lock.write_text(
        "version = 1\nrevision = 3\n\n"
        '[[package]]\nname = "strands-robots"\nsource = { virtual = "." }\n\n'
        f"[package.metadata]\nrequires-dist = [\n{requires_dist}\n]\n",
        encoding="utf-8",
    )
    return pyproject, lock


_AGREEING_MANIFEST = 'dependencies = ["foo>=1.0,<2.0"]\n\n[project.optional-dependencies]\nx = ["bar[extra1]>=2.0"]\n'
_AGREEING_LOCK = (
    '    { name = "foo", specifier = ">=1.0,<2.0" },\n'
    '    { name = "bar", extras = ["extra1"], marker = "extra == \'x\'", specifier = ">=2.0" },'
)


def test_a_matching_planted_pair_reports_nothing(tmp_path: Path) -> None:
    """The control. Without it, every finding test below could pass on a checker that always fires."""
    pyproject, lock = _plant(tmp_path, _AGREEING_MANIFEST, _AGREEING_LOCK)
    assert mod.check(pyproject, lock) == []


def test_a_declared_distribution_absent_from_the_lock_is_reported(tmp_path: Path) -> None:
    """The ``roslibpy`` class: declared under an extra, recorded nowhere."""
    pyproject, lock = _plant(
        tmp_path,
        _AGREEING_MANIFEST,
        '    { name = "foo", specifier = ">=1.0,<2.0" },',
    )
    findings = mod.check(pyproject, lock)
    assert [finding.kind for finding in findings] == [mod.MISSING_FROM_LOCK]
    assert findings[0].key == ("x", "bar", ("extra1",))


def test_a_recorded_distribution_the_manifest_does_not_declare_is_reported(tmp_path: Path) -> None:
    """The ``peft`` class: an extra stopped declaring it and the lock still records it."""
    pyproject, lock = _plant(
        tmp_path,
        'dependencies = ["foo>=1.0,<2.0"]\n',
        _AGREEING_LOCK,
    )
    findings = mod.check(pyproject, lock)
    assert [finding.kind for finding in findings] == [mod.ABSENT_FROM_MANIFEST]
    assert findings[0].key == ("x", "bar", ("extra1",))


def test_a_lowered_floor_is_one_finding_naming_both_sides(tmp_path: Path) -> None:
    """The ``lerobot`` / ``huggingface-hub`` class.

    Reported once, as drift, rather than twice as an absence and an addition:
    the remedy is a relock, and two rows would read as two problems.
    """
    pyproject, lock = _plant(
        tmp_path,
        _AGREEING_MANIFEST,
        '    { name = "foo", specifier = ">=1.0,<2.0" },\n'
        '    { name = "bar", extras = ["extra1"], marker = "extra == \'x\'", specifier = ">=1.0" },',
    )
    findings = mod.check(pyproject, lock)
    assert [finding.kind for finding in findings] == [mod.SPECIFIER_DRIFT]
    assert findings[0].declared == (">=2.0",)
    assert findings[0].recorded == (">=1.0",)


def test_the_extras_of_a_requirement_are_part_of_its_identity(tmp_path: Path) -> None:
    """``qpsolvers[daqp]`` recorded as bare ``qpsolvers`` is drift, not a match."""
    pyproject, lock = _plant(
        tmp_path,
        _AGREEING_MANIFEST,
        '    { name = "foo", specifier = ">=1.0,<2.0" },\n'
        '    { name = "bar", marker = "extra == \'x\'", specifier = ">=2.0" },',
    )
    findings = mod.check(pyproject, lock)
    kinds = sorted(finding.kind for finding in findings)
    assert kinds == sorted([mod.ABSENT_FROM_MANIFEST, mod.MISSING_FROM_LOCK])


def test_a_marker_that_is_not_an_extra_marker_is_reported_rather_than_dropped(tmp_path: Path) -> None:
    """An environment marker must not silently shrink the compared set."""
    pyproject, lock = _plant(
        tmp_path,
        _AGREEING_MANIFEST,
        '    { name = "foo", specifier = ">=1.0,<2.0" },\n'
        '    { name = "bar", extras = ["extra1"], marker = "extra == \'x\'", specifier = ">=2.0" },\n'
        '    { name = "baz", marker = "python_full_version < \'3.13\'", specifier = ">=1.0" },',
    )
    findings = mod.check(pyproject, lock)
    assert mod.UNREADABLE_MARKER in [finding.kind for finding in findings]


def test_specifier_ordering_is_not_drift(tmp_path: Path) -> None:
    """``>=1.0,<2.0`` and ``<2.0,>=1.0`` are the same constraint.

    Without normalisation this reports the whole manifest as drifted the first
    time uv writes a specifier in a different order from the manifest.
    """
    pyproject, lock = _plant(
        tmp_path,
        'dependencies = ["foo>=1.0,<2.0"]\n',
        '    { name = "foo", specifier = "<2.0,>=1.0" },',
    )
    assert mod.check(pyproject, lock) == []


def test_a_self_reference_cycle_terminates(tmp_path: Path) -> None:
    """Two extras referencing each other must not recurse until the interpreter stops it.

    The manifest has no cycle today, so this is a property of the reader rather
    than of the input -- which is exactly why it needs a planted pair.
    """
    pyproject, lock = _plant(
        tmp_path,
        "dependencies = []\n\n[project.optional-dependencies]\n"
        'sim-mujoco = ["strands-robots[wbc]", "foo>=1.0"]\n'
        'wbc = ["strands-robots[sim-mujoco]", "bar>=2.0"]\n',
        '    { name = "foo", marker = "extra == \'sim-mujoco\'", specifier = ">=1.0" },\n'
        '    { name = "bar", marker = "extra == \'sim-mujoco\'", specifier = ">=2.0" },\n'
        '    { name = "foo", marker = "extra == \'wbc\'", specifier = ">=1.0" },\n'
        '    { name = "bar", marker = "extra == \'wbc\'", specifier = ">=2.0" },',
    )
    assert mod.check(pyproject, lock) == []


def test_a_name_is_compared_canonically(tmp_path: Path) -> None:
    """``zope.interface`` and ``zope-interface`` are one distribution (PEP 503)."""
    pyproject, lock = _plant(
        tmp_path,
        'dependencies = ["Zope.Interface>=5.0"]\n',
        '    { name = "zope-interface", specifier = ">=5.0" },',
    )
    assert mod.check(pyproject, lock) == []


# ---------------------------------------------------------------------------
# The script's exit contract, which is what makes it usable before a push.
# ---------------------------------------------------------------------------


def test_main_exits_zero_on_the_live_pair(capsys: pytest.CaptureFixture[str]) -> None:
    assert mod.main(["--repo-root", str(_ROOT)]) == 0
    assert "findings | **0**" in capsys.readouterr().out


def test_main_exits_one_and_annotates_on_a_finding(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pyproject, lock = _plant(tmp_path, _AGREEING_MANIFEST, '    { name = "foo", specifier = ">=1.0,<2.0" },')
    assert mod.main(["--pyproject", str(pyproject), "--lock", str(lock)]) == 1
    out = capsys.readouterr().out
    assert "::error title=uv.lock no longer transcribes pyproject.toml::" in out
    assert "uv lock" in out


def test_main_exits_two_when_a_file_is_missing(tmp_path: Path) -> None:
    """Distinct from a finding: an unreadable input is not evidence of drift."""
    assert mod.main(["--repo-root", str(tmp_path)]) == 2


def test_main_exits_two_on_a_lock_with_no_root_metadata(tmp_path: Path) -> None:
    """A lock uv did not write, or one from a future format, is refused rather than read as clean."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "strands-robots"\nversion = "0.0.0"\ndependencies = []\n', encoding="utf-8")
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\n\n[[package]]\nname = "strands-robots"\n', encoding="utf-8")
    assert mod.main(["--pyproject", str(pyproject), "--lock", str(lock)]) == 2
