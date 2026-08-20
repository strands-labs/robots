"""``docs/policies/cosmos3.md`` enumerates the embodiments the provider registers.

The cosmos3 provider is the one policy whose behaviour is selected by a second
name: ``create_policy("cosmos3", embodiment=...)``. That name picks the
conditioning domain, the action width and the column layout, so the set of
accepted embodiments is a public API surface in its own right - and it is
enumerated **nine times by hand**. Six sit on the pages a reader consults: the
front-matter description, the ``## Embodiments`` table, the inline
``# droid | umi | ...`` comment in the first worked example, the
domain/width/bundled-stats table, the bundled-vs-unbundled count in the prose
above it, and the provider row in ``README.md``. Three more were found by
sweeping for the class rather than by reading the provider page, and are graded
here for the same reason: the README quickstart's ``Embodiments: ...``
paragraph, which sits under the runnable rollout command and is what a reader
who never opens the provider page relies on (it also states three of ``droid``'s
entry facts); the ``Available embodiments:`` sentence in the package docstring,
one import from the registry and what ``help()`` prints; and the parenthetical
in :class:`~strands_robots.policies.cosmos3.policy.Cosmos3Policy`'s
``embodiment:`` ``Args:`` entry, which is the accepted-value list for the
parameter a caller passes.

The two docstring surfaces are graded through ``__doc__`` rather than by
reading the source, so a reflow or a moved definition cannot disarm them and
the graded text is exactly what a reader is shown.

One nearby enumeration is deliberately **not** graded: the README's "other
embodiments such as ``umi``/``av``/``bridge`` need only ``observation/image``"
sentence. "such as" disclaims exhaustiveness, so requiring it to name every
embodiment would convert a hedge into a maintained list. The claim it does make
is about :attr:`Cosmos3Embodiment.camera_keys`, not about the accepted set, and
belongs to a camera-key guard rather than this one.

Nothing tied any of those to
:data:`~strands_robots.policies.cosmos3.embodiments.EMBODIMENTS`. The
provider-level catalogue is guarded - ``tests/test_docs_policy_coverage.py``
ties the overview table to ``policies.json`` - but that guard grades
*providers*, so an embodiment added to an existing provider is graded by
nothing, and a reader is told the accepted set is smaller than it is. The
failure is silent in the direction that matters: the code accepts the new name,
so no call breaks, and only the documentation disagrees.

The companion guard in this directory,
``tests/policies/cosmos3/test_documented_backend_knob_routes.py``, already
applies the same rule to this page's *keywords*: it grades them against
``inspect.signature`` "rather than against a copied list, so ... a newly
documented one is graded without touching this file". This file applies that
rule to the page's embodiment names, and to the per-embodiment facts the tables
state (domain, raw action width, whether quantile stats ship bundled), which are
read from the entry and from the stats directory rather than restated here.

Every grader below is a pure function of ``(page text, registry)`` so
:class:`TestTheGradersAreNotVacuous` can hand them a registry carrying an
embodiment the page cannot mention and assert each one reports it. A grader
that cannot see a missing embodiment would report a clean page forever.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from strands_robots.policies import cosmos3 as cosmos3_package
from strands_robots.policies.cosmos3.embodiments import EMBODIMENTS, Cosmos3Embodiment
from strands_robots.policies.cosmos3.policy import Cosmos3Policy

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PAGE = _REPO_ROOT / "docs" / "policies" / "cosmos3.md"
_README = _REPO_ROOT / "README.md"
_STATS_DIR = _REPO_ROOT / "strands_robots" / "policies" / "cosmos3" / "stats"

# The bundled-stats column states a property of the shipped package, not of the
# entry: load_action_stats() resolves "<domain>_stats.json" in this directory and
# refuses the domain when the file is absent. Reading the directory keeps the
# column true when a domain's quantiles are added or removed.
_STATS_SUFFIX = "_stats.json"

# The README quickstart's rollout section states the accepted set in prose, and
# for ``droid`` it also states three facts that live on the entry. Located by the
# paragraph's opening word rather than by line number, so a re-wrap or a section
# move does not disarm the graders below.
_QUICKSTART_PREFIX = "Embodiments:"

# The parenthetical facts that paragraph states, mapped to the entry attribute
# each one restates. A parenthetical carrying none of these patterns states no
# fact and is graded on nothing, which is what lets "(post-training only)" and
# any future annotation coexist with the graded numbers.
_QUICKSTART_FACTS: tuple[tuple[str, str], ...] = (
    ("raw_action_dim", r"\b(\d+)D\b"),
    ("action_chunk_size", r"\bchunk (\d+)\b"),
    ("fps", r"\b(\d+) fps\b"),
)

# Spelled cardinals the count sentence uses. Small on purpose: the sentence
# describes a split of the registered embodiments, and a registry large enough to
# need a bigger word would have been rewritten to a digit long before.
_CARDINALS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _page_text() -> str:
    """Return the cosmos3 provider page."""
    return _PAGE.read_text(encoding="utf-8")


def _readme_text() -> str:
    """Return the repository README."""
    return _README.read_text(encoding="utf-8")


def _bundled_domains() -> set[str]:
    """Return the domains whose quantile stats ship in the package."""
    return {p.name[: -len(_STATS_SUFFIX)] for p in _STATS_DIR.glob(f"*{_STATS_SUFFIX}")}


def _cells(line: str) -> list[str]:
    """Split one markdown table row into its cells.

    Args:
        line: A single ``| a | b |`` table line.

    Returns:
        The inner cells, stripped. Splits on unescaped pipes so a cell
        containing ``\\|`` stays one cell.
    """
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def _table(md: str, header: list[str]) -> list[list[str]] | None:
    """Return the rows of the table whose header matches ``header``.

    Located by header cells rather than by position, so the page can grow
    sections above or below without moving the graded table.

    Args:
        md: Markdown document text.
        header: Expected header cells, lower-cased.

    Returns:
        The data rows, or ``None`` when no table carries that header.
    """
    lines = md.split("\n")
    for i, line in enumerate(lines[:-1]):
        if not line.strip().startswith("|"):
            continue
        if not re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            continue
        if [h.lower() for h in _cells(line)] != header:
            continue
        rows = []
        for row in lines[i + 2 :]:
            if not row.strip().startswith("|"):
                break
            rows.append(_cells(row))
        return rows
    return None


def _names(cell: str) -> set[str]:
    """Return the backtick-quoted identifiers in a cell or sentence."""
    return set(re.findall(r"`([A-Za-z0-9_]+)`", cell))


# --------------------------------------------------------------------------- #
# Graders. Each takes the registry so the vacuity meta-test can plant an entry.
# Each returns the embodiments the surface fails to account for, both ways.
# --------------------------------------------------------------------------- #


def _front_matter_gap(md: str, registered: dict[str, Cosmos3Embodiment]) -> set[str]:
    """Return registered embodiments the front-matter description omits."""
    match = re.search(r"^description:(.*)$", md, re.M)
    assert match is not None, f"{_PAGE} has no front-matter 'description:' line"
    described = set(re.findall(r"[a-z0-9_]+", match.group(1)))
    return set(registered) - described


def _embodiment_table_gap(md: str, registered: dict[str, Cosmos3Embodiment]) -> tuple[set[str], set[str]]:
    """Return (registered but undocumented, documented but unregistered)."""
    rows = _table(md, ["embodiment", "robot hardware", "strands sim asset"])
    assert rows is not None, f"{_PAGE} has no '## Embodiments' table (header changed?)"
    listed = {row[0].strip("`") for row in rows if row}
    return set(registered) - listed, listed - set(registered)


def _inline_enum_gap(md: str, registered: dict[str, Cosmos3Embodiment]) -> tuple[set[str], set[str]]:
    """Return the gap in the ``embodiment=...  # a | b`` example comment."""
    match = re.search(r'embodiment="[^"]*",\s*#\s*([A-Za-z0-9_ |]+)', md)
    assert match is not None, f"{_PAGE} has no 'embodiment=\"...\"  # a | b' example comment"
    listed = {tok.strip() for tok in match.group(1).split("|") if tok.strip()}
    return set(registered) - listed, listed - set(registered)


def _domain_table_gap(md: str, registered: dict[str, Cosmos3Embodiment]) -> tuple[set[str], set[str]]:
    """Return the gap in the domain / raw dim / bundled-stats table."""
    rows = _table(md, ["embodiment", "domain", "raw dim", "bundled stats"])
    assert rows is not None, f"{_PAGE} has no domain/raw-dim/bundled-stats table (header changed?)"
    listed = {row[0].strip("`") for row in rows if row}
    return set(registered) - listed, listed - set(registered)


def _domain_table_facts(md: str, registered: dict[str, Cosmos3Embodiment]) -> list[str]:
    """Return one message per row whose stated facts disagree with the entry."""
    rows = _table(md, ["embodiment", "domain", "raw dim", "bundled stats"]) or []
    bundled = _bundled_domains()
    problems = []
    for row in rows:
        name = row[0].strip("`")
        entry = registered.get(name)
        if entry is None:
            continue
        stated_domain, stated_dim, stated_bundled = row[1].strip("`"), row[2].strip(), row[3].strip().lower()
        if stated_domain != entry.domain_name:
            problems.append(f"{name}: table says domain {stated_domain!r}, entry says {entry.domain_name!r}")
        if stated_dim != str(entry.raw_action_dim):
            problems.append(f"{name}: table says raw dim {stated_dim}, entry says {entry.raw_action_dim}")
        truth = "yes" if entry.domain_name in bundled else "no"
        if stated_bundled != truth:
            problems.append(
                f"{name}: table says bundled stats {stated_bundled!r}, but "
                f"{entry.domain_name}{_STATS_SUFFIX} is "
                f"{'present' if truth == 'yes' else 'absent'} in {_STATS_DIR.name}/"
            )
    return problems


def _readme_gap(readme: str, registered: dict[str, Cosmos3Embodiment]) -> set[str]:
    """Return registered embodiments the README's cosmos3 provider row omits."""
    rows = _table(readme, ["provider", "backend", "notes"])
    assert rows is not None, f"{_README} has no '| Provider | Backend | Notes |' table"
    row = [r for r in rows if r and r[0].strip("`") == "cosmos3"]
    assert row, f"{_README} provider table has no 'cosmos3' row"
    return set(registered) - _names(row[0][-1])


def _quickstart_paragraph(readme: str) -> str:
    """Return the README quickstart paragraph that enumerates the embodiments.

    Args:
        readme: README text.

    Returns:
        The paragraph with its internal wrapping collapsed to single spaces, so
        the graders reading it are insensitive to a re-wrap.
    """
    found = [
        collapsed
        for block in readme.split("\n\n")
        if (collapsed := " ".join(block.split())).startswith(_QUICKSTART_PREFIX)
    ]
    assert len(found) == 1, (
        f"{_README} has {len(found)} paragraphs beginning {_QUICKSTART_PREFIX!r}, expected exactly 1 - "
        "the cosmos3 quickstart's embodiment paragraph moved, was reworded or was duplicated."
    )
    return found[0]


def _quickstart_gap(readme: str, registered: dict[str, Cosmos3Embodiment]) -> set[str]:
    """Return registered embodiments the README quickstart paragraph omits.

    Only the missing direction is graded, for the same reason as
    :func:`_readme_gap`: the paragraph is prose and carries backtick-quoted
    identifiers that are not embodiments (``ConnectionError``), so an "extra"
    check here would report the prose rather than a drift. The ``## Embodiments``
    table is where a name the registry does not accept is caught.

    Args:
        readme: README text.
        registered: Embodiment registry to grade against.

    Returns:
        The registered names the paragraph does not mention.
    """
    return set(registered) - _names(_quickstart_paragraph(readme))


def _quickstart_facts(readme: str, registered: dict[str, Cosmos3Embodiment]) -> list[str]:
    """Return one message per quickstart parenthetical fact the entry contradicts.

    Args:
        readme: README text.
        registered: Embodiment registry to grade against.

    Returns:
        A list with one message per contradicted fact, empty when every stated
        number agrees with its entry.
    """
    problems = []
    for name, paren in re.findall(r"`([A-Za-z0-9_]+)`\s*\(([^)]*)\)", _quickstart_paragraph(readme)):
        entry = registered.get(name)
        if entry is None:
            continue
        for attr, pattern in _QUICKSTART_FACTS:
            match = re.search(pattern, paren)
            if match is None:
                continue
            stated, truth = int(match.group(1)), getattr(entry, attr)
            if stated != truth:
                problems.append(f"{name}: quickstart says {attr.replace('_', ' ')} {stated}, entry says {truth}")
    return problems


def _package_docstring_gap(doc: str, registered: dict[str, Cosmos3Embodiment]) -> set[str]:
    """Return registered embodiments the package docstring's list omits.

    Args:
        doc: ``strands_robots.policies.cosmos3.__doc__``.
        registered: Embodiment registry to grade against.

    Returns:
        The registered names absent from the ``Available embodiments:`` sentence.
    """
    match = re.search(r"Available embodiments:\s*([^(]*)", " ".join(doc.split()))
    assert match is not None, (
        "strands_robots.policies.cosmos3.__doc__ has no 'Available embodiments: ...' sentence. It is what "
        "help() on the package prints, so the sentence is the graded surface - reword it and this guard "
        "must be re-pointed rather than silently reporting a clean set."
    )
    return set(registered) - set(re.findall(r"[a-z0-9_]+", match.group(1)))


def _policy_args_gap(doc: str, registered: dict[str, Cosmos3Embodiment]) -> set[str]:
    """Return registered embodiments the ``embodiment:`` Args entry omits.

    Args:
        doc: :class:`Cosmos3Policy`'s docstring.
        registered: Embodiment registry to grade against.

    Returns:
        The registered names absent from the entry's first parenthetical.
    """
    match = re.search(r"embodiment:[^(]*\(([^)]*)\)", " ".join(doc.split()))
    assert match is not None, (
        "Cosmos3Policy.__doc__ has no 'embodiment: ... (...)' Args entry. That parenthetical is the "
        "accepted-value list for the parameter a caller passes, so it is the graded surface."
    )
    return set(registered) - set(re.findall(r"[a-z0-9_]+", match.group(1)))


def _count_sentence_problems(md: str, registered: dict[str, Cosmos3Embodiment]) -> list[str]:
    """Return a message when the bundled/unbundled count sentence disagrees.

    The sentence above the domain table states the split as words ("Two domains
    ship them bundled; the other two ... do not"). It is graded only when that
    shape is present, so rewording it to carry no count is allowed - what is not
    allowed is a stated count that the registry contradicts.

    Args:
        md: Markdown document text.
        registered: Embodiment registry to grade against.

    Returns:
        A list with one message per contradicted count, empty when the sentence
        agrees or states no count.
    """
    # The sentence wraps across a line break in the page, so collapse
    # whitespace before matching: a re-wrap must not silently disarm this.
    match = re.search(
        r"(\w+) domains? ship them bundled; the other (\w+) registered embodiments? do not",
        " ".join(md.split()),
    )
    if match is None:
        return []
    bundled_domains = _bundled_domains()
    bundled = {n for n, e in registered.items() if e.domain_name in bundled_domains}
    unbundled = set(registered) - bundled
    problems = []
    for word, truth, label in (
        (match.group(1), len(bundled), "bundled"),
        (match.group(2), len(unbundled), "unbundled"),
    ):
        stated = _CARDINALS.get(word.lower())
        if stated is not None and stated != truth:
            problems.append(f"sentence says {word!r} ({stated}) {label}, registry has {truth}: {sorted(registered)}")
    return problems


class TestEveryRegisteredEmbodimentIsDocumented:
    """The page and the README account for exactly the registered set."""

    def test_front_matter_description_names_every_embodiment(self) -> None:
        missing = _front_matter_gap(_page_text(), EMBODIMENTS)
        assert not missing, (
            f"docs/policies/cosmos3.md front-matter omits {sorted(missing)}. It is the "
            "page's search/summary line, so an embodiment absent there is one a reader "
            "browsing the docs never learns create_policy('cosmos3') accepts."
        )

    def test_embodiments_table_lists_exactly_the_registered_set(self) -> None:
        missing, extra = _embodiment_table_gap(_page_text(), EMBODIMENTS)
        assert not missing, (
            f"the '## Embodiments' table omits {sorted(missing)}, which "
            "create_policy('cosmos3', embodiment=...) accepts. Add a row naming the "
            "hardware and the strands sim asset (or '-' when there is none)."
        )
        assert not extra, (
            f"the '## Embodiments' table lists {sorted(extra)}, which the registry does "
            "not accept - a reader following the table gets a loud unknown-embodiment "
            "failure. Remove the row or register the embodiment."
        )

    def test_first_example_comment_lists_exactly_the_registered_set(self) -> None:
        missing, extra = _inline_enum_gap(_page_text(), EMBODIMENTS)
        assert not missing, (
            f"the 'embodiment=\"...\"  # ...' comment in the page's first example omits "
            f"{sorted(missing)}. That comment is the enumeration a reader copying the "
            "snippet reads, so it is the most-read list on the page."
        )
        assert not extra, f"the first example's comment lists unregistered embodiments {sorted(extra)}"

    def test_domain_table_lists_exactly_the_registered_set(self) -> None:
        missing, extra = _domain_table_gap(_page_text(), EMBODIMENTS)
        assert not missing, (
            f"the domain/raw-dim/bundled-stats table omits {sorted(missing)}. That table "
            "is what tells a caller whether it must supply stats= and stats_domain= to "
            "decode_cosmos_chunk_to_targets, so an omitted row reads as 'no stats needed'."
        )
        assert not extra, f"the domain table lists unregistered embodiments {sorted(extra)}"

    def test_readme_provider_row_names_every_embodiment(self) -> None:
        missing = _readme_gap(_readme_text(), EMBODIMENTS)
        assert not missing, (
            f"the README cosmos3 provider row omits {sorted(missing)}. The row enumerates "
            "the accepted embodiments, so it drifts the same way the page does."
        )

    def test_readme_quickstart_paragraph_names_every_embodiment(self) -> None:
        missing = _quickstart_gap(_readme_text(), EMBODIMENTS)
        assert not missing, (
            f"the README cosmos3 quickstart's 'Embodiments: ...' paragraph omits {sorted(missing)}. It sits "
            "directly under the runnable rollout command, so it is the enumeration a reader who never opens "
            "the provider page relies on."
        )

    def test_package_docstring_names_every_embodiment(self) -> None:
        missing = _package_docstring_gap(cosmos3_package.__doc__ or "", EMBODIMENTS)
        assert not missing, (
            f"the 'Available embodiments:' sentence in strands_robots.policies.cosmos3's docstring omits "
            f"{sorted(missing)}. It is one import from the registry and is what help() on the package prints."
        )

    def test_policy_embodiment_arg_names_every_embodiment(self) -> None:
        missing = _policy_args_gap(Cosmos3Policy.__doc__ or "", EMBODIMENTS)
        assert not missing, (
            f"Cosmos3Policy's 'embodiment:' Args entry omits {sorted(missing)}. That parenthetical is the "
            "accepted-value list for the parameter, so a caller reading it is told the registry accepts less "
            "than it does."
        )


class TestTheDocumentedFactsMatchTheEntries:
    """The per-embodiment facts the tables state are read, not restated."""

    def test_domain_table_facts_match_the_registry_and_the_stats_directory(self) -> None:
        problems = _domain_table_facts(_page_text(), EMBODIMENTS)
        assert not problems, "docs/policies/cosmos3.md domain table disagrees with the code:\n  " + "\n  ".join(
            problems
        )

    def test_quickstart_parenthetical_facts_match_the_registry(self) -> None:
        problems = _quickstart_facts(_readme_text(), EMBODIMENTS)
        assert not problems, (
            "the README cosmos3 quickstart states per-embodiment facts the registry contradicts:\n  "
            + "\n  ".join(problems)
        )

    def test_the_bundled_count_sentence_matches_the_registry(self) -> None:
        problems = _count_sentence_problems(_page_text(), EMBODIMENTS)
        assert not problems, (
            "the sentence above the domain table states a count the registry contradicts:\n  " + "\n  ".join(problems)
        )


class TestThePremisesHold:
    """A reformat must fail loudly rather than make the graders report clean."""

    def test_every_graded_surface_is_found(self) -> None:
        md, readme = _page_text(), _readme_text()
        assert _table(md, ["embodiment", "robot hardware", "strands sim asset"])
        assert _table(md, ["embodiment", "domain", "raw dim", "bundled stats"])
        assert _table(readme, ["provider", "backend", "notes"])
        assert re.search(r'embodiment="[^"]*",\s*#', md)
        assert re.search(r"^description:", md, re.M)
        assert _quickstart_paragraph(readme)
        assert re.search(r"Available embodiments:", " ".join((cosmos3_package.__doc__ or "").split()))
        assert re.search(r"embodiment:[^(]*\(", " ".join((Cosmos3Policy.__doc__ or "").split()))

    def test_the_registry_is_non_trivial(self) -> None:
        assert len(EMBODIMENTS) >= 4, (
            "fewer embodiments than the four this guard was written against - if the "
            "registry shrank deliberately, re-read the graded surfaces."
        )

    def test_the_stats_directory_is_readable(self) -> None:
        assert _bundled_domains(), (
            f"no *{_STATS_SUFFIX} under {_STATS_DIR}: the bundled-stats column would grade "
            "every domain as unbundled and the count sentence would follow it."
        )


class TestTheGradersAreNotVacuous:
    """Each grader must report an embodiment the page cannot mention."""

    @staticmethod
    def _planted() -> dict[str, Cosmos3Embodiment]:
        """Return the live registry plus one embodiment no surface names."""
        planted = dict(EMBODIMENTS)
        planted["zzz_planted_embodiment"] = Cosmos3Embodiment(
            name="zzz_planted_embodiment",
            domain_name="zzz_planted_domain",
            raw_action_dim=10,
            action_chunk_size=16,
            fps=15,
        )
        return planted

    def test_front_matter_grader_reports_it(self) -> None:
        assert "zzz_planted_embodiment" in _front_matter_gap(_page_text(), self._planted())

    def test_embodiment_table_grader_reports_it(self) -> None:
        missing, _ = _embodiment_table_gap(_page_text(), self._planted())
        assert "zzz_planted_embodiment" in missing

    def test_inline_enum_grader_reports_it(self) -> None:
        missing, _ = _inline_enum_gap(_page_text(), self._planted())
        assert "zzz_planted_embodiment" in missing

    def test_domain_table_grader_reports_it(self) -> None:
        missing, _ = _domain_table_gap(_page_text(), self._planted())
        assert "zzz_planted_embodiment" in missing

    def test_readme_grader_reports_it(self) -> None:
        assert "zzz_planted_embodiment" in _readme_gap(_readme_text(), self._planted())

    def test_quickstart_grader_reports_it(self) -> None:
        assert "zzz_planted_embodiment" in _quickstart_gap(_readme_text(), self._planted())

    def test_package_docstring_grader_reports_it(self) -> None:
        assert "zzz_planted_embodiment" in _package_docstring_gap(cosmos3_package.__doc__ or "", self._planted())

    def test_policy_args_grader_reports_it(self) -> None:
        assert "zzz_planted_embodiment" in _policy_args_gap(Cosmos3Policy.__doc__ or "", self._planted())

    def test_count_sentence_grader_reports_a_contradicted_count(self) -> None:
        """The planted domain ships no stats, so the unbundled count grows by one."""
        problems = _count_sentence_problems(_page_text(), self._planted())
        assert problems, "the count sentence grader did not notice an extra unbundled embodiment"
        assert any("unbundled" in p for p in problems), problems

    def test_domain_table_fact_grader_reports_a_wrong_stated_fact(self) -> None:
        """A row whose stated width disagrees with its entry must be reported."""
        page = _page_text().replace("| `av` | `av` | 9 | no |", "| `av` | `av` | 7 | no |")
        assert page != _page_text(), "the av row this test rewrites is no longer in the page"
        problems = _domain_table_facts(page, EMBODIMENTS)
        assert any("raw dim" in p and p.startswith("av:") for p in problems), problems

    def test_quickstart_fact_grader_reports_a_wrong_stated_fact(self) -> None:
        """A stated chunk size the entry contradicts must be reported."""
        readme = _readme_text().replace("chunk 32", "chunk 31")
        assert readme != _readme_text(), "the quickstart's 'chunk 32' fact is no longer in the README"
        problems = _quickstart_facts(readme, EMBODIMENTS)
        assert any("chunk" in p and p.startswith("droid:") for p in problems), problems

    def test_extra_direction_reports_an_unregistered_name(self) -> None:
        """A documented embodiment the registry drops must be reported too."""
        trimmed = {k: v for k, v in EMBODIMENTS.items() if k != "av"}
        _, extra = _embodiment_table_gap(_page_text(), trimmed)
        assert "av" in extra
        _, extra_domain = _domain_table_gap(_page_text(), trimmed)
        assert "av" in extra_domain


def test_a_missing_surface_is_reported_rather_than_skipped() -> None:
    """A moved page or renamed table must raise, never grade an empty set."""
    assert _PAGE.is_file(), _PAGE
    assert _README.is_file(), _README
    assert _table("| a | b |\n|---|---|\n| 1 | 2 |\n", ["provider", "backend", "notes"]) is None
    with pytest.raises(AssertionError):
        _readme_gap("no table here", EMBODIMENTS)
    with pytest.raises(AssertionError):
        _embodiment_table_gap("no table here", EMBODIMENTS)
    with pytest.raises(AssertionError):
        _quickstart_paragraph("no such paragraph here")
    with pytest.raises(AssertionError):
        _package_docstring_gap("no 'Available embodiments' sentence here", EMBODIMENTS)
    with pytest.raises(AssertionError):
        _policy_args_gap("no embodiment Args entry here", EMBODIMENTS)
