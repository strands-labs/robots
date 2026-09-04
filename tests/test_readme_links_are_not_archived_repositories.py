"""Repo hygiene: no README link steers a reader at an archived repository.

The README is the front door. Its header nav is a short list of "go here next"
destinations, and until #3192 one of them was ``strands-labs/robots-sim``, which
was archived at 01:33 UTC on 2026-08-06 and is read-only. A newcomer following
it lands on a repository presented as a live sibling that accepts no issue, no
pull request and no push, while the maintained simulation stack it implies is
elsewhere is in fact *here* (``strands_robots/simulation/``, MuJoCo + Isaac
backends).

Nothing graded that, and the two checks a reader would expect to are both blind
to it by construction:

* ``tests/test_markdown_links_resolve.py`` resolves *relative* targets and says
  so - ``http``/``mailto`` and other schemed targets "name nothing in the tree"
  and are deliberately out of its scope. The nav entry was also raw ``<a href>``
  HTML rather than Markdown link syntax, so it was outside that module twice
  over.
* ``mkdocs build --strict`` reads only files under ``docs_dir``, so ``README.md``
  is never read by it at all.

And an archived repository is invisible to the usual reads: a link to one is a
200, its issues and pull requests still list normally, and ``viewerPermission``
keeps reporting the permission actually granted. ``isArchived`` is the only field
that answers the question, which is why this is a curated fact below rather than
something a link checker could have inferred.

**Scoped to ``README.md``, and the scope is load-bearing.** A tree-wide ban on
the ``robots-sim`` URL would be wrong, because this repository deliberately
carries references to that repository as *provenance*:
``strands_robots/simulation/isaac/`` was absorbed from it by #1156, and
``tests/simulation/isaac/test_migrated_reference_provenance.py`` exists to
*require* those references be written ``robots-sim#N`` so they name the
repository they belong to. Changelog fragments record the same history. Those
citations are correct and must stay. The distinction that matters is between
"this arrived from there" (history, correct) and "go there for the live thing"
(a destination, wrong once the target is read-only), and that distinction is not
decidable from a URL - so this module grades the one file that is unambiguously
a list of destinations and carries no provenance references at all. Measured on
the fix commit: ``README.md`` holds 14 distinct ``github.com`` URLs naming 9
repositories, and not one of them is a ``robots-sim#N``-style citation.

**The archived map is a floor, not an oracle.** No offline check can discover
that a repository was archived yesterday, and this suite does not reach the
network. What the map does guarantee is that a target already known to be
read-only cannot reappear in the README - which is the regression this pins -
and that the population it is checked against is *derived* from the README, so a
nav entry or prose link added later is graded on arrival rather than needing this
file to be edited.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_README = _REPO_ROOT / "README.md"

# Repositories this project knows to be read-only, each with the date measured
# from the GitHub API (`repository(...) { isArchived }`) and the reason it is
# terminal. AGENTS.md records the robots-sim archival and its consequence: an
# archived repository is terminal, so it is closed for good and anything still
# naming it as a destination has to be satisfied or closed on this side.
_ARCHIVED_REPOSITORIES: dict[str, str] = {
    "strands-labs/robots-sim": (
        "archived 2026-08-06T01:33Z; the simulation stack it hosted was absorbed "
        "into strands_robots/simulation/ by #1156"
    ),
}

# github.com paths whose first segment is a site route rather than an account.
# `https://github.com/orgs/strands-labs/projects/2` is in the README nav and
# would otherwise decode as the repository `orgs/strands-labs`.
_RESERVED_OWNERS = frozenset(
    {
        "about",
        "apps",
        "codespaces",
        "collections",
        "enterprise",
        "explore",
        "features",
        "join",
        "login",
        "marketplace",
        "new",
        "notifications",
        "orgs",
        "pricing",
        "search",
        "settings",
        "sponsors",
        "topics",
    }
)

# Every spelling of a github.com URL at once - Markdown `[text](url)`, an
# autolink `<url>`, an HTML `href="url"` and a bare URL in prose all reduce to
# the same characters. Matching the URL rather than the link syntax is why the
# raw `<a href>` nav entry this module exists for is in the population.
# Terminators exclude the delimiters those four spellings close with.
_GITHUB_URL = re.compile(r"https?://(?:www\.)?github\.com/([^\s\"'<>)\]]+)", re.IGNORECASE)


def _github_repository(url: str) -> str | None:
    """Return the lowercase ``owner/repo`` a github.com URL names, if any.

    Args:
        url: A URL of any scheme. Anything that is not a github.com URL naming a
            repository returns ``None``.

    Returns:
        ``owner/repo`` lowercased, or ``None`` for a non-github.com URL, a URL
        naming only an account, or a site route such as ``/orgs/...``.
    """
    match = _GITHUB_URL.match(url)
    if match is None:
        return None

    segments = [segment for segment in match.group(1).split("/") if segment]
    if len(segments) < 2:
        return None

    owner, repo = segments[0], segments[1]
    if owner.lower() in _RESERVED_OWNERS:
        return None

    # Trailing sentence punctuation first, so a clone URL's `.git` is still the
    # suffix when it is stripped: `VERA.git` -> `VERA`, `robots-sim.` -> `robots-sim`.
    repo = repo.rstrip(".,;:!?").removesuffix(".git")
    if not repo:
        return None

    return f"{owner.lower()}/{repo.lower()}"


def _readme_repositories() -> dict[str, int]:
    """Return every repository the README links, mapped to its 1-based line number.

    Returns:
        A mapping of ``owner/repo`` to the first README line naming it.
    """
    found: dict[str, int] = {}
    for number, line in enumerate(_README.read_text(encoding="utf-8").splitlines(), start=1):
        for url in _GITHUB_URL.finditer(line):
            repository = _github_repository(url.group(0))
            if repository is not None:
                found.setdefault(repository, number)
    return found


def test_no_readme_link_names_an_archived_repository() -> None:
    """The README does not offer a read-only repository as a destination."""
    offenders = [
        f"README.md:{line} links {repository} ({_ARCHIVED_REPOSITORIES[repository]})"
        for repository, line in sorted(_readme_repositories().items(), key=lambda item: item[1])
        if repository in _ARCHIVED_REPOSITORIES
    ]
    assert not offenders, (
        "README.md steers a reader at an archived repository, which accepts no issue, "
        "pull request or push:\n  " + "\n  ".join(offenders)
    )


def test_the_sweep_reads_the_readmes_real_links() -> None:
    """The population is non-empty and holds the README's actual destinations.

    Guards the vacuous pass. A regex that stopped matching, or a normaliser that
    returned ``None`` for everything, would leave the pin above green while
    grading nothing at all.
    """
    found = _readme_repositories()

    # A floor rather than the exact count, so removing a link is not a failure.
    # The README names 9 repositories across 14 github.com URLs as of #3192.
    assert len(found) >= 8, f"the sweep lost most of the README's links, found {len(found)}: {sorted(found)}"
    for expected in ("strands-labs/robots", "huggingface/lerobot", "google-deepmind/mujoco"):
        assert expected in found, f"the sweep lost a known README link: {expected}"


def test_the_archived_map_states_a_repository_and_a_reason() -> None:
    """Each archived entry is ``owner/repo`` lowercased and carries its reasoning.

    The map is the whole oracle, so an entry that is misspelled matches no link
    and the guard passes while grading nothing.
    """
    assert _ARCHIVED_REPOSITORIES, "the archived map is the oracle; emptying it makes the pin vacuous"

    for repository, reason in _ARCHIVED_REPOSITORIES.items():
        assert repository == repository.lower(), f"{repository} must be lowercased to match a normalised link"
        assert repository.count("/") == 1, f"{repository} must be owner/repo"
        assert _github_repository(f"https://github.com/{repository}") == repository, (
            f"{repository} is not what the normaliser produces for its own URL"
        )
        assert len(reason) > 20, f"{repository} needs a reason a reader can act on, got {reason!r}"


class TestTheRepositoryExtractor:
    """The normaliser's edge cases, each present in this README or adjacent to it."""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://github.com/strands-labs/robots-sim", "strands-labs/robots-sim"),
            # A deep path still names the repository it is inside.
            ("https://github.com/strands-labs/robots/blob/main/LICENSE", "strands-labs/robots"),
            ("https://github.com/strands-labs/robots/issues/2062", "strands-labs/robots"),
            # A clone URL, as the README's VERA link is written.
            ("https://github.com/sizhe-li/VERA.git", "sizhe-li/vera"),
            # GitHub account and repository names are case-insensitive.
            ("https://github.com/NVIDIA/Isaac-GR00T", "nvidia/isaac-gr00t"),
            ("https://GitHub.com/Strands-Labs/Robots-Sim", "strands-labs/robots-sim"),
            ("http://github.com/strands-labs/robots-sim", "strands-labs/robots-sim"),
            ("https://www.github.com/strands-labs/robots-sim", "strands-labs/robots-sim"),
            ("https://github.com/strands-labs/robots-sim/", "strands-labs/robots-sim"),
            # Trailing sentence punctuation is not part of the name.
            ("https://github.com/strands-labs/robots-sim.", "strands-labs/robots-sim"),
            # A site route, not an account: the README nav links the project board.
            ("https://github.com/orgs/strands-labs/projects/2", None),
            # An account with no repository.
            ("https://github.com/strands-labs", None),
            # Not github.com. The README's badges are shields.io URLs that embed
            # a repository path, and they are not destinations.
            ("https://img.shields.io/github/stars/strands-labs/robots-sim", None),
            ("https://strandsagents.com/", None),
        ],
    )
    def test_a_url_reduces_to_the_repository_it_names(self, url: str, expected: str | None) -> None:
        """Each spelling decodes to the repository a reader would land on."""
        assert _github_repository(url) == expected

    @pytest.mark.parametrize(
        "spelling",
        [
            '<a href="https://github.com/strands-labs/robots-sim">Robots Sim</a>',
            "[Robots Sim](https://github.com/strands-labs/robots-sim)",
            "<https://github.com/strands-labs/robots-sim>",
            "see https://github.com/strands-labs/robots-sim for the sim stack",
        ],
    )
    def test_every_link_spelling_is_in_the_population(self, spelling: str) -> None:
        """The sweep matches the URL, so link syntax cannot hide a destination.

        The nav entry #3192 removed was raw HTML. Markdown-only link parsing is
        exactly how it stayed ungraded, so all four spellings are pinned.
        """
        match = _GITHUB_URL.search(spelling)
        assert match is not None, f"no github.com URL found in {spelling!r}"
        assert _github_repository(match.group(0)) == "strands-labs/robots-sim"
