"""An empty robot-name selection is refused, never widened to every robot.

``download_robots(names=...)`` and the ``download_assets`` tool's ``robots=``
string both carry a SUBSET of the sim robots the registry lists. ``None`` means
"all of them"; an empty selection means *none*, which is the opposite answer
rather than a spelling of the same one.

Read by truthiness, ``names=[]`` fell through to the branches that never read it:
measured on the shipped registry it downloaded 56 robots on its own, and 13 - the
whole ``humanoid`` category - when a ``category`` was also passed, reporting
either as the caller's own request. Nothing has to write ``[]`` to reach that: the
tool's own ``if r.strip()`` filter turns the non-empty ``robots=","`` into zero
names.

The controls matter as much as the refusals here. This surface resolves each name
by membership into a dict, so a repeat, a mapping and a one-shot iterator are all
honored as written today - which is why the shape is deliberately NOT routed
through the shared ``name_list_error`` domain, and why those spellings are pinned
below rather than left to a later sweep to "tidy up".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from strands_robots.assets import download as dl
from strands_robots.tools.download_assets import download_assets

_MOD = "strands_robots.assets.download"
_TOOL_MOD = "strands_robots.tools.download_assets"


def _entry(name: str, category: str = "arm") -> dict[str, Any]:
    return {"asset": {"dir": name, "model_xml": "model.xml"}, "category": category}


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, dict[str, Any]]:
    """A two-category sim registry with the download side effects stubbed out."""
    entries = {
        "arm1": _entry("arm1", "arm"),
        "arm2": _entry("arm2", "arm"),
        "hum1": _entry("hum1", "humanoid"),
    }
    monkeypatch.setattr(f"{_MOD}.get_user_assets_dir", lambda: tmp_path)
    monkeypatch.setattr(f"{_MOD}.registry_list_robots", lambda mode: [{"name": n} for n in entries])
    monkeypatch.setattr(f"{_MOD}.get_robot", lambda n: entries.get(n))
    monkeypatch.setattr(f"{_MOD}.resolve_robot_name", lambda n: n)
    # Nothing is present, so every selected robot reaches the download partition -
    # the count a widened selection would inflate.
    monkeypatch.setattr(f"{_MOD}._needs_download", lambda *a, **k: True)
    monkeypatch.setattr(f"{_MOD}._robot_descriptions_available", lambda: False)
    return entries


def _considered(**kwargs: Any) -> set[str]:
    """The set of robots a call actually hands to the download partition."""
    seen: set[str] = set()

    def _git(robots: dict[str, Any], dest_dir: Path) -> dict[str, str]:
        seen.update(robots)
        return {n: "downloaded" for n in robots}

    with (
        patch(f"{_MOD}._download_via_git", side_effect=_git),
        patch(f"{_MOD}._download_from_github", return_value="downloaded"),
    ):
        dl.download_robots(**kwargs)
    return seen


# The refusals


def test_empty_name_selection_is_refused(registry: dict[str, Any]) -> None:
    """``names=[]`` selects no robot, so it is refused rather than read as "all"."""
    with pytest.raises(ValueError) as excinfo:
        dl.download_robots(names=[])
    message = str(excinfo.value)
    assert "selects no robot" in message
    # The remedy names both honorable alternatives, so it is actionable as written.
    assert "names=None" in message


def test_empty_name_selection_is_refused_and_not_read_as_the_category(registry: dict[str, Any]) -> None:
    """``names=[]`` with a ``category`` is refused, not silently widened to it.

    This is the second widening branch and the quieter one: the empty selection
    fell through to the ``category`` filter, so the call downloaded an entire
    category while reporting it as the caller's own named request.
    """
    with pytest.raises(ValueError):
        dl.download_robots(names=[], category="humanoid")


def test_refused_selection_downloads_nothing(registry: dict[str, Any]) -> None:
    """The refusal lands before any download is attempted."""
    with (
        patch(f"{_MOD}._download_via_git") as git,
        patch(f"{_MOD}._download_from_github") as github,
    ):
        with pytest.raises(ValueError):
            dl.download_robots(names=[])
    git.assert_not_called()
    github.assert_not_called()


def test_refused_selection_does_not_create_the_asset_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The refusal precedes ``get_user_assets_dir()``, which creates the cache dir.

    Checked with the real directory helper rather than a stub: it is the call that
    performs the ``mkdir``, so a guard placed after it would leave the directory
    behind for a call that never downloaded anything.
    """
    cache = tmp_path / "assets-cache"
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(cache))
    assert not cache.exists()

    with pytest.raises(ValueError):
        dl.download_robots(names=[])

    assert not cache.exists(), "a refused selection created the asset cache directory"


# The controls: every honored spelling is unchanged


def test_none_selects_every_sim_robot(registry: dict[str, Any]) -> None:
    assert _considered() == {"arm1", "arm2", "hum1"}


def test_none_with_a_category_selects_that_category(registry: dict[str, Any]) -> None:
    assert _considered(category="humanoid") == {"hum1"}


def test_a_real_subset_selects_exactly_that_subset(registry: dict[str, Any]) -> None:
    assert _considered(names=["arm2"]) == {"arm2"}


@pytest.mark.parametrize(
    "names",
    [
        pytest.param(["arm2", "arm2"], id="repeat"),
        pytest.param({"arm2": 1}, id="mapping"),
        pytest.param(iter(["arm2"]), id="one-shot-iterator"),
    ],
)
def test_membership_resolution_keeps_tolerating_its_own_spellings(registry: dict[str, Any], names: Any) -> None:
    """Spellings the shared name-list domain refuses are honored here, by design.

    ``download_robots`` resolves each name by membership into a dict, so a repeat
    collapses to its first occurrence, a mapping is read over its keys and a
    one-shot iterator is consumed exactly once. Routing this selector through
    ``name_list_error`` would reject all three, which is why only the emptiness
    verdict is taken locally. Pinned so that choice is deliberate rather than
    incidental.
    """
    assert _considered(names=names) == {"arm2"}


# The tool surface refuses in its own vocabulary


@pytest.mark.parametrize("robots", [",", " ", ",,,", " , ", ", ,"])
def test_tool_refuses_a_non_empty_value_that_names_no_robot(robots: str) -> None:
    """A ``robots=`` string can carry content while naming nothing.

    Every field is dropped by the tool's own blank filter, so the parse yields an
    empty selection from a non-empty argument.
    """
    with patch(f"{_TOOL_MOD}.download_robots") as download:
        result = download_assets(action="download", robots=robots)

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    # Refused in this surface's vocabulary: the caller passed ``robots=``, not
    # ``names=``, so that is the argument the remedy has to name.
    assert "robots=" in text
    assert "names=" not in text
    download.assert_not_called()


@pytest.mark.parametrize(
    ("robots", "expected"),
    [
        pytest.param(None, None, id="omitted-means-all"),
        pytest.param("", None, id="empty-string-means-all"),
        pytest.param("so100", ["so100"], id="one-name"),
        pytest.param("so100,panda", ["so100", "panda"], id="two-names"),
        pytest.param(" so100 , panda ", ["so100", "panda"], id="whitespace-stripped"),
        pytest.param("so100,,panda", ["so100", "panda"], id="blank-field-dropped"),
    ],
)
def test_tool_forwards_every_honored_value_unchanged(robots: str | None, expected: list[str] | None) -> None:
    """An absent or empty ``robots=`` still means "all"; a named subset is forwarded.

    For a single string argument, unset and empty genuinely coincide - only a value
    that carries content while naming nothing is a caller mistake.
    """
    fake = {"downloaded": 0, "skipped": 0, "failed": 0, "method": "git clone"}
    with patch(f"{_TOOL_MOD}.download_robots", return_value=fake) as download:
        result = download_assets(action="download", robots=robots)

    assert result["status"] == "success"
    assert download.call_args.kwargs["names"] == expected
