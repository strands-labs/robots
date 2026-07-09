"""Behavior tests for the ``download_assets`` agent tool.

Covers the four documented actions (``list``, ``status``, ``download``,
unknown) plus the error path. The tool is a thin wrapper around
:mod:`strands_robots.assets.download`; the underlying download logic is
mocked so these tests run hardware- and network-free, asserting on the
tool's parsing/formatting contract rather than implementation details.
"""

from __future__ import annotations

from unittest.mock import patch

from strands_robots.tools.download_assets import download_assets

_MOD = "strands_robots.tools.download_assets"


def test_list_action_returns_robot_table() -> None:
    """``action='list'`` returns the formatted registry table."""
    with patch(f"{_MOD}.format_robot_table", return_value="ROBOT-TABLE"):
        result = download_assets(action="list")
    assert result["status"] == "success"
    assert "ROBOT-TABLE" in result["content"][0]["text"]


def test_status_action_summarizes_availability() -> None:
    """``action='status'`` counts available vs missing and shows the cache dir."""
    robots_info = [
        {"name": "so100", "category": "arm", "description": "arm", "available": True},
        {"name": "panda", "category": "arm", "description": "arm", "available": False},
    ]
    with (
        patch(f"{_MOD}.list_available_robots", return_value=robots_info),
        patch(f"{_MOD}.get_user_assets_dir", return_value="/tmp/assets"),
    ):
        result = download_assets(action="status")
    text = result["content"][0]["text"]
    assert result["status"] == "success"
    assert "1 available, 1 missing" in text
    assert "/tmp/assets" in text
    assert "so100" in text and "panda" in text


def test_status_action_marks_each_row_available_or_missing() -> None:
    """Each robot row carries a per-row availability marker.

    Regression: the marker was an empty-both ``'' if r['available'] else ''``
    ternary (an emoji-stripping fossil), so downloaded and missing robots
    rendered identically and the per-row status conveyed nothing. Assert the
    marker is present and correctly maps availability to the robot on its row.
    """
    robots_info = [
        {"name": "so100", "category": "arm", "description": "arm", "available": True},
        {"name": "panda", "category": "arm", "description": "arm", "available": False},
    ]
    with (
        patch(f"{_MOD}.list_available_robots", return_value=robots_info),
        patch(f"{_MOD}.get_user_assets_dir", return_value="/tmp/assets"),
    ):
        result = download_assets(action="status")
    rows = {
        name: next(ln for ln in result["content"][0]["text"].splitlines() if name in ln) for name in ("so100", "panda")
    }
    # Available and missing rows are visually distinguishable.
    assert rows["so100"] != rows["panda"].replace("panda", "so100")
    # ...and the marker maps to the correct robot.
    assert "[ok]" in rows["so100"] and "[--]" not in rows["so100"]
    assert "[--]" in rows["panda"] and "[ok]" not in rows["panda"]


def test_download_action_parses_names_and_reports_counts() -> None:
    """``action='download'`` splits comma names and surfaces result counts."""
    fake_result = {
        "downloaded": 2,
        "skipped": 1,
        "failed": 0,
        "method": "robot_descriptions",
        "assets_dir": "/tmp/assets",
    }
    with patch(f"{_MOD}.download_robots", return_value=fake_result) as mock_dl:
        result = download_assets(action="download", robots="so100, panda ", category="arm", force=True)
    mock_dl.assert_called_once_with(names=["so100", "panda"], category="arm", force=True)
    text = result["content"][0]["text"]
    assert result["status"] == "success"
    assert "Downloaded: 2, Skipped: 1, Failed: 0" in text
    assert "robot_descriptions" in text


def test_download_action_with_no_names_passes_none() -> None:
    """Omitting ``robots`` downloads all (names=None)."""
    fake_result = {"downloaded": 0, "skipped": 0, "failed": 0, "method": "git", "assets_dir": "/d"}
    with patch(f"{_MOD}.download_robots", return_value=fake_result) as mock_dl:
        download_assets(action="download")
    mock_dl.assert_called_once_with(names=None, category=None, force=False)


def test_download_action_lists_failed_details() -> None:
    """Failed downloads are itemized in the output."""
    fake_result = {
        "downloaded": 0,
        "skipped": 0,
        "failed": 1,
        "method": "git",
        "assets_dir": "/d",
        "failed_details": {"badbot": "clone failed"},
    }
    with patch(f"{_MOD}.download_robots", return_value=fake_result):
        result = download_assets(action="download", robots="badbot")
    text = result["content"][0]["text"]
    assert "badbot" in text and "clone failed" in text


def test_unknown_action_returns_error() -> None:
    """An unrecognized action is rejected with the valid-action list."""
    result = download_assets(action="bogus")
    assert result["status"] == "error"
    assert "Unknown action" in result["content"][0]["text"]


def test_underlying_exception_is_caught_and_reported() -> None:
    """Exceptions from the download layer become a structured error result."""
    with patch(f"{_MOD}.download_robots", side_effect=RuntimeError("boom")):
        result = download_assets(action="download", robots="so100")
    assert result["status"] == "error"
    assert "boom" in result["content"][0]["text"]
