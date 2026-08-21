"""A test's name describes the behaviour it verifies, not the release that added it.

A test is read far more often than it is written, and the first thing a
maintainer sees is its name. ``TestHardwareConfigV040Followups`` tells the
reader which review round produced the class and nothing about what it
checks, so finding the test that covers a behaviour means opening every
plausible file; and once the release is long past, the name actively
misleads - it reads as "historical", which is an invitation to skip it.

Two tokens are refused, both of which name the *provenance* of a test rather
than its subject:

* **A release token** - three or more digits after a ``v`` (``V040`` is
  ``0.4.0`` with the dots dropped). Two digits or fewer are left alone,
  because a *data format* version genuinely is the behaviour under test:
  ``test_load_v3_parses_every_field`` (the ``.spz`` v3 layout),
  ``test_a_v21_root_is_refused_and_pointed_at_lerobots_converter`` (a
  LeRobot v2.1 dataset), ``test_current_transform_version_is_v4``. Those
  names must keep working, and are pinned below.
* **A review-round token** - ``Followup`` / ``Followups``. A bundle named
  for its review round is also usually a bundle of unrelated checks, which
  is the other half of the problem: no single behaviour name fits, so the
  fix is to split it rather than to rename it.

Provenance is not lost by this rule, it moves to where a reader wants it:
the docstring. Every test renamed for this guard kept its ``#NNN:`` issue
reference in its own docstring.

This encodes what the tree already does - at the commit that added this
guard, 2 of 3294 test classes and 0 of 17508 test functions named a
release.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TEST_TREES = ("tests", "tests_integ")

#: Three-or-more-digit version token: a release triple with the dots dropped.
#: A preceding letter is allowed on purpose - the offender this guard was
#: written for spelled it ``...CaV040Followups`` in CamelCase.
_RELEASE_TOKEN = re.compile(r"[Vv]\d{3,}")

#: A name for the review round that produced the test, not for its subject.
_REVIEW_ROUND_TOKEN = re.compile(r"[Ff]ollowups?")

#: A clean sweep would prove nothing if the scan stopped reaching the trees.
_MINIMUM_SCANNED_NAMES = 15_000


def _test_case_names() -> list[tuple[pathlib.Path, int, str]]:
    """Every ``class Test*`` and ``def test_*`` name in both test trees."""
    found: list[tuple[pathlib.Path, int, str]] = []
    for tree in _TEST_TREES:
        for path in sorted((_REPO_ROOT / tree).rglob("*.py")):
            try:
                module = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - a broken test file fails elsewhere
                continue
            for node in ast.walk(module):
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                    found.append((path, node.lineno, node.name))
                elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test_"):
                    found.append((path, node.lineno, node.name))
    return found


def _provenance_named(names: list[tuple[pathlib.Path, int, str]]) -> list[str]:
    """The offending names, formatted with the token that disqualified each."""
    offenders: list[str] = []
    for path, lineno, name in names:
        reasons = []
        if match := _RELEASE_TOKEN.search(name):
            reasons.append(f"release token {match.group(0)!r}")
        if match := _REVIEW_ROUND_TOKEN.search(name):
            reasons.append(f"review-round token {match.group(0)!r}")
        if reasons:
            rel = path.relative_to(_REPO_ROOT)
            offenders.append(f"{rel}:{lineno} {name} ({', '.join(reasons)})")
    return offenders


def test_no_test_case_is_named_for_a_release_or_a_review_round() -> None:
    """No test class or function names the release or review round that added it."""
    names = _test_case_names()
    assert len(names) >= _MINIMUM_SCANNED_NAMES, (
        f"only {len(names)} test-case names reached the scan (expected at least "
        f"{_MINIMUM_SCANNED_NAMES}); the discovery has gone blind, so a clean "
        "result would prove nothing"
    )
    offenders = _provenance_named(names)
    assert not offenders, (
        "test case(s) named for their provenance rather than the behaviour they verify:\n  "
        + "\n  ".join(offenders)
        + "\nName the behaviour and keep the issue reference in the docstring."
    )


@pytest.mark.parametrize(
    "name",
    [
        # Real names in the tree at the commit that added this guard: each
        # states the version of a *data format* it parses, which is the
        # behaviour under test.
        "test_load_v3_parses_every_field",
        "test_load_v2_takes_the_three_byte_rotation_path",
        "test_decode_rotations_v3_roundtrips_quaternions",
        "test_a_v21_root_is_refused_and_pointed_at_lerobots_converter",
        "test_current_transform_version_is_v4",
        "test_non_v3_video_path_template_returns_report_not_keyerror",
        # The class form of the same allowance - no class in the tree carries a
        # format-version token today, so this pins that the rule is about the
        # token, not about which node kind it sits on.
        "TestSpzV3Layout",
    ],
)
def test_a_data_format_version_stays_accepted(name: str) -> None:
    """One- and two-digit version tokens name a format, not a release."""
    assert not _provenance_named([(_REPO_ROOT / "tests" / "x.py", 1, name)]), (
        f"{name!r} names the version of a data format it parses, which is the "
        "behaviour under test; only a release triple is refused"
    )


@pytest.mark.parametrize(
    ("name", "expected_token"),
    [
        ("TestHardwareConfigV040Followups", "release token 'V040'"),
        ("TestEnsureCaV040Followups", "release token 'V040'"),
        ("TestSomethingV051", "release token 'V051'"),
        ("test_something_v0410_regression", "release token 'v0410'"),
        ("TestReviewFollowups", "review-round token 'Followups'"),
        ("test_pr_review_followup_cases", "review-round token 'followup'"),
    ],
)
def test_a_provenance_named_case_is_reported(name: str, expected_token: str) -> None:
    """Both refused tokens are detected, so a clean sweep is a real result."""
    offenders = _provenance_named([(_REPO_ROOT / "tests" / "x.py", 1, name)])
    assert offenders, f"{name!r} names its provenance and must be reported"
    assert expected_token in offenders[0], f"expected {expected_token!r} in {offenders[0]!r}"


def test_the_scan_reaches_both_test_trees() -> None:
    """A scan that stopped covering ``tests_integ`` would report clean falsely."""
    reached = {path.relative_to(_REPO_ROOT).parts[0] for path, _lineno, _name in _test_case_names()}
    assert set(_TEST_TREES) <= reached, f"scan reached only {sorted(reached)}"
