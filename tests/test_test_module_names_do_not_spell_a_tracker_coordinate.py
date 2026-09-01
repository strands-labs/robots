"""Regression: a test module's name must name the behaviour, not a tracker item.

A test module name is the whole diagnosis a reader gets from a failure line, and
it is the index a maintainer scans when asking "is this behaviour already
graded?".  A name that encodes the issue or pull request that birthed the file
answers neither question: the reader still has to open the file to learn what
broke, and the coordinate ages out of usefulness the moment the item is closed.

The shape this refuses is a *tracker coordinate written with an underscore where
the ``#`` would be*.  ``..._for_harness_361`` is ``harness#361`` spelled for a
filesystem, and it is unresolvable for exactly the reason
:mod:`test_source_strings_resolve_their_issue_references` documents for the
operator-facing string it grades: an unowned slug names a repository with no
owner, so the reader has no coordinate to open.  That module owns the predicate
and this one reuses it, so the two surfaces cannot drift to two definitions of
"resolvable".

The rule needs no exemption list, which is what makes it derivable rather than
curated.  A vendor model or version number is spelled *attached* to the word it
qualifies -- ``so101``, ``gr00t``, ``n17``, ``cosmos3``, ``go2``, ``ipv6`` --
and every one of those is a single token with no underscore before its digits,
so none of them matches.  A bare run of digits standing as its own token is
either a tracker coordinate or a magic number, and neither is a behaviour.

It would have failed while ``tests/drivers/`` carried
``test_g1_send_action_success_is_the_acceptance_criterion_for_harness_361.py``,
whose behaviour -- ``send_action`` reaching the wire on a healthy, fully wired
driver -- is now what its name says.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.test_source_strings_resolve_their_issue_references import _unresolvable_references

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The two test trees a merge gates on.
_TEST_ROOTS = ("tests", "tests_integ")

#: A word, an underscore, then a run of digits standing as its own token: the
#: filesystem spelling of ``<slug>#<number>``.  ``so101`` and ``gr00t`` do not
#: match (no underscore before the digits) and neither does ``2fa`` (the digits
#: are not the whole token).
_NAME_COORDINATE = re.compile(r"([A-Za-z][A-Za-z0-9]*)_(\d+)(?=_|$)")


def _test_modules() -> list[Path]:
    """Every Python module under the test trees, ``__pycache__`` excluded."""
    return sorted(
        path for root in _TEST_ROOTS for path in (_REPO_ROOT / root).rglob("*.py") if "__pycache__" not in path.parts
    )


def _coordinates_in_name(stem: str) -> list[str]:
    """Tracker coordinates ``stem`` spells, rendered as ``<slug>#<number>``.

    Args:
        stem: A module name without its ``.py`` suffix.

    Returns:
        The unresolvable coordinates the name spells, in the same
        ``<slug>#<number>`` form :func:`_unresolvable_references` grades, so
        both surfaces answer to one definition of "resolvable".
    """
    return [
        reference
        for word, number in _NAME_COORDINATE.findall(stem)
        for reference in _unresolvable_references(f"{word}#{number}")
    ]


def test_test_modules_discovered() -> None:
    """Guard: the scan walked both test trees, not one subtree."""
    modules = _test_modules()
    assert len(modules) > 1000, f"only {len(modules)} test modules found; the scan is too narrow"
    roots = {path.relative_to(_REPO_ROOT).parts[0] for path in modules}
    assert set(_TEST_ROOTS) <= roots, roots


def test_no_test_module_name_spells_a_tracker_coordinate() -> None:
    """A test module name must state a behaviour, not a tracker coordinate."""
    offenders = [
        f"{path.relative_to(_REPO_ROOT)}: {', '.join(coordinates)}"
        for path in _test_modules()
        if (coordinates := _coordinates_in_name(path.stem))
    ]
    assert not offenders, (
        "A test module name encodes a tracker coordinate rather than the "
        "behaviour it grades. Rename the file after what it verifies - the name "
        "is the whole diagnosis a reader gets from a failure line, and the "
        "coordinate resolves to nothing:\n" + "\n".join(offenders)
    )


def test_an_underscored_coordinate_in_a_name_is_flagged() -> None:
    """The predicate flags the shape this guard exists to keep out."""
    assert _coordinates_in_name("test_g1_send_action_is_the_criterion_for_harness_361") == ["harness#361"]
    assert _coordinates_in_name("test_mesh_acl_scoping_issue_2935") == ["issue#2935"]


def test_a_name_cannot_spell_an_owner_which_is_why_the_rule_is_absolute() -> None:
    """A module name has no ``/``, so every coordinate it spells is unowned.

    The sibling's predicate accepts an owner-qualified coordinate as resolvable.
    A module name cannot produce one, because the separator that completes a
    slug is not a path token -- which is why this guard admits no spelling of a
    coordinate in a name where the string guard admits two.
    """
    assert _unresolvable_references("strands-labs/robots#2765") == []
    assert "/" not in Path("test_g1_criterion_for_harness_361.py").stem
    assert _coordinates_in_name("test_strands_labs_robots_2765") == ["robots#2765"]


def test_a_model_or_version_number_in_a_name_is_not_flagged() -> None:
    """Domain numbers must pass, or the rule is blanket strictness about digits."""
    accepted = [
        "test_examples_so101_pick_lifts",  # robot model
        "test_gr00t_container_hardening",  # policy family
        "test_n17_live_server",  # policy version
        "test_cosmos3_lifecycle",  # policy provider
        "test_go2_driver",  # robot model
        "test_server_address_port_ipv6",  # protocol version
        "test_dashboard_auth_2fa_ceremony",  # digits are not the whole token
    ]
    for stem in accepted:
        assert _coordinates_in_name(stem) == [], f"should not be flagged: {stem!r}"


def test_the_predicate_separates_the_two_shapes() -> None:
    """Non-vacuity: the predicate answers both ways over the exemplars."""
    outcomes = {bool(_coordinates_in_name(stem)) for stem in ("test_a_for_harness_361", "test_examples_so101_pick")}
    assert outcomes == {True, False}
