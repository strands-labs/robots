"""Both of the tracker's observation-key ladders read their prefixed spelling.

:class:`~strands_robots.policies.protomotions.ProtoMotionsPolicy` resolves the
two frame signals its ONNX graph needs -- the anchor-body rotation and the root
angular velocity -- through a fallback ladder that tries several observation
keys in turn. Every such ladder in ``strands_robots`` pairs a bare key with
``observation.<that key>``: the root-angular-velocity ladder reads
``("root_ang_vel_local", "observation.root_ang_vel_local")``, and both of
``WBCPolicy``'s read ``("base_ang_vel", "observation.base_ang_vel")`` and
``("base_quat", "observation.base_quat")``.

The anchor ladder was the exception. It read ``("anchor_rot_xyzw",
"observation.anchor_rot")`` -- prefixed, but with the ``_xyzw`` suffix dropped
-- so the spelling the convention implies was absent. Measured on one
observation dict whose two keys are both written to the convention:

=================================  ============================  ===========
observation key                    signal                        before
=================================  ============================  ===========
``observation.root_ang_vel_local`` root angular velocity         resolves
``observation.anchor_rot_xyzw``    anchor rotation               ``KeyError``
=================================  ============================  ===========

One dict, two ladders on the same class, one convention, two answers. The
refusal that fired named the runtime's ``body.<anchor>.quat`` key, the
``anchor_rot_xyzw=`` kwarg and ``body_rot_xyzw`` -- not any prefixed spelling,
so it did not tell the caller that a prefixed form was read at all, nor which
one.

Why nothing caught it
---------------------
The ladder rung is uncovered. ``_extract_anchor_rot``'s prefixed rung
(``policy.py`` line 469 before this change) and the ``body_rot_xyzw`` rung below
it were both absent from the coverage report, because every existing test
supplies the anchor rotation either as a kwarg or as the runtime's
``body.<anchor>.quat``. So neither the accepted spelling nor the refused one was
graded, and no assertion anywhere compared the two ladders' key vocabularies.

What this pins
--------------
* one observation dict carrying both signals under the convention spelling
  resolves both, and to the same values the bare keys give;
* the convention itself, derived from the package rather than listed here, so a
  ladder added later is held to it the hour it lands;
* the suffix-less ``observation.anchor_rot`` still resolves -- the fix is
  additive and removes no spelling a caller may already write;
* the runtime's ``body.<anchor>.quat`` still wins over the whole ladder, so a
  simulation rollout resolves exactly the rotation it always did.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import numpy as np
import pytest

import strands_robots.policies.protomotions.policy as policy_mod
from strands_robots.policies.protomotions import ProtoMotionsPolicy
from strands_robots.policies.protomotions.state_utils import mujoco_wxyz_to_xyzw
from tests.policies.protomotions.test_observation_state_reads_the_same_through_every_convention import (
    _policy,
)

PACKAGE_ROOT = pathlib.Path(policy_mod.__file__).parent.parent.parent

#: 45 degrees about +z in ``xyzw``. Every component differs from the identity
#: quaternion, so a ladder that resolved nothing and fell through to a default
#: could not coincide with the right answer.
ANCHOR_XYZW: list[float] = [0.0, 0.0, 0.3826834, 0.9238795]

#: Three distinct components, so a mis-ordered read is visible.
ROOT_AVEL: list[float] = [0.1, -0.2, 0.3]

#: A different rotation, used only where two candidate keys are present at once
#: and the test is about which one wins.
OTHER_WXYZ: list[float] = [0.7071068, 0.7071068, 0.0, 0.0]

#: The observation-key ladders this package is known to carry, as
#: ``module suffix -> how many``. A floor rather than an exact inventory: a
#: ladder added later must raise the count, never lower it, and the derived scan
#: below is what holds a new one to the convention.
KNOWN_LADDER_FILES: dict[str, int] = {
    "policies/protomotions/policy.py": 2,
    "policies/wbc/policy.py": 2,
}


# ----------------------------------------------------------------------------
# The convention, derived from source rather than restated
# ----------------------------------------------------------------------------
def _ladders_in_source(source: str) -> list[tuple[int, list[str], str]]:
    """Every observation-key fallback ladder in ``source``.

    A ladder is a tuple or list of string literals that is *consumed as keys* --
    the iterable of a ``for`` statement, or an argument to a call -- and that
    mixes at least one bare key with at least one ``observation.``-prefixed one.
    Both halves of that test are load-bearing, and they earn their place on
    different inputs. The consumption site is what excludes the two shapes this
    package really has -- a module-level list of default camera feature keys
    (``lerobot_local.molmoact2``) and a membership test over dataset columns
    (``transforms.base``), neither of which is a fallback ladder, and the second
    of which does carry a bare key. Requiring a bare key has no instance in the
    package today, so it is graded on an exemplar below: an all-prefixed key
    loop names nothing to derive the convention spelling from, and reading one
    as a ladder would demand ``observation.observation.images.left``.

    Args:
        source: Python source. Taken as text rather than as a function object so
            the same predicate can grade the shipped package and the constructed
            exemplars below, which have no source file.

    Returns:
        One ``(line, keys, bare_key)`` per ladder, where ``bare_key`` is the
        first key carrying no ``observation.`` prefix.
    """
    tree = ast.parse(source)
    sites: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            sites.append(node.iter)
        elif isinstance(node, ast.Call):
            sites.extend(node.args)
            sites.extend(keyword.value for keyword in node.keywords)

    found: list[tuple[int, list[str], str]] = []
    for node in sites:
        if not isinstance(node, ast.Tuple | ast.List):
            continue
        keys = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if len(keys) != len(node.elts) or len(keys) < 2:
            continue
        bare = [k for k in keys if not k.startswith("observation.")]
        if not bare or not any(k.startswith("observation.") for k in keys):
            continue
        found.append((node.lineno, keys, bare[0]))
    return found


def _ladders_in_package() -> list[tuple[str, int, list[str], str]]:
    """Every observation-key ladder in ``strands_robots``, as ``(path, line, keys, bare)``."""
    out: list[tuple[str, int, list[str], str]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for line, keys, bare in _ladders_in_source(path.read_text(encoding="utf-8")):
            out.append((str(path.relative_to(PACKAGE_ROOT)), line, keys, bare))
    return out


def _violations(ladders: list[tuple[int, list[str], str]]) -> list[tuple[int, str]]:
    """Ladders whose bare key has no ``observation.``-prefixed spelling beside it."""
    return [(line, bare) for line, keys, bare in ladders if f"observation.{bare}" not in keys]


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture
def tracker() -> ProtoMotionsPolicy:
    """A tracker with a stub session and a flat reference motion."""
    return _policy()


def _anchor(pol: ProtoMotionsPolicy, obs: dict[str, Any], **kwargs: Any) -> np.ndarray:
    return np.asarray(pol._extract_anchor_rot(obs, dict(kwargs))).ravel()


def _avel(pol: ProtoMotionsPolicy, obs: dict[str, Any], **kwargs: Any) -> np.ndarray:
    return np.asarray(pol._extract_root_local_ang_vel(obs, dict(kwargs))).ravel()


# ----------------------------------------------------------------------------
# The regression
# ----------------------------------------------------------------------------
class TestOneObservationDictResolvesBothSignals:
    """The finding: two ladders, one convention, and they must agree."""

    def test_both_signals_resolve_when_both_keys_carry_the_prefix(self, tracker: ProtoMotionsPolicy) -> None:
        """The money case -- a caller who prefixes both keys the same way.

        Before this change ``observation.root_ang_vel_local`` resolved and
        ``observation.anchor_rot_xyzw`` raised ``KeyError``, out of one dict.
        """
        obs = {
            "observation.anchor_rot_xyzw": ANCHOR_XYZW,
            "observation.root_ang_vel_local": ROOT_AVEL,
        }
        assert _anchor(tracker, obs) == pytest.approx(ANCHOR_XYZW, abs=1e-6), (
            "the prefixed anchor rotation must resolve, as its sibling ladder's prefixed key does"
        )
        assert _avel(tracker, obs) == pytest.approx(ROOT_AVEL, abs=1e-6), (
            "premise: the sibling ladder already accepted its prefixed key"
        )

    @pytest.mark.parametrize(
        "key",
        ["anchor_rot_xyzw", "observation.anchor_rot_xyzw", "observation.anchor_rot"],
        ids=["bare", "prefixed", "prefixed-suffixless"],
    )
    def test_every_anchor_ladder_key_resolves_the_same_rotation(self, tracker: ProtoMotionsPolicy, key: str) -> None:
        """The ladder's rungs are spellings of one signal, so they must agree.

        This rung does not reorder components -- only the runtime's
        ``body.<anchor>.quat`` rung converts ``wxyz`` to ``xyzw`` -- so every key
        here carries ``xyzw`` and must come back unchanged.
        """
        assert _anchor(tracker, {key: ANCHOR_XYZW}) == pytest.approx(ANCHOR_XYZW, abs=1e-6)

    def test_the_refusal_names_the_prefixed_spelling(self, tracker: ProtoMotionsPolicy) -> None:
        """A caller who reads the refusal must learn the prefixed form is read.

        The old message named the runtime key, the kwarg and ``body_rot_xyzw``,
        so a caller whose prefixed key had just been refused was not told that a
        prefixed spelling existed at all.
        """
        with pytest.raises(KeyError) as excinfo:
            _anchor(tracker, {})
        assert "observation.anchor_rot_xyzw" in str(excinfo.value), (
            "the remedy must name the prefixed spelling this ladder reads"
        )


class TestEveryLadderCarriesTheConventionSpelling:
    """Derived from the package, so a ladder added later is graded on arrival."""

    def test_every_observation_key_ladder_carries_the_convention_spelling(self) -> None:
        offenders = [
            f"{path}:{line} {keys} is missing 'observation.{bare}'"
            for path, line, keys, bare in _ladders_in_package()
            if f"observation.{bare}" not in keys
        ]
        assert not offenders, (
            "every observation-key ladder pairs its bare key with "
            "'observation.<that key>', so a caller can spell one dict's keys "
            "one way: " + "; ".join(offenders)
        )

    def test_the_scan_finds_every_ladder_known_to_exist(self) -> None:
        """Non-vacuity: a scan that reads nothing would pass the rule above."""
        ladders = _ladders_in_package()
        assert len(ladders) >= sum(KNOWN_LADDER_FILES.values()), (
            f"the scan found {len(ladders)} ladders; it has gone blind"
        )
        by_file: dict[str, int] = {}
        for path, _line, _keys, _bare in ladders:
            by_file[path] = by_file.get(path, 0) + 1
        for suffix, count in KNOWN_LADDER_FILES.items():
            matched = sum(n for path, n in by_file.items() if path.replace("\\", "/").endswith(suffix))
            assert matched >= count, f"expected at least {count} ladder(s) in {suffix}, found {matched}"

    def test_the_rule_grades_a_constructed_ladder_and_reports_a_constructed_violation(self) -> None:
        """The package is clean after the fix, so the rule is graded on exemplars."""
        compliant = 'for key in ("gripper_width", "observation.gripper_width"):\n    read(obs, key)\n'
        violating = 'for key in ("gripper_width", "observation.gripper"):\n    read(obs, key)\n'
        not_a_ladder = 'DEFAULT_KEYS = ["observation.images.left", "observation.images.right"]\n'
        prefixed_only = 'for key in ("observation.images.left", "observation.images.right"):\n    read(obs, key)\n'

        assert _violations(_ladders_in_source(compliant)) == []
        assert [bare for _line, bare in _violations(_ladders_in_source(violating))] == ["gripper_width"]
        assert _ladders_in_source(not_a_ladder) == [], (
            "a feature list that is never consumed as keys is not a fallback ladder"
        )
        assert _ladders_in_source(prefixed_only) == [], (
            "an all-prefixed key loop names no bare key to derive the convention "
            "spelling from, so it is not a ladder this rule can grade"
        )
        outcomes = {_violations(_ladders_in_source(src)) == [] for src in (compliant, violating)}
        assert outcomes == {True, False}, "the rule must be able to answer both ways"


class TestWhatIsUnchanged:
    """Every cell here held before the change and must go on holding."""

    def test_the_suffixless_prefixed_spelling_still_resolves(self, tracker: ProtoMotionsPolicy) -> None:
        """The fix is additive: it removes no spelling a caller may already write."""
        obs = {"observation.anchor_rot": ANCHOR_XYZW}
        assert _anchor(tracker, obs) == pytest.approx(ANCHOR_XYZW, abs=1e-6)

    def test_the_bare_keys_still_resolve(self, tracker: ProtoMotionsPolicy) -> None:
        obs = {"anchor_rot_xyzw": ANCHOR_XYZW, "root_ang_vel_local": ROOT_AVEL}
        assert _anchor(tracker, obs) == pytest.approx(ANCHOR_XYZW, abs=1e-6)
        assert _avel(tracker, obs) == pytest.approx(ROOT_AVEL, abs=1e-6)

    def test_an_explicit_kwarg_still_wins_over_the_ladder(self, tracker: ProtoMotionsPolicy) -> None:
        obs = {"observation.anchor_rot_xyzw": [0.0, 0.0, 0.0, 1.0]}
        got = _anchor(tracker, obs, anchor_rot_xyzw=ANCHOR_XYZW)
        assert got == pytest.approx(ANCHOR_XYZW, abs=1e-6), "kwargs are checked before any observation key"

    def test_the_runtime_supplied_body_quat_still_wins_over_the_ladder(self, tracker: ProtoMotionsPolicy) -> None:
        """A simulation rollout resolves exactly the rotation it always did.

        The runtime merges ``body.<anchor>.quat`` from this policy's
        ``required_bodies``, and that rung is checked before the ladder. Adding a
        key to the ladder therefore cannot move the rollout path, which is what
        makes this change safe for every existing caller.
        """
        anchor_key = f"body.{tracker._config.anchor_body_name}.quat"
        obs = {anchor_key: OTHER_WXYZ, "observation.anchor_rot_xyzw": ANCHOR_XYZW}
        expected = np.asarray(mujoco_wxyz_to_xyzw(np.asarray(OTHER_WXYZ, dtype=np.float32))).ravel()
        assert _anchor(tracker, obs) == pytest.approx(expected, abs=1e-6), (
            "the runtime's declared-body rung must still win over the ladder below it"
        )

    def test_an_observation_that_resolves_nothing_still_raises(self, tracker: ProtoMotionsPolicy) -> None:
        with pytest.raises(KeyError):
            _anchor(tracker, {"some.unrelated.key": 1.0})
        with pytest.raises(KeyError):
            _avel(tracker, {"some.unrelated.key": 1.0})
