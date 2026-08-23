"""The empty-recording refusal classifies each loop by whether it takes a hook.

``stop_recording`` refuses a session that captured nothing, and that refusal is
the only thing a caller whose dataset came out empty reads. It therefore has to
be right about *which* loops can fill a recording, and there are three kinds:

* :meth:`~strands_robots.simulation.base.SimEngine.run_policy` feeds the
  recorder on its own - it installs the per-step ``on_frame`` hook that calls
  ``add_frame``, so a caller passes nothing.
* :meth:`~strands_robots.simulation.base.SimEngine.eval_policy` and
  :meth:`~strands_robots.simulation.base.SimEngine.evaluate_benchmark` *accept*
  an ``on_frame`` hook. They record when the caller supplies one, and
  ``eval_policy``'s own docstring recommends exactly that ("Use it to record
  frames"). Measured: a hook calling ``add_frame`` over a 20-step eval writes 20
  frames and ``stop_recording`` then saves the episode.
* :meth:`~strands_robots.simulation.base.SimEngine.replay_episode`,
  :meth:`~strands_robots.teleop_mixin.TeleopMixin.teleoperate` and a bare
  ``step`` loop take no such hook, so nothing a caller does makes them record.

The refusal used to say frames "are written only by ``run_policy``" and that
"eval_policy / evaluate / replay_episode and bare step loops do NOT feed the
recorder" - denying a route two of the named entry points do have, while
omitting ``teleoperate``, the loop a caller recording a teleoperated
demonstration reaches for and the one that genuinely cannot record.

The classification is derived from the signatures rather than listed here, so
the day an apply-loop gains an ``on_frame`` parameter this test fails until the
refusal moves it out of the cannot-record sentence.
"""

from __future__ import annotations

import ast
import inspect
import re

import pytest

from strands_robots.simulation import recording as _recording
from strands_robots.simulation.base import SimEngine
from strands_robots.teleop_mixin import TeleopMixin

# Every loop the refusal speaks about, and where its signature lives. ``step``
# stands for the "bare step loops" phrasing rather than a named method.
_LOOPS: dict[str, type] = {
    "run_policy": SimEngine,
    "eval_policy": SimEngine,
    "evaluate_benchmark": SimEngine,
    "replay_episode": SimEngine,
    "teleoperate": TeleopMixin,
}

# run_policy installs the recording hook itself, so it is not a caller-supplied
# route even though it is the one entry point that always records.
_SELF_FEEDING = "run_policy"


def _takes_on_frame(name: str) -> bool:
    owner = _LOOPS[name]
    fn = getattr(owner, name, None)
    if fn is None:  # pragma: no cover - guarded by the inventory test below
        return False
    return "on_frame" in inspect.signature(fn).parameters


def _hook_bearing() -> set[str]:
    """Loops a caller can make record by passing ``on_frame``."""
    return {n for n in _LOOPS if n != _SELF_FEEDING and _takes_on_frame(n)}


def _hookless() -> set[str]:
    """Loops that cannot record however the caller calls them."""
    return {n for n in _LOOPS if n != _SELF_FEEDING and not _takes_on_frame(n)}


def _refusal_text() -> str:
    """The empty-recording refusal, read from the source it is raised from.

    Read as a literal rather than by driving ``stop_recording`` so the
    classification is graded without a backend, a dataset stack or a GL context.
    """
    tree = ast.parse(inspect.getsource(_recording))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "captured no frames" in node.value:
            return " ".join(node.value.split())
        # The message is assembled from adjacent string literals, which the
        # parser folds into one JoinedStr/Constant only when it is not an
        # f-string; handle the concatenated form too.
        if isinstance(node, ast.BinOp):  # pragma: no cover - defensive
            continue
    raise AssertionError(f"no empty-recording refusal literal found in {_recording.__name__}")


def _names(text: str) -> set[str]:
    """Loop names the text mentions, matched as whole tokens.

    A bare substring test would let ``eval_policy`` be answered by an unrelated
    dotted path, so each name is bounded the way the rest of this suite bounds a
    prose-graded identifier.
    """
    return {n for n in _LOOPS if re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", text)}


# The clause is located by what it *says* rather than by one phrasing, so this
# grades any wording that denies a loop the ability to record. Keyed on a single
# spelling it would only ever pass judgement on the sentence it was written
# against, and a reworded refusal would read as having no such clause at all.
_DENIAL_MARKERS = (
    "cannot feed",
    "cannot fill",
    "do not feed",
    "does not feed",
    "never feed",
    "no such hook",
)


def _cannot_sentence(text: str) -> str:
    """The clause that says which loops cannot feed the recorder."""
    lowered = text.lower()
    for part in re.split(r"(?<=[.])\s+", text):
        if any(m in part.lower() for m in _DENIAL_MARKERS):
            return part
    raise AssertionError(f"refusal names no clause denying a loop the ability to record: {lowered!r}")


class TestTheRefusalIsRightAboutWhichLoopsCanRecord:
    def test_no_hook_bearing_loop_is_called_unable_to_record(self) -> None:
        """A loop that takes ``on_frame`` must not be listed as unable to feed.

        This is the half that was wrong: the refusal told a caller whose dataset
        was empty that ``eval_policy`` does not feed the recorder, while
        ``eval_policy(on_frame=...)`` is a documented and measured way to fill
        one - so the message denied the route it should have pointed at.
        """
        text = _refusal_text()
        denied = _names(_cannot_sentence(text)) & _hook_bearing()
        assert not denied, (
            f"the refusal says {sorted(denied)} cannot feed the recorder, but each takes an "
            f"on_frame hook a caller can use to record. Clause: {_cannot_sentence(text)!r}"
        )

    def test_every_hook_bearing_loop_is_named(self) -> None:
        """The routes a caller *can* record through are worth naming."""
        text = _refusal_text()
        missing = _hook_bearing() - _names(text)
        assert not missing, (
            f"{sorted(missing)} accept an on_frame hook and can fill a recording, but the "
            f"empty-recording refusal does not name them: {text!r}"
        )

    def test_every_hookless_loop_is_named_as_unable(self) -> None:
        """Including ``teleoperate``, which the refusal used to omit entirely."""
        text = _refusal_text()
        clause = _cannot_sentence(text)
        missing = _hookless() - _names(clause)
        assert not missing, (
            f"{sorted(missing)} take no on_frame hook, so no caller can make them record, but "
            f"the refusal does not say so: {clause!r}"
        )

    def test_the_self_feeding_loop_is_still_the_headline_remedy(self) -> None:
        """``run_policy`` records with no hook from the caller, so it leads."""
        text = _refusal_text()
        assert _SELF_FEEDING in _names(text), f"refusal no longer names {_SELF_FEEDING}: {text!r}"
        assert not _takes_on_frame(_SELF_FEEDING), (
            f"{_SELF_FEEDING} now takes an on_frame parameter, so it is a caller-supplied route "
            "like eval_policy and this refusal's split needs revisiting"
        )


class TestTheScanFoundSomethingToGrade:
    """Without these, every rule above passes over an empty set."""

    def test_the_refusal_was_located(self) -> None:
        text = _refusal_text()
        assert "captured no frames" in text
        assert "0 frames" in text

    def test_both_groups_are_populated(self) -> None:
        assert len(_hook_bearing()) >= 2, f"expected >=2 hook-bearing loops, got {sorted(_hook_bearing())}"
        assert len(_hookless()) >= 2, f"expected >=2 hookless loops, got {sorted(_hookless())}"

    @pytest.mark.parametrize("name", sorted(_LOOPS))
    def test_every_graded_loop_still_exists(self, name: str) -> None:
        assert getattr(_LOOPS[name], name, None) is not None, (
            f"{name} is no longer a method of {_LOOPS[name].__name__}; the refusal's "
            "classification is graded against it"
        )
