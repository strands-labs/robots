"""What ``run`` and ``lockstep`` share around THE tick: the declared rate and the run events."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .bind import Bound
from .result import RunResult
from .source import RunPlan

if TYPE_CHECKING:
    from ..robot import Robot


def emit(robot: Robot, event: str, payload: dict[str, Any]) -> None:
    events = getattr(robot, "events", None)
    if events is not None:
        events.emit(event, {"device_id": robot.id, **payload})


def label(policy: object) -> str:
    """The policy as the operator reads it: a spec as written, a function or a registered handle by its name."""
    from ...policy.callable import CallablePolicy

    if isinstance(policy, CallablePolicy):
        return policy.name
    return str(policy) if isinstance(policy, str | PathLike) else str(getattr(policy, "__name__", None) or repr(policy))


def files_digest(policy: object) -> str | None:
    """SHA-256 over the names and bytes of the files a policy spec reads from disk, or None if it names none."""
    if not isinstance(policy, str | PathLike):
        return None
    local = Path(str(policy).partition("?")[0]).expanduser()
    if not local.exists():
        return None  # a preset name or a hub repo id: nothing on this machine to change under the operator
    root = local if local.is_dir() else local.parent
    files = sorted(_files(local, set())) if local.is_dir() else [local]
    total = hashlib.sha256()
    for f in files:
        one = hashlib.sha256()  # read in chunks, not hashlib.file_digest: that is Python 3.11+, the floor is 3.10
        with f.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                one.update(chunk)
        total.update(f"{f.relative_to(root).as_posix()}\0{one.hexdigest()}\n".encode())
    return total.hexdigest()


def _files(folder: Path, seen: set[Path]) -> list[Path]:
    """Every file under ``folder``, through symlinked folders as a loader reads them, each real folder once."""
    if (real := folder.resolve()) in seen:
        return []
    seen.add(real)
    found: list[Path] = []
    for entry in folder.iterdir():
        found += _files(entry, seen) if entry.is_dir() else [entry] if entry.is_file() else []
    return found


def bind_pinned(robot: Robot, plan: RunPlan | None, policy: Any, units: str | None) -> tuple[Bound, Path | None]:
    """Bind the policy; for an approved run of local files, bind a private copy whose digest is the approved one.

    The copy is taken and checked before anything is loaded, so a write to the original after the yes (or during the
    load) never reaches the robot. The caller removes the returned folder with ``drop`` when the run ends.
    """
    from ..safety import RequestRefused
    from .bind import bind

    if plan is None or plan.files is None:
        return bind(robot, policy, units), None
    path, sep, query = str(policy).partition("?")
    src = Path(path).expanduser()
    tmp = Path(tempfile.mkdtemp(prefix="strands-run-"))
    try:
        copy = tmp / (src.name or "policy")
        shutil.copytree(src, copy) if src.is_dir() else shutil.copy2(src, copy)  # symlinks are copied as content
        if files_digest(copy) != plan.files:
            raise RequestRefused(f"{robot.id}: {plan.policy} changed on disk after the operator approved it; run again")
        return bind(robot, f"{copy}{sep}{query}", units), tmp
    except BaseException:
        drop(tmp)
        raise


def drop(folder: Path | None) -> None:
    """Remove a private policy copy (``bind_pinned``); nothing for a run that loaded no local files."""
    if folder is not None:
        shutil.rmtree(folder, ignore_errors=True)


def approve_run(robot: Robot, plan: RunPlan) -> None:
    """The operator's one yes for an agent's run on a real body; ``RequestRefused`` names the fix otherwise (K6)."""
    from ..safety import RequestRefused

    if robot.approve is None:
        raise RequestRefused(
            f"{robot.id} is a real robot and this run came from an agent: pass approve=<callable(RunPlan) -> bool> "
            "to Robot(...) so an operator can allow it"
        )
    if not robot.approve(plan):
        raise RequestRefused(f"{robot.id}: the operator refused the run {plan}")


def announce(robot: Robot, result: RunResult) -> None:
    """``run.stopped`` for every finished run, and ``fault`` first when a policy fault ended it."""
    if result.fault is not None and result.stopped_by == "fault":
        emit(robot, "fault", {"reason": result.fault})
    emit(robot, "run.stopped", {"stopped_by": result.stopped_by, "steps": result.steps})


def declared(bounds: Iterable[Bound], hz: float | None) -> float | None:
    """The rate the policies declare (chunks are replayed at it), refusing a disagreement or an ``hz`` against it."""
    rates = {b.hz: b.policy for b in bounds if b.hz is not None}
    if len(rates) > 1:
        said = ", ".join(f"{p!r} {r:g} Hz" for r, p in rates.items())
        raise ValueError(f"these policies declare different rates: {said}; one world steps at one rate")
    if rates and hz is not None and abs(hz - next(iter(rates))) > 1e-9:
        rate, policy = next(iter(rates.items()))
        raise ValueError(f"{policy!r} runs at {rate:g} Hz; hz={hz:g} would replay it at another rate (leave hz out)")
    return next(iter(rates), None)


def rate(bound: Bound, hz: float | None) -> float:
    said = declared([bound], hz)  # an hz against the policy's own rate is refused
    r = hz if hz is not None else (said or 30.0)
    if r <= 0:
        raise ValueError(f"hz must be > 0, got {r}")
    return float(r)
