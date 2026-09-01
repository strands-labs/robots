"""Confirmation that a signalled session process has actually exited.

A tool that stops a background session sends a signal and then has to answer a
separate question: did the process go away? Sending SIGKILL is not the answer.
The kernel delivers it asynchronously, and a task inside an uninterruptible
wait - a serial ioctl on a teleoperation bus, a stalled CUDA or network call in
a training step - stays in the process table until that wait returns. A stop
that reports success on the strength of having *sent* the signal tells the
caller the arm is released and the GPU is free when neither may be true.

:func:`confirm_exit` is that answer and :func:`unstopped_result` is the report
for when it is not affirmative, shared by every session ``stop`` verb so the
rule is stated once rather than per verb.
"""

from __future__ import annotations

from typing import Any

import psutil

# How long to let a process wind itself down after SIGTERM before escalating.
# The session processes this covers flush a dataset shard or a checkpoint on
# the way out, so the grace period is real work, not politeness.
SIGTERM_GRACE_S = 2.0

# How long to wait for SIGKILL to take effect before reporting that it has not.
# A process still present after this is in an uninterruptible wait; more waiting
# does not change the verdict, and the caller needs the verdict.
SIGKILL_CONFIRM_S = 2.0


def confirm_exit(proc: psutil.Process, timeout: float) -> bool | None:
    """Wait up to ``timeout`` seconds for ``proc`` to leave the process table.

    Args:
        proc: The process being stopped. It must have been constructed *before*
            the signal was sent: psutil records the creation time at
            construction, so every probe here is identity-checked and a PID that
            was recycled in the meantime reads as exited rather than as the
            original process still running.
        timeout: Seconds to wait. Returns as soon as the process is gone.

    Returns:
        ``True`` when the process is known to have exited, ``False`` when it is
        known to be still present, and ``None`` when neither could be
        established - this user may not inspect the process
        (:class:`psutil.AccessDenied`). ``None`` must not be read as either
        answer: the PID existing already said it is not death, and being unable
        to look is no evidence that it left.
    """
    try:
        proc.wait(timeout=timeout)
    except psutil.TimeoutExpired:
        return False
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied:
        return None
    return True


def unstopped_result(session_name: str, pid: int, verdict: bool | None, doing: str) -> dict[str, Any]:
    """Report a session whose process was not confirmed gone after SIGKILL.

    The caller keeps the session record when it gets this: that store is the
    only place a detached session's PID is written down, so dropping it would
    leave the process running with no supported way left to stop it.

    Args:
        session_name: Name the session is tracked under.
        pid: PID that was signalled.
        verdict: The non-affirmative :func:`confirm_exit` answer being reported -
            ``False`` (still present) or ``None`` (could not be determined).
        doing: What the process is still doing, in the caller's terms
            (``"driving the robot"``, ``"training"``).

    Returns:
        A tool error result whose ``json`` block carries ``stopped`` verbatim, so
        an unknown outcome stays unknown to whoever reads it.
    """
    if verdict is None:
        what = (
            f"Session '{session_name}' (PID {pid}) was signalled with SIGTERM and SIGKILL, but whether it "
            f"exited could not be determined: this user may not inspect it. A session started under sudo "
            f"for device access and stopped as the invoking user reads this way."
        )
    else:
        what = (
            f"Session '{session_name}' (PID {pid}) is still present after SIGTERM and SIGKILL, so it is "
            f"still {doing}. A process that outlives SIGKILL is in an uninterruptible wait."
        )
    return {
        "status": "error",
        "content": [
            {"text": f"{what} Its record is kept so the session stays stoppable; inspect it with 'ps -p {pid}'."},
            {"json": {"session_name": session_name, "pid": pid, "stopped": verdict}},
        ],
    }
