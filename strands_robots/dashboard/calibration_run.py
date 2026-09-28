"""Run `lerobot-calibrate` as a managed pty session, so the dashboard can walk an
operator through calibration step by step instead of handing them a terminal command.

Why a pty and not a rewrite of the procedure: lerobot's calibrate() is the code that
will also run in every terminal, and it is interactive by design (input() prompts +
a live min/max table). Driving the REAL flow means the wizard can never drift from
what lerobot actually does. The flow, read from source (so_follower.py + motors_bus
.record_ranges_of_motion):

  0. (only when a file for this id already exists) "Press ENTER to use provided
     calibration file ... or type 'c' ... to run calibration"
  1. torque is DISABLED - the arm goes limp; the dashboard commands no motion
  2. "Move ... to the middle of its range of motion and press ENTER"
  3. "Recording positions. Press ENTER to stop..." + a live NAME|MIN|POS|MAX table,
     redrawn with ANSI cursor-up - the LAST table is the current one
  4. "Calibration saved to <path>"

The step parser is pure (text in, verdict out), so the wizard's state machine can
be reasoned about without an arm on the desk.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

__all__ = ["CONFIRM_KEY", "cli_args", "wizard_step", "CalibrationRun", "runs", "start", "get"]

# ---------------------------------------------------------------------------
# the command (same facts as frontend/src/lib/calibrateCommand.ts, draccus shape)
# ---------------------------------------------------------------------------

#: Confirm the run must carry, or a standing grant: the wizard disables torque on a
#: real arm, which is a hardware command even though the motion that follows is the
#: operator's hand.
CONFIRM_KEY = "confirm"

_SEGMENT = re.compile(r"^[A-Za-z0-9._:-]{1,64}\Z")


def cli_args(role: str, model: str, device_id: str, port: str) -> list[str]:
    """argv for the installed draccus CLI - follower is --robot.*, leader is --teleop.*.

    Raises ValueError with an operator-readable sentence rather than guessing: a wrong
    role writes the calibration into the wrong directory tree.
    """
    r = (role or "").strip().lower()
    if r not in ("follower", "leader"):
        raise ValueError(
            f"role must be follower or leader, got {role!r} - the role decides whether "
            "the file lands under robots/ or teleoperators/, so it cannot be guessed"
        )
    m = (model or "").strip().lower()
    if not m or not _SEGMENT.match(m):
        raise ValueError(f"device model {model!r} is not a lerobot model name")
    did = (device_id or "").strip()
    if not did or not _SEGMENT.match(did) or did.startswith("."):
        raise ValueError(
            f"device id {device_id!r} is not usable - lerobot uses it as a FILE NAME, "
            "so it must be one path segment (letters, digits, . _ : -)"
        )
    p = (port or "").strip()
    if not p or not p.startswith("/dev/") or any(c.isspace() for c in p):
        raise ValueError(f"port {port!r} is not a serial device path")
    prefix = "robot" if r == "follower" else "teleop"
    return [
        f"--{prefix}.type={m}",
        f"--{prefix}.id={did}",
        f"--{prefix}.port={p}",
    ]


def _calibrate_argv() -> list[str]:
    """The interpreter running this dashboard runs the wizard, so it always drives the same
    lerobot the arms run, never whatever is on PATH.

    Raises:
        ImportError: lerobot is not installed here; the message names the extra.
    """
    from strands_robots.utils import require_optional

    require_optional(
        "lerobot.scripts.lerobot_calibrate",
        pip_install="lerobot",
        extra="lerobot",
        purpose="the interactive lerobot-calibrate flow the wizard drives",
    )
    return [sys.executable, "-m", "lerobot.scripts.lerobot_calibrate"]


# ---------------------------------------------------------------------------
# the step parser (pure)
# ---------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\r")
_TABLE_ROW = re.compile(r"^(\S[\w ]*?)\s*\|\s*(-?\d+)\s*\|\s*(-?\d+)\s*\|\s*(-?\d+)\s*\Z")
_SAVED = re.compile(r"Calibration saved to\s+(.+)")


def _clean(raw: str) -> str:
    return _ANSI.sub("", raw)


def _last_table(text: str) -> list[dict[str, int | str]]:
    """The most recent NAME|MIN|POS|MAX block - the table is redrawn in place, so only
    rows after the LAST header are current."""
    lines = text.splitlines()
    header_at = -1
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("NAME") and "|" in ln and "MIN" in ln:
            header_at = i
    if header_at < 0:
        return []
    rows: list[dict[str, int | str]] = []
    for ln in lines[header_at + 1 :]:
        m = _TABLE_ROW.match(ln.strip())
        if not m:
            if rows:
                break
            continue
        rows.append(
            {"name": m.group(1).strip(), "min": int(m.group(2)), "pos": int(m.group(3)), "max": int(m.group(4))}
        )
    return rows


def wizard_step(raw: str, *, alive: bool, returncode: int | None) -> dict[str, Any]:
    """What the wizard should show RIGHT NOW, from everything the pty has printed.

    Later markers win over earlier ones - the output is cumulative, so the flow's own
    order (reuse? -> middle -> recording -> saved) is the priority order reversed.
    """
    text = _clean(raw)

    saved = None
    for m in _SAVED.finditer(text):
        saved = m.group(1).strip()
    if saved:
        return {"step": "saved", "path": saved, "waiting": False}

    if not alive:
        # Died without the saved line: name the failure instead of showing a spinner.
        reason = _failure_reason(text, returncode)
        return {"step": "failed", "reason": reason, "returncode": returncode, "waiting": False}

    # Recording phase: the stop-prompt has been printed and a table follows.
    if "Press ENTER to stop" in text:
        return {
            "step": "recording",
            "motors": _last_table(text),
            "prompt": "move every joint (except wrist_roll) through its FULL range by hand, "
            "then press stop - a joint you skip keeps a one-point range and lerobot refuses to save",
            "waiting": True,
        }

    if "middle of its range of motion and press ENTER" in text:
        return {
            "step": "middle",
            "prompt": "torque is off and the arm is limp - hold it at the MIDDLE of every "
            "joint's travel, then continue",
            "waiting": True,
        }

    if "Press ENTER to use provided calibration file" in text:
        return {
            "step": "reuse",
            "prompt": "a calibration for this id already exists - keep it, or redo it from scratch",
            "waiting": True,
        }

    return {"step": "starting", "waiting": False}


def _failure_reason(text: str, returncode: int | None) -> str:
    if "usage: lerobot-calibrate" in text or "unrecognized arguments" in text:
        return "the CLI refused the flags - this dashboard build and the installed lerobot disagree; report this, it is a bug, not your arm"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        low = ln.lower()
        if ln.startswith(("ValueError", "RuntimeError", "ConnectionError", "OSError", "serial.")) or (
            "error" in low and "logging" not in low
        ):
            return ln
    if "Traceback" in text:
        return lines[-1] if lines else f"exited {returncode}"
    if lines:
        return lines[-1]
    return f"exited with {returncode} before printing anything - check the port and the arm's power"


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------

_TAIL_CAP = 64 * 1024  # keep the last 64KB of pty output - plenty for every prompt


class CalibrationRun:
    """One live calibration under a pty. The pty matters twice: input() needs a tty-ish
    stdin to prompt at all, and record_ranges_of_motion polls stdin for the ENTER."""

    def __init__(self, *, role: str, model: str, device_id: str, port: str, argv: list[str] | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.role, self.model, self.device_id, self.port = role, model, device_id, port
        self.started_at = time.time()
        args = cli_args(role, model, device_id, port)
        cmd = (argv if argv is not None else _calibrate_argv()) + args
        self._master, slave = os.openpty()
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,  # its own group: cancel() never signals the dashboard
                close_fds=True,
            )
        finally:
            os.close(slave)
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        while True:
            try:
                chunk = os.read(self._master, 4096)
            except OSError:
                break
            if not chunk:
                break
            with self._lock:
                self._buf.extend(chunk)
                if len(self._buf) > _TAIL_CAP:
                    del self._buf[: len(self._buf) - _TAIL_CAP]

    def output(self) -> str:
        """Everything the pty has printed so far (last 64KB), undecoded of ANSI."""
        with self._lock:
            return self._buf.decode("utf-8", "replace")

    def alive(self) -> bool:
        """Is lerobot-calibrate still running?"""
        return self.proc.poll() is None

    def press(self, key: str) -> None:
        """'enter' answers every prompt; 'c' (reuse step only) asks for a fresh calibration."""
        if not self.alive():
            raise RuntimeError("this calibration run has already finished")
        if key == "enter":
            os.write(self._master, b"\n")
        elif key == "c":
            os.write(self._master, b"c\n")
        else:
            raise ValueError(f"key must be 'enter' or 'c', got {key!r}")

    def cancel(self) -> None:
        """Stop the run: SIGTERM its process group, SIGKILL after three seconds."""
        if self.alive():
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass  # already gone, or not ours to signal: either way there is nothing left to stop
            deadline = time.monotonic() + 3
            while self.alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            if self.alive():
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass  # same as the SIGTERM above

    def close(self) -> None:
        """Cancel and release the pty."""
        self.cancel()
        try:
            os.close(self._master)
        except OSError:
            pass  # the pty master may already be closed by the child's exit

    def status(self) -> dict[str, Any]:
        """The wizard view right now: identity, liveness, the current step and the output tail."""
        raw = self.output()
        alive = self.alive()
        step = wizard_step(raw, alive=alive, returncode=self.proc.returncode)
        tail = _clean(raw).splitlines()[-12:]
        return {
            "id": self.id,
            "port": self.port,
            "device_id": self.device_id,
            "model": self.model,
            "role": self.role,
            "alive": alive,
            "returncode": self.proc.returncode,
            "started_at": self.started_at,
            **step,
            "tail": tail,
        }


# module-level registry: the dashboard has one process, calibration has one operator
runs: dict[str, CalibrationRun] = {}


def start(*, role: str, model: str, device_id: str, port: str, argv: list[str] | None = None) -> CalibrationRun:
    """Begin a run, refusing a second wizard on a port that already has a LIVE one -
    two writers on one servo bus is the 'Port is in use!' collision."""
    for r in runs.values():
        if r.port == port and r.alive():
            raise RuntimeError(
                f"a calibration wizard is already running on {port} (session {r.id}) - finish or cancel it first"
            )
    run = CalibrationRun(role=role, model=model, device_id=device_id, port=port, argv=argv)
    runs[run.id] = run
    # Finished runs stay readable (their 'saved' path is the receipt), but the ledger is capped.
    finished = [s for s, r in runs.items() if not r.alive() and s != run.id]
    while len(runs) > 8 and finished:
        old = runs.pop(finished.pop(0), None)
        if old is not None:
            old.close()
    return run


def get(sid: str) -> CalibrationRun | None:
    """The run with this session id, or None when it was never started or was evicted."""
    return runs.get(sid)
