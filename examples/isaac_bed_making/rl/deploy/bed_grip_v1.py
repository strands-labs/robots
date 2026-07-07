"""Sensor-gated LEFT-hand grip on a bedsheet, coordinated with the arm-overlay pull.

Talks to the running ``brainco_bridge`` (TCP JSON on 127.0.0.1:9877) — NOT the serial port directly.
  * send  {"cmd":"set","left":[6 floats 0-1]}   0=open 1=closed
  * query {"cmd":"get"}  → per-finger left_touch_force [5 u16], left_proximity [5 u16], left_touch_ok

Sequence (sentinel handshake with g1_bed_pull_v2.py):
  1. wait for ``--arm-reached-file`` (the arm has reached the sheet and is holding)
  2. oppose the thumb laterally into a CLAW (``thumb_aux``→``--thumb-claw``) and let it settle, then
     close the four fingers FULLY (ramped over ``--close-s``) and ONLY AFTER they are closed + a settle
     (``--thumb-lag-s``) flex the thumb in to clamp over them — SEQUENCED so the fingers don't catch on
     the base of the thumb. Touch is polled to feed the empty-grab confirm; the close runs to ``--grip-max``.
  3. confirm fabric-in-hand on the HELD grip (see bed_grasp_confirm.py); under ``--require-fabric`` an
     empty verdict writes ``--empty-file`` and releases instead of signaling the draw
  4. create ``--gripped-file``  → the arm starts the draw
  5. hold the grip AND monitor the fingertip force through the draw (write ``--empty-file`` if it slips
     out) until ``--draw-done-file``
  6. release (open the fingers)

The LEFT hand is used because the right Brainco hand is mechanically damaged (kill-9 incident:
middle+pinky distal knuckles, thumb extension). Fingers/sensors are green-light (no leg/arm motion);
this commands only the hand.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time

from bed_grasp_confirm import GraspConfirmMonitor


class Bridge:
    def __init__(self, host, port):
        self._s = socket.socket()
        self._s.settimeout(5.0)
        self._s.connect((host, port))
        self._buf = b""

    def _rpc(self, obj):
        self._s.sendall((json.dumps(obj) + "\n").encode())
        while b"\n" not in self._buf:
            chunk = self._s.recv(8192)
            if not chunk:
                break
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        if not line.strip():
            raise RuntimeError("empty reply from brainco_bridge (connection closed mid-request — "
                               "usually a malformed command; note the bridge reads BOTH 'left' and "
                               "'right' on a 'set').")
        return json.loads(line)

    def get(self):
        return self._rpc({"cmd": "get"})

    def set_left(self, value6):
        # The bridge's "set" handler reads BOTH req["left"] and req["right"] unconditionally —
        # omitting "right" raises a server-side KeyError and the bridge closes the connection
        # (empty reply → JSONDecodeError). Always send both; keep the (damaged) right hand open.
        return self._rpc({"cmd": "set",
                          "left": [float(v) for v in value6],
                          "right": [0.0] * 6})

    def close(self):
        try:
            self._s.close()
        except OSError:
            # Best-effort teardown; the socket may already be closed by the peer.
            pass


def _wait_for(path, timeout_s, log):
    if not path:
        return True
    log(f"[grip] waiting for {path} (arm to reach the sheet)…")
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        if os.path.exists(path):
            return True
        time.sleep(0.1)
    log(f"[grip] timeout waiting for {path}")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9877)
    ap.add_argument("--grip-max", type=float, default=1.0,
                    help="max close fraction (0=open,1=closed). FULL close (1.0): the fingers AND the "
                         "thumb (flexion, in the adducted claw) clamp all the way for a firm fabric pinch "
                         "— hardware-validated 2026-06-19 to give a strong multi-finger grab (index ~51 "
                         "vs ~5 at the old 0.85 with the broken thumb).")
    ap.add_argument("--close-s", type=float, default=0.7,
                    help="seconds for the four FINGERS to flex from open to grip-max (fast, so they close "
                         "decisively before the thumb). The thumb flexion is SEQUENCED after — see --thumb-lag-s.")
    ap.add_argument("--rate-hz", type=float, default=50.0, help="finger position command rate (Hz)")
    ap.add_argument("--thumb-claw", type=float, default=1.0,
                    help="thumb_aux opposition (0=slap/flat, 1=thumb opposed across palm = CLAW). The "
                         "thumb is moved here FIRST so the digits close into the opposed thumb (you "
                         "cannot pinch thin fabric with a flat hand).")
    ap.add_argument("--claw-settle-s", type=float, default=0.6,
                    help="seconds to let the thumb travel to the claw position BEFORE closing the digits")
    ap.add_argument("--thumb-lag-s", type=float, default=0.3,
                    help="SETTLE seconds AFTER the fingers are fully closed before the thumb flexes in. "
                         "The thumb does NOT flex until the fingers have completely closed (thumb_start = "
                         "--close-s + this) — they catch on the BASE of the thumb if its flexion overlaps "
                         "theirs at all, even slowly (operator-specified, 2026-06-19). The thumb stays "
                         "OPPOSED (thumb_aux=claw) the whole time; only its flexion (thumb_curl) is "
                         "sequenced. The close runs until the thumb is fully clamped.")
    ap.add_argument("--hold-timeout", type=float, default=120.0, help="max grip hold before auto-release")
    ap.add_argument("--arm-reached-file", default="/tmp/arm_reached")
    ap.add_argument("--gripped-file", default="/tmp/sheet_gripped")
    ap.add_argument("--draw-done-file", default="/tmp/draw_done")
    # Empty-grab confirmation (did the claw close on FABRIC or on NOTHING?). See bed_grasp_confirm.py.
    ap.add_argument("--empty-file", default="/tmp/grab_empty",
                    help="sentinel written when the close is judged an EMPTY grab — the ask-the-human "
                         "trigger for 'closed on nothing' (only written under --require-fabric)")
    ap.add_argument("--require-fabric", action="store_true",
                    help="GATE the draw on fabric-in-hand: on an empty-grab verdict, write --empty-file, "
                         "skip --gripped-file, release. Default is measure-only (log the verdict, always "
                         "proceed) — calibrate the thresholds on hardware FIRST (analog of --abort-on-stall).")
    ap.add_argument("--confirm-force-rise", type=float, default=4.0,
                    help="per-finger SUSTAINED (held-grip) touch-force rise (u16) over the open-claw "
                         "baseline that counts as fabric. HARDWARE-CALIBRATED 2026-06-19 (touch-only, "
                         "proximity dead): empty/slip floor ≤2.5, light palm-up handoff grab ~4-5, solid "
                         "grab 18-39. Set to 4 (above the slip floor, below the weakest real grab) so a "
                         "light-but-real handoff grab isn't false-rejected — the MID-DRAW grip monitor "
                         "(--draw-lost-rise) is the real backstop, catching a grab that slips WHILE drawing.")
    ap.add_argument("--confirm-prox-dev", type=float, default=150.0,
                    help="per-finger |proximity - baseline| (u16) that counts as something-in-claw — only "
                         "used when --confirm-use-proximity is set (OFF by default; see that flag). "
                         "FIRST-PASS value; would need a closed-empty baseline calibration to be usable.")
    ap.add_argument("--confirm-fingers", type=int, default=1,
                    help="fingers showing fabric (touch OR proximity) to confirm a grab")
    ap.add_argument("--confirm-use-proximity", action="store_true",
                    help="let proximity vote in the grasp verdict. OFF by default: the canonical bridge "
                         "publishes proximity, but closing the claw swings the fingertips toward the "
                         "palm/each other → |prox-baseline| spikes ~30000 on an EMPTY close (geometry, "
                         "not fabric) → false-FABRIC. Needs a closed-empty baseline calibration first; "
                         "until then the verdict is touch-only.")
    ap.add_argument("--confirm-hold-s", type=float, default=0.4,
                    help="seconds to HOLD the closed grip and re-sample before judging fabric-vs-empty. "
                         "PEAK != CAPTURE: a transient brush during the close slips off by full close "
                         "(false-success on a taut anchored sheet, 2026-06-19) — the verdict keys on "
                         "fabric STILL loaded over this hold, not a peak that already collapsed.")
    ap.add_argument("--draw-lost-rise", type=float, default=3.0,
                    help="MID-DRAW grip monitor: per-finger touch-force rise floor (u16) below which the "
                         "grip is judged LOST during the draw (the sheet left the hand). A light grab can "
                         "pass the pre-draw gate then slip while pulling — and the arm-side draw can't see "
                         "it (empty and free draws read identical follow/tau on hardware), so the "
                         "FINGERTIP watches. Below the gate so a steady light grab isn't flagged.")
    ap.add_argument("--draw-lost-sustain", type=float, default=0.3,
                    help="seconds the grip force must stay below --draw-lost-rise to call it lost (vs a "
                         "transient dip while the fabric re-seats during the draw).")
    ap.add_argument("--no-wait", action="store_true", help="grip immediately (skip the arm-reached wait)")
    ap.add_argument("--present-claw", action="store_true",
                    help="show the OPEN claw (thumb opposed, fingers open) while waiting for the "
                         "arm-reached signal — for the human handoff: the hand is ready to receive the "
                         "sheet, then closes on the signal.")
    ap.add_argument("--dry", action="store_true", help="read-only: print touch/proximity, NO finger motion")
    args = ap.parse_args()

    try:
        br = Bridge(args.host, args.port)
    except OSError as e:
        raise SystemExit(f"[grip] cannot reach brainco_bridge at {args.host}:{args.port} ({e}). "
                         "Start the bridge first (brainco_touch).")

    # Clear OUR own coordination sentinels up front so a standalone/repeat run can't act on a stale
    # gripped/draw-done from a previous run (the arm-side script clears them too; this makes grip
    # safe to run on its own).
    for f in (args.gripped_file, args.draw_done_file, args.empty_file):
        try:
            os.remove(f)
        except OSError:
            # Nothing to clear: the sentinel is absent on a fresh run.
            pass

    try:
        st = br.get()
        print(f"[grip] left_touch_ok={st.get('left_touch_ok')}  "
              f"left_touch_force={st.get('left_touch_force')}  left_proximity={st.get('left_proximity')}")
        if args.dry:
            print("[grip:dry] OK (no finger motion commanded)")
            return

        if args.present_claw:                               # handoff: show the open claw while waiting
            claw0 = max(0.0, min(1.0, args.thumb_claw))
            print(f"[grip] presenting OPEN claw (thumb_aux={claw0:.2f}, fingers open) — place the sheet")
            br.set_left([0.0, claw0, 0.0, 0.0, 0.0, 0.0])

        if not args.no_wait and not _wait_for(args.arm_reached_file, args.hold_timeout, print):
            return

        # Motor order for set/get is [thumb_curl, thumb_aux, index, middle, ring, pinky]. thumb_aux is the
        # LATERAL thumb (0=slap/flat, 1=opposed across the palm = claw): you cannot pinch thin fabric with
        # a flat hand. So the thumb is opposed into the claw FIRST (and HELD there); the close below then
        # sequences the four fingers and the thumb FLEXION (thumb_curl) — see the CLOSE SEQUENCE comment.
        claw = max(0.0, min(1.0, args.thumb_claw))

        def cmd(g):  # uniform close fraction g for all digits, thumb opposed (thumb_aux=claw). Used for the
            return [g, claw, g, g, g, g]  # static OPEN present (g=0) and the HELD closed pose (g=grip_max).

        print(f"[grip] pre-positioning thumb → claw (thumb_aux={claw:.2f}), settling {args.claw_settle_s:.1f}s")
        br.set_left(cmd(0.0))                              # fingers open, thumb opposed
        time.sleep(args.claw_settle_s)

        # Empty-grab confirmation: baseline the OPEN claw now (thumb opposed, fingers open — nothing
        # loaded yet), then watch touch + proximity through the close to decide fabric-vs-air.
        gc = GraspConfirmMonitor(force_rise=args.confirm_force_rise, prox_dev=args.confirm_prox_dev,
                                 fingers_needed=args.confirm_fingers,
                                 use_proximity=args.confirm_use_proximity)
        gc.baseline(br.get())

        # CLOSE SEQUENCE (operator-specified 2026-06-19): the four FINGERS close COMPLETELY first while
        # the thumb stays OPPOSED but UNFLEXED, then — only after the fingers are fully closed + a settle
        # (--thumb-lag-s) — the thumb flexes in to clamp. The fingers catch on the BASE of the thumb if
        # its flexion overlaps theirs AT ALL (even slow), so this is TRUE sequencing, not a head start.
        # One smooth continuous flex per channel; touch polled ~10 Hz to feed the confirm. The grasp
        # decision is made entirely by the confirm monitor (below) on the HELD grip — this open-loop close
        # has no contact-gating of its own.
        dt = 1.0 / max(1.0, args.rate_hz)
        speed = args.grip_max / max(1e-3, args.close_s)   # finger close fraction per second
        thumb_start = args.close_s + args.thumb_lag_s     # thumb flexes ONLY after fingers fully close + settle
        t0 = time.time()
        next_poll = 0.0
        last_report = -1.0
        while True:
            elapsed = time.time() - t0
            gf = min(args.grip_max, speed * elapsed)                          # FINGERS close fully FIRST
            gt = min(args.grip_max, speed * max(0.0, elapsed - thumb_start))  # THEN the thumb flexes in
            br.set_left([gt, claw, gf, gf, gf, gf])        # thumb_curl=gt, thumb_aux=claw (held), fingers=gf
            if elapsed >= next_poll:                        # poll touch ~10 Hz to feed the confirm
                reading = br.get()
                gc.update(reading)
                force = list(reading.get("left_touch_force") or [0] * 5)
                next_poll = elapsed + 0.1
                if elapsed - last_report > 0.2:
                    print(f"[grip] fingers={gf:.2f} thumb={gt:.2f}  force={force}")
                    last_report = elapsed
            if gt >= args.grip_max - 1e-6:                  # thumb clamped LAST → claw fully closed, done
                gc.update(br.get())
                break
            time.sleep(dt)
        print("[grip] close complete (fingers, then thumb clamp) — judging fabric-in-hand")

        # HELD re-measure — PEAK != CAPTURE. A transient brush during the dynamic close is NOT a grab:
        # on a taut/anchored sheet a finger's force spikes then the fabric SLIPS OFF, collapsing to ~0 by
        # full close (false-success observed 2026-06-19). Hold the closed grip and sample so the verdict
        # keys on fabric STILL loaded, not a peak that already decayed.
        gc.begin_hold()
        hold_end = time.time() + args.confirm_hold_s
        while time.time() < hold_end:
            br.set_left(cmd(args.grip_max))                            # keep holding the closed position
            gc.update(br.get())
            time.sleep(0.05)

        # Empty-grab verdict — fabric-in-hand or closed-on-nothing? The pull-failure detector is BLIND
        # to this (a draw on air reads perfectly free), so confirm it here before signaling the draw.
        gv = gc.verdict()
        print(f"[grip] GRASP VERDICT: {gc.summary()}")
        if args.require_fabric and not gv["fabric_present"]:
            # Enforced: do NOT signal the draw on an empty hand. Write the empty-grab sentinel (the
            # ask-the-human trigger), release, and exit — the arm-side script breaks its grip-wait on
            # this file and skips the draw, and the orchestrator escalates exactly like /tmp/pull_failed.
            try:
                open(args.empty_file, "w").close()
            except OSError:
                # Best-effort sentinel; the release + return below run regardless.
                pass
            print(f"[grip] ⚠ EMPTY GRAB — closed on nothing. wrote {args.empty_file}; NOT signaling the "
                  f"draw. THIS is the trigger to ask the human for help.")
            br.set_left([0.0] * 6)
            print("[grip] released (fingers open)")
            return
        if not gv["fabric_present"]:
            print("[grip] (measure-only: empty-grab verdict logged but NOT enforced — pass "
                  "--require-fabric once thresholds are calibrated)")

        open(args.gripped_file, "w").close()
        print(f"[grip] signaled {args.gripped_file} — arm drawing. HOLDING + monitoring grip until {args.draw_done_file}")
        # MID-DRAW grip monitor — passing the pre-draw gate is NOT proof the grab survives the pull. A
        # light grab can hold at the verdict then SLIP OUT mid-draw, and the arm-side draw can't tell
        # (empty and free draws read identical follow/tau on hardware, 2026-06-19). So keep holding the
        # closed grip and watch the FINGERTIP force: if it collapses while drawing, the sheet left the
        # hand → write the empty sentinel so the draw aborts and the orchestrator escalates, instead of
        # the arm reporting a confident false-success on an empty hand.
        t_end = time.time() + args.hold_timeout
        lost_since = None
        while not os.path.exists(args.draw_done_file):
            if time.time() > t_end:
                print("[grip] hold timeout — releasing")
                break
            br.set_left(cmd(args.grip_max))                                  # keep holding the closed grip
            # Only the fabric-contact fingers count: the thumb pad self-contacts on the closed
            # claw (~600) and would mask any real grip loss, which is exactly why
            # GraspConfirmMonitor.fabric_indices excludes it.
            _force_rise = gc.update(br.get())["force_rise"]
            held = max(_force_rise[i] for i in gc.fabric_indices)  # strongest FABRIC finger vs open-claw baseline
            if held < args.draw_lost_rise:
                lost_since = lost_since or time.time()
                if time.time() - lost_since >= args.draw_lost_sustain:
                    print(f"[grip] ⚠ GRIP LOST mid-draw — fingertip force {held:.1f} < {args.draw_lost_rise} "
                          f"for {args.draw_lost_sustain:.1f}s; the sheet left the hand.")
                    if args.require_fabric:
                        try:
                            open(args.empty_file, "w").close()
                        except OSError:
                            # Best-effort sentinel; we break and release regardless.
                            pass
                        print(f"[grip] wrote {args.empty_file} — honest failure (not a false-success). "
                              f"THIS is the trigger to ask the human for help.")
                        break
                    print("[grip] (measure-only: grip-loss logged, NOT enforced — pass --require-fabric)")
                    lost_since = None                            # measure-only: re-arm, keep holding
            else:
                lost_since = None
            time.sleep(0.05)

        br.set_left([0.0] * 6)
        print("[grip] released (fingers open)")
    finally:
        br.close()


if __name__ == "__main__":
    main()
