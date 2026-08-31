### Added: `G1Driver` reads its FSM id from the motion-switcher API

Previously `G1Driver._fsm_id` had exactly one writer — the `None` initialiser
in `__init__` — so `_check_motion_gates` refused every `send_action`,
`run_policy` and `start_task` with `FSM id unknown - motion-switcher source
has not been wired`. The wire path the SDK exposes for the FSM state is
`MotionSwitcherClient.CheckMode()` (`LowState_` has no `fsm_id` field; every
SDK example uses the motion switcher's `CheckMode` return under `form`);
issue #2765 collected the wire-format decisions that read shape depends on,
and `strands_robots/tools/g1/_motion_switcher.py` landed the decoder half
in that thread.

The driver-side wiring lands here as the producer. `G1Driver` takes a new
keyword-only `motion_switcher_client_factory` constructor argument — a
callable that returns an open `MotionSwitcherClient`, defaulting to a lazy
loader that imports the SDK on first call — and a private `_refresh_fsm_id`
method that reads through `read_fsm_id` at the top of every
`_check_motion_gates` call. Three read-side branches are handled
explicitly: an OK reading writes `_fsm_id`, a `name == ""` reading (the
SDK's "no motion mode selected" state) clears the cache so a stale value
from before `ReleaseMode()` does not silently open the gate, and a refused
reading leaves `_fsm_id` at its previous value so a transient
`CheckMode()` failure on the tenth step of a rollout does not clobber the
id the previous nine wrote.

`get_status` now surfaces the mode label, the last decoder refusal, and
the factory open error alongside the integer id, so a caller inspecting
the mesh peer sees the same information the gate reads. The default lazy
loader preserves the module-load hygiene invariant every G1 module
already carries: `unitree_sdk2py` is imported inside function bodies,
never at module load, so the driver still imports on Thor, on CI, and in
every unit test with a mocked bus.

The read runs off the control-loop thread. `CheckMode()` is a synchronous
DDS round trip, and `_ControlLoop._run` re-gates every step, so refreshing
inside the gate would have put an RPC inside a 2 ms budget — and one
transient transport failure would have blocked frame publication (and the
loop's `_stop_event` observation) for the SDK's whole RPC timeout while the
joints drooped. Instead `_ControlLoop` owns a second thread that refreshes at
`_FSM_REFRESH_HZ` (10 Hz), and the per-step re-gate reads the cache it fills
via `_check_motion_gates(..., refresh=False)`. The one-shot entry points
(`send_action`, `start_task`, `run_policy` admission) keep the eager read, so
the decision that opens the wire is still made on a fresh reading.

Because the per-step gate now consults a cache, the cache is bounded: an FSM
that the refresher has not confirmed for `_FSM_STALE_AFTER_READS` (10)
consecutive reads refuses the step, and the rollout exits with the named
`gate` reason and a zero-torque frame rather than publishing on a reading
that stopped being evidence. The bound is stated in missed reads, not
seconds, so retuning the cadence cannot silently retune how many transient
`CheckMode` failures the gate absorbs. `_ControlLoop.snapshot` reports
`fsm_refresh_hz` and `fsm_reads` so a poller can distinguish "the FSM is
fine" from "nothing has looked at the FSM since admission".

`_refresh_fsm_id` holds one lock over both opening the client and talking to
it. It is now reachable concurrently from the refresher thread and from an
agent thread in `send_action`; the check-then-act on
`_motion_switcher_client` would otherwise open and leak a second client, and
two `CheckMode()` calls would interleave on one request/response client the
SDK does not document as thread-safe. The lock is never taken on the
control-loop thread, so a refresher parked in an RPC delays another refresh
and a concurrent `send_action`, never frame publication.

The predecessor un-reachability test file (`test_g1_battery_floor_is_gated_behind_the_unwired_fsm.py`)
is replaced by `test_g1_battery_floor_reaches_with_wired_fsm.py`, which
grades the flipped reachability directly: a driver with a wired FSM and a
critical pack refuses for the *battery*, not the FSM. The acceptance test
`test_send_action_returns_success_on_a_healthy_driver_that_has_a_decoded_lowstate`
turns over from `strict=True` XFAIL into a passing cell — the same
mechanical checkpoint the predecessor's docstring promised the wiring
commit would fire.

Closes the driver-side half of harness#361 and issue #2765's "decidable
now" list. What still needs a G1 in the room: measuring which `form`
integer the firmware reports for `HANDSHAKE_FSMS` under `CheckMode`;
whether `ReleaseMode()` is required before an `rt/lowcmd` write (the SDK's
own G1 low-level example calls it, and this PR does not); and whether
10 Hz is the right refresh cadence against a real motion-switcher's RPC
cost. The cadence is reasoned rather than measured — fast enough that an
FSM transition ends a rollout within 100 ms, slow enough that the RPC is
not the frame budget — and it is a module constant from which the
staleness bound is derived, so retuning it against a measurement is one
edit and cannot change how many missed reads the gate tolerates.
