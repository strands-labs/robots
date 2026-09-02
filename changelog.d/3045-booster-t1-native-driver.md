### Added: native driver for the Booster Robotics T1

`Robot("booster_t1", mode="real")` could not be built at all: the T1 has no
lerobot robot type, so the default lerobot driver raised "Unsupported robot type"
and listed the sixteen robots lerobot does know. `BoosterDriver` drives it
through the vendor SDK (`booster_robotics_sdk_python`, a pybind11 wrapper over
the robot's DDS transport), and the registry entry declares
`hardware.driver="strands"` so no `driver=` keyword is needed.

Control of the T1 is split, and the driver enforces the split rather than
documenting it. An onboard whole-body controller owns the legs, waist and head;
the eight arm joints (slots 2-9) can be handed to a host, and only after
`UpperBodyCustomControl`. Every frame the driver publishes therefore carries
`kp=kd=0` on all fifteen other slots - zero gain is what leaves the onboard
controller in charge of balance - and an uncommanded arm joint holds its last
*observed* position rather than zero, so commanding one arm does not swing the
other to its zero pose. `send_action` refuses before the gate is open, before the
first `LowState` has arrived (the frame width and the hold positions both come
from the robot's own report), while the robot reports a fall state other than
`IS_READY`, and for a joint outside the upper body - naming `move()` or
`rotate_head()`, which reach it through the controller that owns it.

Every wire literal is transcribed from the vendor's own reference client shipped
inside the SDK wheel: `mode=0x0A` is its position mode, `kp=60`/`kd=3` its
upper-body gains, zero-gain legs its rule. The joint table and the fall-state
codes are graded against the SDK's `JointIndex` and `FallDownStateType` enums, so
a vendor renumbering fails a test instead of moving a robot. The battery read
reaches `get_status()` as the shared `battery_pct` field and gates nothing: the
SDK names it `soc` and documents no scale, and a floor compared against an
unverified scale refuses every frame or none while looking like a working check.

The `domain_id` the driver opens its DDS channels on is answered by the shared
domain every other DDS surface in the project uses (`dds_domain_id_error`), not by
a check local to the driver. A domain id indexes the RTPS port map, so it has a
ceiling as well as a floor: the ports for domain 233 do not fit the 16-bit port
space, and a driver that accepts it hands `ChannelFactory.Init` a domain nothing
can be reached on while the telemetry bridges that advertise the same robot's
topics refuse it. Sharing the guard also means one wording for the mistake at
every surface, and that wording states the range.

The SDK is a vendor wheel rather than a declared dependency - the same footing as
the G1's `unitree-sdk2` - so it is imported inside function bodies and its
absence is a named refusal from `connect_eagerly()`, never an `ImportError` at
import.
