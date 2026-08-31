### Fixed: the Booster T1 driver constructs its DDS endpoints under the shared lock

`BoosterDriver.connect_eagerly` built the channel factory, the `B1LocoClient` and
its four channels with no serialisation against the rest of the process.
Constructing one CycloneDDS endpoint while another is under construction
segfaults the bindings, and `DDSSubscriberSet.subscribe` creates every subscriber
under `_DDS_INIT_LOCK`, so anything else in the process touching DDS raced this
driver. That loss is not catchable by the surrounding "record the reason and stay
usable for reads" boundary: the process dies, possibly while a 1.2 m biped is
standing under its own controller. Measured with the real subscriber set on one
thread and the real connect on the other, 40 against 40, there were 80
overlapping construction pairs; there are now none.

The construction block is wrapped in `_DDS_INIT_LOCK`, matching
`Go2Driver._motion_switcher_client`. The lazy SDK import stays outside it - it
creates no endpoint, and holding a process-wide lock across an import would stall
every subscriber construction for its duration - and so does the teardown of a
partial channel set, matching `DDSSubscriberSet.close`, which releases under its
own bookkeeping lock: the shared lock serialises construction, not release. No
lock is nested, so no new ordering is introduced.

`tests/tools/g1/test_every_dds_endpoint_is_created_under_the_shared_lock.py`
already refused the two call sites. Because it matches on the callee's name, and
a call laundered through a lower-case local would evade it, the driver's own test
module now also pins the behaviour: a competing holder of the shared lock makes
`connect_eagerly` block with no endpoint constructed, which is what distinguishes
taking the shared lock from taking a private one that would exclude nothing.
