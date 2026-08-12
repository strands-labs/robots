### Quality: pin the receive side of the teleop stream-replacement contract

`Robot.start_teleop_receive` keys its receiver registry on
`source_peer_id/device_name` and stops whatever is registered under that key
before installing the replacement. That step had no test: the publish mirror was
pinned with the rate guard, but the receive side only had its identifier
refusals, and `test_rejected_receive_leaves_a_live_stream_running`'s
"the live stream survived" assertion holds equally for a body that tears nothing
down. Four cases now drive the accepted path -- the superseded receiver is
stopped and replaced, its subscription is dropped rather than leaked, and the
compound key scopes the replacement so a second leader under the same
`device_name` leaves the first stream running. Tests only; no behaviour change.
