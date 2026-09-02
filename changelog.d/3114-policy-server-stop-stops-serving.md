### Fixed: a `PolicyServer` told to stop stops serving the clients it has

`PolicyServer.stop()` closed the listening socket and joined the accept loop,
which is only half of a teardown. Through websockets 16.x,
`websockets.sync.server.Server.shutdown()` closed that socket and nothing else,
and each accepted connection was served on a thread that outlived the server
object -- so `stop()` returned while a client that was already connected went on
streaming observations in and receiving action chunks back. Measured on
websockets 16.1.1: `stop()` returned in 0.18ms and the same open connection was
answered with actions 19 more times over the following second. On a robot that is
the policy still driving the arm after the operator was told the server stopped.
Returning from the foreground `serve()` had the same gap.

websockets 17.0 closes the connections it accepted (code 1001) and returns from
`shutdown()` only once every connection handler has terminated, which is exactly
the teardown `PolicyServer` documents. The `[inference]` and `[cosmos3-service]`
floors therefore move from `websockets>=13.0` to `websockets>=17.0`: the property
is a guarantee of the dependency, so the floor is where it is expressed, rather
than re-implemented beside a version that already provides it. Measured against
the released wheels on unchanged sources -- 16.1.1 still serves that client, 17.0
does not. An install pinned below 17.0 now fails to resolve instead of quietly
shipping a teardown that does not stop the robot.

Nothing caught the gap because the lifecycle tests graded the server's own state:
`test_stop_is_idempotent` and `test_context_manager_starts_and_stops` assert
`_server is None`, which is equally true of a server that is still serving. The
new `tests/inference/test_a_stopped_server_stops_serving_its_clients.py` grades
what a *client* observes through both teardown doors, and pins that a stop landing
mid-inference waits for that call rather than racing the chunk it has not produced
yet; `tests/test_websockets_floor_ships_the_imported_api.py` now derives the
declared floor from behaviours as well as imported names, so a downgrade below
17.0 fails there rather than on a bench.
