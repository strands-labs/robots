### Fixed: a Reachy Mini link teardown drops the handle its "am I connected?" guard reads

`ReachyMiniDriver.disconnect()` left the stopped link in `_hw` and
`WebSocketLink.stop()` left the closed socket in `_ws`, so neither send path's
not-connected guard could see a teardown. A movement RPC issued after a
disconnect was therefore not refused: on the Wireless variant it published to
`<prefix>/command` and reported success, actuating the head after the driver was
told to let go of it, and on the Lite variant it reached a closed socket as a
`ConnectionClosed` raised out of a method documented to be a no-op. Each
teardown now drops its handle before awaiting the release, so a release that
itself fails still leaves the object refusing commands.
