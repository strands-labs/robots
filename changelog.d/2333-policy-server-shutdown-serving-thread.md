### Fixed

`PolicyServer.stop()` no longer kills the thread serving the server. `stop()`
closes the listening socket from the caller's thread while the accept loop is
still waiting on it, and the websockets sync server raises out of
`serve_forever()` when that happens. Both shapes come out of one call -
registering the listening socket with the selector - and which one surfaces
depends on where in it the close lands: before `fileno()` is read the descriptor
is `-1` and it raises `ValueError: Invalid file descriptor: -1`, and between that
read and `epoll_ctl` the descriptor still looks live and it raises
`OSError: [Errno 9] Bad file descriptor`. 12.0 lets both escape; 13.0 through
17.x guard the call against `ValueError` only, so every supported release can
still die on the `OSError` and raising the dependency floor cannot fix this. The serving thread is a daemon, so its death was reported nowhere: `stop()`
returned normally and the server looked cleanly shut down (measured 12 of 20
`start()`/`stop()` cycles on websockets 12.0, 3 of 12 on 17.0.1). The accept loop
now runs through a helper that absorbs the failure only while a stop is in
progress, so a socket failure with no stop pending still propagates.
