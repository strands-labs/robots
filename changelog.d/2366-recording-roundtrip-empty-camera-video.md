### Fixed

The MuJoCo camera-recording round-trip test now fails when the recording lands a camera video that carries no bytes. Frame access raises `RuntimeError` both when no video decoder can be loaded and when the video has no readable stream, and the guard around it tolerated both, so an empty camera video passed every check. The recorded files are now asserted on disk before any decode, which needs no decoder and cannot be absorbed by a guard.
