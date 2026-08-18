### Fixed

- **training**: the in-process run log now has a single writer, so a chatty stdout can no longer overwrite the root-logger records in it. The log holds stdout/stderr and logger output whole, and a trainer's "RUNNING != learning" verdict (`latest_step` / `learning` / `liveness_ok`) reads a healthy run correctly again.
