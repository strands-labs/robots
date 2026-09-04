### Fixed

- The job-summary report of how much of the required suite ran reads the extent
  the session itself counted (`tests/session_truncation.py`) instead of
  re-deriving it from the log's text, and falls back to the arithmetic only for a
  log carrying no such section. The fallback now applies pytest's own rules --
  the number selected is `collected - deselected`, and the collection line's
  trailing tokens are read in any order -- so a run that reached every item it
  selected is no longer reported as truncated, and the number of items executed
  can no longer exceed the number collected.
