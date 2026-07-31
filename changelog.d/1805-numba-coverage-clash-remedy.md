### Fixed: the numba/coverage clash diagnostic now names a remedy that works

The LIBERO adapter recognises the `numba` / `coverage` import clash and, until
now, told the caller to pin `coverage` *down* - the opposite of the fix.
`coverage.types.Tracer`, which `numba.misc.coverage_support` subclasses with no
version guard, is provided only from `coverage` 7.6.1 onward (7.4.0-7.6.0 name
the same protocol `TracerCore`, 7.0-7.3 name it `TTracer`, and 6.x ships no
`coverage/types.py` at all), so the clash is a coverage-too-old condition.
Following the old advice made things worse: on `coverage<7` the failure text
becomes `module 'coverage' has no attribute 'types'`, which the detector did not
recognise, so the strict install error degraded into "GR00T actions will no-op" -
the silent action-dropping the strict classification exists to prevent.

The remedy now names `pip install 'coverage>=7.6.1'`, keeps removing `coverage`
entirely as the alternative (numba guards `import coverage` with
`except ImportError`), and no longer suggests a downgrade or a numba bump. It is
built in one place and reaches all three raise sites - the adapter's install
hint and both lazy-import guards in `_LiberoOSCController.from_sim`, which
previously reported only the symptom. The detector also recognises the
coverage-6.x failure shape, so an environment that already followed the old
advice is still diagnosed instead of silently degrading.
