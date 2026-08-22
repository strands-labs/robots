### Fixed: a failed pipeline resolution is reported by the transform preflight, not raised at the caller

`CosmosTransferTransform` binds a caller-supplied video2video pipeline, so resolving the seam runs the
caller's code at three points: the module import and attribute lookup, the zero-arg construction of a
class or factory target, and the read of the object's `generate` surface. `validate()` is documented to
return a list of problems and `transform()` to return `status="error"` for anything but a backend
shape/dtype bug, but only `ImportError`, `AttributeError` and `TypeError` were reported - 51 of 52 builtin
exception classes raised out of both surfaces.

The classes that escaped are the ones a real generation stack raises, because constructing one loads
weights and touches a device: `ImportError` from an optional dependency imported inside a factory body
(so `ModuleNotFoundError` escaped a handler written for exactly that case, which wrapped only the module
import), `RuntimeError` with no driver, `OSError` with the weights absent, and `ValueError` on a malformed
config - the last indistinguishable from the shape/dtype `ValueError` that `transform()` documents as its
only raise. A lazy handle whose `generate` is a property raised at the third point, where `getattr`'s
default absorbs only `AttributeError`.

Each probe point now reports, with the cause named accurately: a factory that raised is constructible, so
it is not folded into the "not constructible zero-arg" wording, which would name the wrong remedy. The
`generate` read has one guarded owner. `Exception`, not `BaseException` - an operator's Ctrl-C during a
multi-gigabyte load is not a bad spec and still propagates. Every spelling that already reached a verdict
keeps it, wording included.
