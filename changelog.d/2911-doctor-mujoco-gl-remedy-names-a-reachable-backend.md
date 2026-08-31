### Fixed: the MUJOCO_GL remedy names a backend this host can load

`check_mujoco_gl` already reasons about not recommending a value MuJoCo would refuse - its own comment
says "what to recommend has to be valid here", and a sibling case pins that a Darwin remedy never
offers a Linux backend. That validated the advice against the platform's vocabulary, and valid is not
the same as reachable: each offscreen backend loads a system library, so a host can accept a value
whose library is not installed.

Measured on a headless Linux host with `libEGL.so.1` and no `libOSMesa.so`, the verdict was right and
its remedy was not. `MUJOCO_GL=off` earned `FAIL  MUJOCO_GL=off disables MuJoCo's GL context` and then
offered `export MUJOCO_GL=egl`, correct only because EGL sorts first; `MUJOCO_GL=glfw` earned a `WARN`
and advised `Set MUJOCO_GL=egl or osmesa for headless`, naming a backend that cannot render there; and
with neither library installed every one of those remedies offered an export where nothing exportable
can render, because the real remedy is the library.

`_configure_gl_backend` has always probed exactly this - it is how the package picks a backend at
import - and the verdict did not read that probe. `_mujoco_gl_loadable_offscreen_values` is that probe
as a helper, over one table of the `(backend, library)` pairs the configurator itself probes, so the
two readers of that table cannot disagree about what a host can reach. The verdict now recommends from
the reachable set and keeps the platform set only to tell "this platform has no offscreen backend"
apart from "this host is missing their libraries", which earns the library name instead of a variable.
The verdict a setting earns is unchanged; only the remedy moved.

The sibling suite's `linux_headless` fixture staged the platform and not the libraries, so its remedy
assertions read whichever libraries the machine running them happened to have - agreeing about the
verdict while disagreeing about the advice. It now stages both axes, which is what its docstring
already claimed, and a new case grades the probe against the loader directly rather than against the
stand-in every host model installs.
