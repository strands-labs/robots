### Fixed: an Isaac eval driven from a worker thread fails loudly (or marshals to the pump) instead of deadlocking forever

`examples/libero/run_isaac_agent.py` deadlocked indefinitely: the Strands
`Agent(...)` call executed its tool on a worker thread, the tool ran
`evaluate_benchmark` -> `reset()` -> Isaac `SimulationContext.stop()`, and that
call never returned because kit updates are only pumped from the thread that
created `SimulationApp` - which was itself parked inside `Agent.__call__`
waiting on the tool future. The example now uses the documented threading
model: the agent runs on a side thread, the main thread owns the kit pump
(`run_pump_forever(stop_event=...)`), and the tool wrapper submits the eval
body back to the owning thread via `run_on_main` - the same pump-on-main +
worker-callers architecture as `examples/isaac_gs/app.py`.

Defense in depth in the library: `IsaacSimulation.reset()` / `.step()` called
off the owning thread are now auto-marshalled through `run_on_main` when
`run_pump_forever` is engaged (the recording facade's schema-probe pattern),
and raise an actionable `RuntimeError` naming the pump/`run_on_main` recipe
when it is not - turning an indefinite, signal-free deadlock into an
immediate error. Worker-thread `evaluate_benchmark` / `run_policy` calls are
covered transitively through their per-episode `reset()`.
