### Changed: the `strands-agents` floor rises to 1.13.0, the release that makes `BeforeToolCallEvent` usable

`strands.hooks.BeforeToolCallEvent` is imported at module scope by
`strands_robots.dashboard.agent_hitl`, the human-in-the-loop gate that pauses an
agent tool call before real hardware moves. Measured against every released 1.x
wheel, the name and the API the gate uses on it do not ship together:

- **1.10.0** exports the class from `strands.hooks`. Below that the name is
  absent rather than moved, so on 1.9.1 -- a release the old `>=1.7.0` floor
  admitted -- importing that module raises `ImportError: cannot import name
  'BeforeToolCallEvent' from 'strands.hooks'`.
- **1.13.0** is where `interrupt` enters the event's method resolution order
  (via `HookEvent` and `_Interruptible`), and where `cancel_tool` becomes a
  field the SDK's tool executor translates into a tool-result error. Those are
  the two members the gate actually calls.

The floor is therefore 1.13.0, not 1.10.0. The band between them is the one
worth naming: the class is exported, the import succeeds, nothing refuses at
resolve time or at start-up, and `event.interrupt(...)` raises `AttributeError`
the first time a tool call would move a real robot. A floor recording only where
the *name* arrived would admit exactly that band, which is a worse outcome than
refusing the install.

The bound is declared in `project.dependencies`, in the `[ollama]` extra that
re-declares it, and in `uv.lock`'s transcription of both. The resolved
`strands-agents` in the lock is unchanged, so no dependency version moves.
`tests/test_strands_agents_floor_ships_the_imported_api.py` owns the measurement
and now also grades the members the gate calls, so a release that exports the
name without them cannot pass for one that ships the capability.

Two written install hints that restated the old floor move with it: the doctor's
remedy for "strands-agents not importable" (`strands_robots/doctor.py`) and the
agent/UI install line in `examples/so101_curobo/README.md`, which still named
`>=1.0` and `>=0.1` respectively. A remedy below the real floor is the failure
mode worth closing rather than a cosmetic mismatch -- it is already satisfied by
a release this package refuses, so pip answers "Requirement already satisfied",
upgrades nothing, and the reader is left running the command that printed it.
Raising the floor without them would have widened that gap rather than left it
where it was. Measured against the 60 published `strands-agents` 1.x releases,
`>=1.0` admits 8 that the old `>=1.7.0` floor refused and 16 that `>=1.13.0`
refuses -- so the floor raise doubles the remedy's error on its own. The
README's `>=0.1` admits those 16 plus all 14 published `0.x` releases. Three of
the 16 are the worst kind for this particular remedy -- 1.10.0, 1.11.0 and
1.12.0 import `agent_hitl` cleanly and fail at gate time instead -- so the hint
printed when `strands` is unimportable would have pointed at a range in which
the gate this floor exists to protect is itself unusable.
