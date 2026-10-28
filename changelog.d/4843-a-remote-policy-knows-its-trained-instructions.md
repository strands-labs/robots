### Added: a remote policy refuses an instruction it was never trained on, naming the trained ones

`Policy("zenoh://…", tasks=[...])` — or a server whose descriptor carries `tasks` — makes `run(instruction=)` with any
other string a `ValueError` listing what the model was trained to do, raised before the first frame goes out;
`policy.tasks` exposes them. A chunked policy fine-tuned on a few dozen exact strings (a tic-tac-toe deployment: 18
place strings and a clean-up) returns a confident plan for `"juggle"` too; the refusal replaces a hand-kept allow-list
in user code. With no `tasks` known, any instruction is sent as before.
