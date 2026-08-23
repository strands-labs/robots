### Fixed: a LIBERO driver's HF-token preflight resolves what the Hub resolves

The three `examples/libero` drivers refuse `--policy groot` up front when no
HuggingFace token is available, rather than letting the gated Cosmos-Reason2-2B
download fail later inside `gr00t_inference`. That refusal read
`~/.cache/huggingface/token` directly, so on any host that relocated its HF cache
it named a file the Hub will not open: a box that was logged in got refused, three
drivers out of three. It now asks `huggingface_hub.get_token()` and quotes
`HF_TOKEN_PATH` in the refusal, so a relocated `HF_HOME` / `XDG_CACHE_HOME` /
`HF_TOKEN_PATH` is honoured and, when there really is no token, the message names
the path it looked in.

The remedy it offered could not be run either. `huggingface-cli` is not published
as a console script at the declared `huggingface_hub>=1.5` floor at all - 1.5.0's
`console_scripts` are `hf` and `tiny-agents` - and the later 1.x releases that do
install it exit "deprecated and no longer works". The drivers now prescribe
`hf auth login`, which is what `doctor`, `dataset_recorder`, the README,
`docs/troubleshooting.md` and `huggingface_hub.get_token`'s own docstring already
name.

`run_mujoco_agent.py` carried the check inline in its lifecycle block rather than
as a resolver, and never read `HF_TOKEN` at all, so an `HF_TOKEN`-only host - the
environment the drivers' own prose calls preferred for CI - was refused outright.
It now shares the same resolver as its two siblings.
