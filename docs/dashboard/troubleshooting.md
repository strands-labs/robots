# When it does not work

Every entry here is a message you can actually get, quoted closely enough to
match what you paste into a search box. Each one is the software refusing on
purpose or telling the truth about hardware - the fix is in the text more often
than it looks.

## Startup

**`dashboard: port 8090 is already in use by <pid/command> - refusing to start a
second instance (--force to override, ...)`**

A dashboard is a mesh peer. A second one on the same port is not a duplicate
window, it is a second hub partitioning your fleet - so the launch is refused
*before* any mesh session opens, and the message names the owner when it can
find it. Talk to the running one, or stop it. `--force` exists for when you know
better.

**`strands_robots: torchcodec needs Homebrew ffmpeg on the dyld path to decode
video. Set it before launching Python: export
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`**

macOS only, printed at import. Video decoding (dataset replay, streaming
episodes with frames) needs it; everything else works without it. Proprio-only
streaming via `drop_videos=True` needs no ffmpeg at all. Child processes inherit
the variable, so exporting it once in the shell that launches the dashboard
covers every peer it spawns.

## Cameras

**`ValueError: Camera 'main' config must be a mapping of option name to value,
got int: 3.`**

A camera is declared as a mapping of options, never a bare index:

```python
cameras={"wrist": {"index_or_path": 1}, "top": {"index_or_path": 0}}
```

An unknown key is refused rather than dropped, deliberately: a silently
discarded option would report success while the camera streamed at the default.

**A tile says `camera busy - another app is holding this device`**

Exactly what it says. On macOS, Photo Booth, QuickTime, Zoom, Chrome's camera
permission prompt, or a *previous run of your own robot* holds the device. USB
cameras are single-claim; close the other app.

**A tile is greyed out with the last frame still visible**

The stream stalled - those are old pixels, and the UI dims them rather than
letting a frozen image pass as live. Not a rendering bug: check the peer.

**The wrong camera is labelled "wrist"**

Indices are assigned by enumeration order, and they move when you replug or
power-cycle. Wave at one camera and see which tile moves, then fix
`index_or_path`. There is no way to infer this from the device name - two
identical `USB2.0_CAM1` entries are exactly what two SO-101 wrist cameras look
like.

## Arms and teleop

**Leader and follower look swapped**

The dashboard cannot tell them apart electrically - both are Feetech buses. The
one whose motors are unpowered/back-drivable is the leader you hold; assign the
roles when you spawn, and give the peers names you can read
(`so101-leader`, `so101-follower`).

**`teleop/receive` seems to hang for ~20 seconds**

The first `declare_subscriber` on a peer includes zenoh gossip propagation and
can exceed 15s. The route budgets 45s for exactly this. Slow, not stuck - do not
kill it at 10 seconds and conclude the mesh is broken.

**A peer's card is greyed as `stale`, then vanishes**

Stale after 15s of silence; aged out after `STRANDS_DASHBOARD_PEER_TTL_S`
(default 300). A peer with a live managed process is never dropped, so a running
robot cannot be erased by a state-stream hiccup - if a card disappeared, that
process really is gone.

## Policies and checkpoints

**`Policy provider 'lerobot_local' loads HuggingFace models with
trust_remote_code=True ... set STRANDS_TRUST_REMOTE_CODE=1`**

Loading that checkpoint executes code from its repository. The refusal is the
feature. Opt in per shell, for orgs you actually trust:

```bash
STRANDS_TRUST_REMOTE_CODE=1 python -m strands_robots dashboard --port 8090
```

**`pretrained_name_or_path='Org/some-checkpoint' not in allowlist. Set
STRANDS_MESH_HF_REPO_ALLOW to add an org/repo prefix.`**

Not a missing file - a mesh command validator refusing to let a wire message
point your robot at an arbitrary model. Same family:
`policy_host=... not in allowlist` (`STRANDS_MESH_POLICY_HOST_ALLOW`),
`policy_type=... not in allowlist` (`STRANDS_MESH_POLICY_TYPE_ALLOW`), and
`model_path=... contains disallowed characters or path-traversal segments`.

```bash
STRANDS_MESH_HF_REPO_ALLOW="your-org/,lerobot/" python -m strands_robots dashboard
```

**A run "succeeds" but the policy behaved like the default**

Only allowlisted keys reach a peer, and everything else is dropped silently by
the wire validator. A nested `policy_config` posted to
`/api/robots/{peer}/task` is the common casualty - use the flat wire keys
(`pretrained_name_or_path`, `policy_type`, `policy_host`, ...). Full list in
[Collect, train, deploy](collect-train-deploy.md#4-deploy-the-checkpoint).

**A task reports `timeout` while the arm is clearly still working**

The peer answers only when the policy *finishes*. The route forces the timeout
to at least `duration + 10s`; if you call the API yourself with a smaller one,
you get a timeout on a healthy run.

## Datasets

**The Record panel shows a banner about a mock**

The mock only appears when the frontend's probe of `/api/record/session` 404s,
so the router did not mount. The record API is real at this head (see
`strands_robots/dashboard/record_api.py` and its mount in `server.py`); if the
banner appears, the useful troubleshooting entry is inverse - the server did
not attach the router. Check the dashboard's log for a load-time error on the
record module before assuming the panel is a mock.

**A dataset has far fewer frames than the time you spent**

`/api/training/datasets` reports parquet truth, not intent. Check `fps` and
`total_frames` per episode: a stopped teleop stream, a camera that never opened,
or an episode boundary you did not mean to cross all look like this.

**`robot_type: unknown` in the dataset list**

The dataset was recorded against something the registry could not name. It will
still train, but nothing downstream can check embodiment compatibility for you.

## Access

**`{"detail":"unauthorized"}` from a browser that is not on this Mac**

Working as designed. With no credential configured the API is loopback-only, and
any forwarding header (`x-forwarded-for`, `cf-connecting-ip`, `x-real-ip`) marks
the client as remote. See [Remote access](remote-access.md).

**The passkey gate says the origin cannot be used**

WebAuthn needs a secure context (HTTPS or `http://localhost`) and an `rp_id`
that is a hostname - a raw IP like `https://192.168.1.50` is refused by the
browser before the ceremony starts. `/api/auth/status` reports this in
`warning`, `secure_context` and `rpid_usable`.

**A resume after e-stop is refused**

`POST /api/safety/resume` needs `override_code`, verified locally and
brute-force throttled; the code itself never crosses the wire. Every peer must
share the same `STRANDS_MESH_OVERRIDE_CODE`, or each one stays locked until its
process restarts.

## Tests

**`ModuleNotFoundError: No module named 'psutil'` / `'msgpack'` during pytest
collection**

Optional dependencies missing from your venv, not a broken tree. They surface as
collection errors on ~17 files; the rest of the suite runs. Install them, or run
the file you care about with `--no-cov` (a single file otherwise trips the
global 80% coverage gate).

## Still stuck

Ask the dashboard itself - the agent dock can read the fleet state, and
`/api/devices/logs/{peer_id}` returns the stdout of a peer the dashboard
spawned, which is where a hardware exception actually lands.
