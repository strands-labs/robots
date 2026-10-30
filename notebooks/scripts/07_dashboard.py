# Paired script for notebooks/07_dashboard.ipynb. Run: .venv/bin/python notebooks/scripts/07_dashboard.py
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # nb: script-only

# %% [markdown]
# # 07 · The dashboard
#
# `Dashboard(sim)` serves one world to a browser tab: a live 3D twin, a card per robot, an agent chat and an e-stop.
# The page reads the world itself (under `sim.lock`), so anything a script does shows up with nothing to sync.
#
# ![a script and a browser tab share one world](media/07/one_world_two_views.svg)

# %% [markdown]
# ## Serve a world
#
# Two robots, two objects and a camera. `start()` runs the server on a background thread and returns, `serve()` blocks.
# Passing `model=` turns the chat panel on.

# %%
import os
from strands_robots import Robot, Simulation
from strands_robots.dashboard import Dashboard

sim = Simulation()
arm = Robot("so101", sim=sim)
g1 = Robot("g1", sim=sim, position=[1.0, 0.0, 0.0])
sim.add_object("cube", shape="box", position=(0.25, 0.05, 0.02), size=(0.02, 0.02, 0.02), color=(0.85, 0.1, 0.1, 1))
sim.add_object("ball", shape="sphere", position=(0.22, -0.12, 0.025), size=0.025, color=(0.1, 0.35, 0.9, 1))
sim.camera("front", position=(0.9, -0.9, 0.7), target=(0.3, 0.0, 0.2), size=(320, 240))

model = None
if os.environ.get("AWS_BEARER_TOKEN_BEDROCK") and os.environ.get("STRANDS_MODEL_ID"):
    from strands.models import BedrockModel

    model = BedrockModel(model_id=os.environ["STRANDS_MODEL_ID"])

dash = Dashboard(sim, model=model)
server = dash.start(port=8731)  # loopback only; port=0 picks a free one
print(server.url, "| robots:", list(sim.robots), "| chat:", "on" if model else "off (no Bedrock credentials)")

# %% [markdown]
# ## Open it in a browser
#
# Headless Chromium stands in for your browser. Playwright's sync API refuses to run inside Jupyter's event loop,
# so it gets its own thread. WebGL runs on SwiftShader (software), which is slow but draws the same pixels.
# Every page records its console, and every screenshot reports the errors it saw.

# %%
import json, queue, threading, time
import _nb

MEDIA = _nb.MEDIA / "07"
MEDIA.mkdir(parents=True, exist_ok=True)


class Chromium:
    """One headless Chromium on its own thread; ``call(fn)`` runs ``fn(browser)`` there and returns its result."""

    def __init__(self):
        self._jobs, self._done = queue.Queue(), queue.Queue()
        self._thread = threading.Thread(target=self._main, daemon=True)
        self._thread.start()
        self.call(lambda b: None)

    def _main(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            while (job := self._jobs.get()) is not None:
                try:
                    self._done.put((True, job(b)))
                except Exception as exc:  # noqa: BLE001 - raised again on the caller's thread
                    self._done.put((False, exc))
            b.close()

    def call(self, fn, timeout=300):
        self._jobs.put(fn)
        ok, value = self._done.get(timeout=timeout)
        if not ok:
            raise value
        return value

    def close(self):
        self._jobs.put(None)
        self._thread.join(timeout=30)


chromium = Chromium()
pages, logs, seen = {}, {}, {}
PAPER = "localStorage.setItem('__palette', JSON.stringify({color: {scheme: 'paper'}}))"  # the docs site's palette key


def painted(page, loads=1):
    page.wait_for_function(f"window.__sceneLoaded >= {loads} || window.__error", timeout=180_000)
    page.wait_for_function("window.__twin && window.__twin.frames > 5", timeout=60_000)
    page.wait_for_function("window.__twin.fps > 0", timeout=60_000)  # the render loop has drawn for a second


def open_page(name, url, width, height, scheme="dark", mobile=False, wait=painted):
    def job(b):
        ctx = b.new_context(viewport={"width": width, "height": height}, is_mobile=mobile, has_touch=mobile)
        if scheme == "paper":
            ctx.add_init_script(PAPER)
        page = ctx.new_page()
        log = logs.setdefault(name, [])
        page.on("console", lambda m: log.append((m.type, m.text)))
        page.on("pageerror", lambda e: log.append(("pageerror", str(e))))
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        if wait:
            wait(page)
        pages[name] = page
        return {"page": name, "scheme": page.evaluate("document.documentElement.dataset.scheme"),
                "mode": page.evaluate("window.__state ? window.__state.mode : null"),
                "ready_s": round(time.perf_counter() - t0, 1)}

    return chromium.call(job)


def close_page(name):
    chromium.call(lambda b: pages.pop(name).context.close())  # SwiftShader draws every open page on the CPU


def on(name, js, arg=None):
    return chromium.call(lambda b: pages[name].evaluate(js, arg))


def shot(name, file, clip=None):
    """Screenshot into media/07/; returns the console errors and warnings logged since this page's last shot."""
    from PIL import Image
    import numpy as np

    path = MEDIA / file
    for _ in range(4):  # a stage of one flat colour means WebGL has not drawn yet (SwiftShader under load): wait, retry
        chromium.call(lambda b: pages[name].screenshot(path=str(path), clip=clip))
        img = np.asarray(Image.open(path).convert("L"))
        h, w = img.shape
        drawn = float(img[h // 6: h * 5 // 6, w // 12: w * 7 // 12].std()) > 2.0
        if drawn:
            break
        chromium.call(lambda b: pages[name].wait_for_timeout(3000))
    new = logs[name][seen.get(name, 0):]
    seen[name] = len(logs[name])
    errors = [t for k, t in new if k in ("error", "pageerror")]
    warnings = [t for k, t in new if k == "warning"]
    return {"file": file, "kb": path.stat().st_size // 1024, "drawn": drawn, "errors": len(errors), "warnings": len(warnings),
            "first": (errors or warnings or [""])[0][:90]}


print(open_page("desktop", server.url, 1440, 900))
first = shot("desktop", "desktop_dark.png")
print(first)

# %% [markdown]
# ![desktop, dark](media/07/desktop_dark.png)
#
# The pill reads `server`: this tab draws the Python process's world. Each robot gets a card with one slider per
# joint, the camera `front` gets a card with what a policy would see, and the chat sits under the cards.

# %% [markdown]
# ## Light and phone layouts
#
# The page is dark by default. It turns paper only when the docs site's palette says so (Material keeps it in
# `localStorage` under a key ending in `__palette`), which is what the light shots set. Under 56 em wide the drawer
# becomes a bottom sheet with tabs.

# %%
rows = [first]
print(open_page("desktop_paper", server.url, 1440, 900, scheme="paper"))
rows.append(shot("desktop_paper", "desktop_paper.png"))
close_page("desktop_paper")
for scheme in ("dark", "paper"):
    name = f"phone_{scheme}"
    print(open_page(name, server.url, 440, 956, scheme=scheme, mobile=True))
    rows.append(shot(name, f"phone_{scheme}.png"))
    chromium.call(lambda b: pages[name].click("#toggle"))
    chromium.call(lambda b: pages[name].wait_for_timeout(600))
    rows.append(shot(name, f"phone_{scheme}_drawer.png"))
    close_page(name)
_nb.show(_nb.table(rows))

# %% [markdown]
# | desktop, paper | phone, dark | phone, drawer open | phone, paper |
# |---|---|---|---|
# | ![](media/07/desktop_paper.png) | ![](media/07/phone_dark.png) | ![](media/07/phone_dark_drawer.png) | ![](media/07/phone_paper_drawer.png) |

# %% [markdown]
# ## What the page is built from
#
# `GET /api/scene` is the merged MJCF plus a card per robot; `WS /ws/state` streams joint values and root poses.
# The browser compiles the MJCF with MuJoCo WebAssembly for kinematics only: the server's world is the truth.

# %%
import http.client


def api(method, path, body=None, headers=None, port=8731):
    """One request to the dashboard, returning (status, json); headers= can override Host, Origin, Content-Type."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    data = None if body is None else (body if isinstance(body, (bytes, str)) else json.dumps(body))
    hdrs = {"Content-Type": "application/json"} if body is not None else {}
    hdrs.update(headers or {})
    conn.request(method, path, body=data, headers=hdrs)
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw  # a camera PNG


status, scene = api("GET", "/api/scene")
print(status, sorted(scene))
print("robots:", [(c["id"], c["name"], len(c["joints"])) for c in scene["robots"]])
card = next(c for c in scene["robots"] if c["id"] == "so101")
JOINT = dict(zip(card["labels"], card["joints"]))  # action key -> the browser model's joint name
print("so101 joints in the browser model:", JOINT)
print("objects:", scene["objects"], "| cameras:", scene["cameras"])
print("physics:", scene["physics"], "| stats:", scene["stats"], "| assets:", len(scene["assets"]))
status, png = api("GET", "/api/cameras/front.png")  # sim.render("front"): what a policy sees, as PNG
(MEDIA / "camera_front.png").write_bytes(png)
print(status, "camera_front.png", len(png) // 1024, "KB")

# %% [markdown]
# ![GET /api/cameras/front.png](media/07/camera_front.png)

# %% [markdown]
# ## A server-side act moves the browser twin
#
# `arm.act(..., seconds=1.0)` ramps the arm from Python. Nothing is sent to the page: it sees the new joint values
# in the next state frame. In a world `seconds` is simulated time, so the call returns as fast as physics steps.

# %%
target = {"shoulder_pan": 1.0, "shoulder_lift": -0.6, "elbow_flex": 0.8, "wrist_flex": 0.4}
before = {j: on("desktop", "j => window.__twin.qpos(j)", JOINT[j]) for j in target}
print(shot("desktop", "act_before.png"))
t0 = time.perf_counter()
arm.act(target, seconds=1.0)
print(f"arm.act(seconds=1.0) returned after {time.perf_counter() - t0:.3f} s wall, sim t={sim.time:.2f} s")
server_q = {j: arm.observe(images=False)[j] for j in target}
chromium.call(lambda b: pages["desktop"].wait_for_function(
    "([j, v]) => Math.abs(window.__twin.qpos(j) - v) < 0.01", arg=[JOINT["shoulder_pan"], server_q["shoulder_pan"]],
    timeout=30_000))
after = {j: on("desktop", "j => window.__twin.qpos(j)", JOINT[j]) for j in target}
print(shot("desktop", "act_after.png"))
_nb.show(_nb.table([{"joint": j, "page before": round(before[j], 3), "server after": round(server_q[j], 3),
                     "page after": round(after[j], 3)} for j in target]))

# %% [markdown]
# | before | after `arm.act(..., seconds=1.0)` |
# |---|---|
# | ![](media/07/act_before.png) | ![](media/07/act_after.png) |

# %% [markdown]
# ## The twin during a run
#
# `arm.start(policy, ...)` runs the loop on a background thread. A plain function is a policy. While it runs, the
# browser's own screencast records what it draws; the frames become an animated WebP.

# %%
import base64, io, math
from PIL import Image


def wave(obs):
    s = math.sin(2 * math.pi * obs.t / 3.0)
    return {"shoulder_pan": 0.9 * s, "shoulder_lift": -0.5 + 0.3 * s, "elbow_flex": 0.7 - 0.4 * s, "wrist_roll": 1.2 * s}


def screencast(seconds):
    """Chromium's own screencast (CDP Page.startScreencast): every frame the compositor draws, for ``seconds``."""

    def job(b):
        page = pages["desktop"]
        cdp, got = page.context.new_cdp_session(page), []

        def frame(ev):
            got.append((ev["metadata"]["timestamp"], ev["data"]))
            cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

        cdp.on("Page.screencastFrame", frame)
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 85, "maxWidth": 640, "maxHeight": 420})
        page.wait_for_timeout(seconds * 1000)
        cdp.send("Page.stopScreencast")
        cdp.detach()
        return got

    return chromium.call(job)


chromium.call(lambda b: pages["desktop"].click("#toggle"))  # hide the drawer: the twin fills the window
chromium.call(lambda b: pages["desktop"].set_viewport_size({"width": 640, "height": 420}))  # fewer pixels per frame
chromium.call(lambda b: pages["desktop"].wait_for_timeout(1500))
task = arm.start(wave, seconds=9.0, realtime=True)
raw = screencast(8)
result = task.stop()
chromium.call(lambda b: pages["desktop"].set_viewport_size({"width": 1440, "height": 900}))
chromium.call(lambda b: pages["desktop"].click("#toggle"))
raw = raw[:: max(1, len(raw) // 60)]  # at most ~60 frames keeps the GIF small
frames = [Image.open(io.BytesIO(base64.b64decode(d))).convert("RGB") for _, d in raw]
keep = [i for i, f in enumerate(frames) if f.size == frames[-1].size]  # drop frames drawn before the resize landed
frames, raw = [frames[i] for i in keep], [raw[i] for i in keep]
gaps = [max(40, round((b - a) * 1000)) for (a, _), (b, _) in zip(raw, raw[1:])] + [400]
gif = MEDIA / "twin_run.webp"  # animated WebP: ~15x smaller than the same GIF; GitHub and Jupyter play it
frames[0].save(gif, save_all=True, append_images=frames[1:], duration=gaps, loop=0, quality=80, method=4)
print({k: getattr(result, k) for k in ("steps", "stopped_by", "deadline_misses", "fault")})
print(f"{len(frames)} screencast frames over {raw[-1][0] - raw[0][0]:.1f} s -> {gif.name} ({gif.stat().st_size // 1024} KB)")

# %% [markdown]
# ![the browser twin following arm.start(wave)](media/07/twin_run.webp)
#

# %% [markdown]
# ## The world changes, the page follows
#
# A robot added over HTTP (`POST /api/robots`) and an object added from Python (`sim.add_object`) both bump
# `sim.revision`. The page sees the new revision in a state frame and recompiles.

# %%
loads = on("desktop", "() => window.__sceneLoaded")
print(api("POST", "/api/robots", {"name": "so101", "id": "arm2", "position": [0.0, 0.45, 0.0]}))
sim.add_object("crate", shape="box", position=(0.3, 0.3, 0.03), size=(0.03, 0.03, 0.03), color=(0.0, 0.8, 0.38, 1))
chromium.call(lambda b: painted(pages["desktop"], loads + 1))
chromium.call(lambda b: pages["desktop"].wait_for_function(
    f"window.__twin.revision === {sim.revision}", timeout=120_000))
print("revision", sim.revision, "| page reloads", on("desktop", "() => window.__sceneLoaded"),
      "| cards", on("desktop", "() => [...document.querySelectorAll('.card')].map(c => c.dataset.id || 'camera')"))
print(shot("desktop", "world_changed.png"))

# %% [markdown]
# ![a second arm from HTTP, a green crate from Python](media/07/world_changed.png)

# %% [markdown]
# ## Run endpoints
#
# The run row drives the same `robot.start` over HTTP: `POST /api/robots/{id}/run` starts a policy, `GET` reads its
# status and `POST .../stop` ends it. `hold` is built in; any `Policy(spec)` string works too.

# %%
status, pol = api("GET", "/api/policies")
print(status, len(pol["policies"]), "policies:", [p["name"] for p in pol["policies"]][:3], "...")
print("spec forms Policy() accepts:", " | ".join(f.split("  ")[0].strip() for f in pol["forms"]))

def run_api(method, path, body=None):
    st, doc = api(method, path, body)
    if isinstance(doc.get("result"), dict):
        doc["result"].pop("hz_achieved", None)  # a wall-clock rate from this machine is not a claim worth printing
    return st, doc


print(run_api("POST", "/api/robots/so101/run", {"policy": "hold", "seconds": 1.0}))
print(run_api("GET", "/api/robots/so101/run"))
time.sleep(2.5)
print(run_api("GET", "/api/robots/so101/run"))
print(run_api("POST", "/api/robots/arm2/run", {"policy": "hold", "seconds": None}))
print(run_api("POST", "/api/robots/arm2/run", {"policy": "hold", "seconds": None}))  # one task per robot
print(run_api("POST", "/api/robots/arm2/stop", {}))

# %% [markdown]
# ## The guard
#
# Loopback is not a fence against the browser: any page you have open can POST to `127.0.0.1`. So every request and
# WebSocket handshake must name this server in `Host` (no DNS rebinding), carry no foreign `Origin`, and POST JSON
# (which a cross-site form cannot send without a preflight this server never answers).
#
# ![three checks on every request](media/07/guard.svg)

# %%
ACT = "/api/robots/so101/act"
move = {"targets": {"shoulder_pan": 0.2}}
_nb.show(_nb.table([
        {"request": name, "status": st, "detail": (body.get("detail", "") if isinstance(body, dict) else "")[:95]}
    for name, (st, body) in [
        ("same-origin JSON POST", api("POST", ACT, move)),
        ("Origin: https://evil.example", api("POST", ACT, move, {"Origin": "https://evil.example"})),
        ("Host: evil.example:8731", api("GET", "/api/scene", headers={"Host": "evil.example:8731"})),
        ("form POST (text/plain)", api("POST", ACT, "targets=1", {"Content-Type": "text/plain"})),
        ("unknown joint", api("POST", ACT, {"targets": {"elbow": 1.0}})),
        ("unknown robot", api("POST", "/api/robots/r2d2/act", move)),
    ]
]))

# %% [markdown]
# ## E-stop and resume
#
# The E-STOP button needs no confirmation and latches. While latched every act is a 409; a resume needs the name of
# whoever cleared the stop. A real robot would also need the header `X-Confirm: yes` from the operator.

# %%
print(api("POST", "/api/robots/so101/estop", {}))
chromium.call(lambda b: pages["desktop"].wait_for_function(
    "document.querySelector('#safety [data-robot=so101]')?.dataset.latched === 'yes'", timeout=30_000))
print(shot("desktop", "estop_latched.png"))
print(api("POST", ACT, move))
print(api("POST", "/api/robots/so101/resume", {}))
print(api("POST", "/api/robots/so101/resume", {"approved_by": "cagatay"}))
print(api("POST", ACT, move)[0])

# %% [markdown]
# ![so101 latched in the e-stop panel](media/07/estop_latched.png)

# %% [markdown]
# ## The chat panel
#
# The chat is a strands `Agent` over `sim.tool` and every `robot.tool`, built from `Dashboard(sim, model=...)`.
# One real turn, typed into the page.

# %%
CHAT = "Add a yellow sphere named lemon, radius 3 cm, on the floor at x=0.3 y=-0.3. Then list the robots in one line."
if model is None:
    print("skipped: no AWS_BEARER_TOKEN_BEDROCK / STRANDS_MODEL_ID in the environment")
else:
    loads = on("desktop", "() => window.__sceneLoaded")
    t0 = time.perf_counter()
    chromium.call(lambda b: (pages["desktop"].fill("#prompt", CHAT), pages["desktop"].press("#prompt", "Enter")))
    chromium.call(lambda b: pages["desktop"].wait_for_function(
        "(window.__chatEvents || []).some(e => e.type === 'done')", timeout=240_000), timeout=300)
    events = on("desktop", "() => window.__chatEvents")
    print(f"turn took {time.perf_counter() - t0:.1f} s")
    for ev in events:
        if ev["type"] == "tool_use":
            print("tool_use   ", ev["name"], json.dumps(ev["input"])[:110])
        elif ev["type"] == "tool_result":
            print("tool_result", ev["name"], ev["status"], ev["text"][:90].replace("\n", " "))
        elif ev["type"] == "error":
            print("error      ", ev["message"][:200])
    print("text:", "".join(e["delta"] for e in events if e["type"] == "text").strip()[:300])
    print("lemon in the world:", any(o.name == "lemon" for o in sim.scene.objects))
    if any(o.name == "lemon" for o in sim.scene.objects):
        chromium.call(lambda b: painted(pages["desktop"], loads + 1))
    chromium.call(lambda b: pages["desktop"].wait_for_timeout(1500))
    print(shot("desktop", "chat.png"))

# %% [markdown]
# ![one chat turn: the agent added the lemon through sim.tool](media/07/chat.png)

# %% [markdown]
# ## Standalone: no server at all
#
# With no dashboard reachable, the page loads a bundled world and steps it itself: MuJoCo WebAssembly for physics
# and an ONNX walking skill on onnxruntime-web. `python -m strands_robots.dashboard.standalone <dir>` writes the
# bundle; here it goes to a temporary folder served by a plain static file server.

# %%
import http.server, posixpath, subprocess, sys, tempfile, urllib.parse
from pathlib import Path
import strands_robots.dashboard as dashboard_pkg

STATIC = Path(dashboard_pkg.__file__).parent / "static"
bundle = Path(tempfile.mkdtemp(prefix="sr-standalone-")) / "microduck"
out = subprocess.run([sys.executable, "-m", "strands_robots.dashboard.standalone", str(bundle)],
                     capture_output=True, text=True, timeout=600)
print(out.stdout.strip() or out.stderr.strip()[-400:])


class Files(http.server.SimpleHTTPRequestHandler):
    """The dashboard's static page at /, the bundle at /bundle/."""

    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm"}

    def translate_path(self, path):
        path = posixpath.normpath(urllib.parse.unquote(path.split("?", 1)[0]))  # no `..` climbs out of a root
        if path.startswith("/bundle/"):
            return str(bundle / path[len("/bundle/"):])
        return str(STATIC / path.lstrip("/"))

    def log_message(self, *args):
        pass


files = http.server.ThreadingHTTPServer(("127.0.0.1", 8732), Files)
threading.Thread(target=files.serve_forever, daemon=True).start()
url = "http://127.0.0.1:8732/?standalone=/bundle/scene.json&cli=127.0.0.1:8739"  # cli=: a port nothing listens on


def walking(page):
    page.wait_for_function("window.__standalone && window.__standalone.stats || window.__error", timeout=240_000)


print(open_page("standalone", url, 1440, 900, wait=walking))
z0 = on("standalone", "() => window.__standalone.stats")
chromium.call(lambda b: pages["standalone"].keyboard.down("w"))
chromium.call(lambda b: pages["standalone"].wait_for_timeout(4000))
print(shot("standalone", "standalone.png"))
chromium.call(lambda b: pages["standalone"].keyboard.up("w"))
s = on("standalone", "() => window.__standalone.stats")
print("mode:", on("standalone", "() => window.__state.mode"), "| skill:", on("standalone", "() => window.__standalone.skill"),
      "| skills:", on("standalone", "() => window.__standalone.skills"))
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k in ("t", "z", "falls")},
      "| worker errors:", on("standalone", "() => window.__standalone.errors"))

# %% [markdown]
# ![standalone: the microduck walks in the browser, no server](media/07/standalone.png)
#
# The few errors this page logs are its probes for a server that is not there (`GET /api/scene`, `ws://.../ws/state`).
# The re-probe backs off from 5 s to 60 s, and with no server the e-stop panel is hidden and never polls `/api/mesh`.

# %% [markdown]
# ## With a real arm
#
# Pass hardware robots in `robots=`; their cards say `real`, and every slider move or run asks you first
# (the page sends `X-Confirm: yes` only after a confirm dialog). Needs a real arm, so it is not run here:
#
# ```python
# follower = Robot("so101", port="/dev/ttyACM0", id="blue_follower")   # needs a real arm; its twin is simulated
# Dashboard(follower.twin.sim, robots=[follower]).serve(port=8090)
# ```

# %% [markdown]
# ## Stop everything

# %%
console = {name: sum(1 for k, _ in log if k in ("error", "pageerror")) for name, log in logs.items()}
print("console errors per page:", console)
chromium.close()
files.shutdown()
files.server_close()
server.stop()
sim.close()
import socket

with socket.socket() as s1, socket.socket() as s2:
    print("ports 8731/8732 free:", s1.connect_ex(("127.0.0.1", 8731)) != 0 and s2.connect_ex(("127.0.0.1", 8732)) != 0)
