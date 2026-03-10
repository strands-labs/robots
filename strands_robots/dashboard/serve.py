#!/usr/bin/env python3
"""
Strands Robots Dashboard Server

Bridges Zenoh mesh + all strands-robots capabilities to the browser:
- WebSocket: live state streaming from mesh peers
- REST API: robot discovery, camera capture, calibration, datasets, inference, training
- Static files: dashboard HTML

Usage:
    python -m strands_robots.dashboard.serve [port]
"""

import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
WS_PORT = PORT + 1

# ── Shared state ──────────────────────────────────────────────
_ws_clients = set()
_peers = {}
_states = {}
_streams = {}

# ── Headless sim for rendering ──

_render_lock = threading.Lock()
_sim_cache = {}  # robot_name → (model, data)


def _get_or_create_sim(robot_name):  # thread-safe
    """Get or create a headless MuJoCo sim for rendering."""
    if robot_name in _sim_cache:
        return _sim_cache[robot_name]
    try:
        import mujoco

        from strands_robots.assets import resolve_model_path

        model_path = resolve_model_path(robot_name, prefer_scene=True)
        if not model_path:
            return None
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        _sim_cache[robot_name] = (model, data)
        logger.info(f"Created headless sim for {robot_name}")
        return (model, data)
    except Exception as e:
        logger.error(f"Failed to create sim for {robot_name}: {e}")
        return None


def _render_sim(robot_name, width=640, height=480, joint_data=None, camera="default"):
    """Render a frame from the headless sim, optionally with joint positions applied."""
    import mujoco

    sim = _get_or_create_sim(robot_name)
    if not sim:
        return None
    model, data = sim

    # Apply joint positions from mesh state if provided
    if joint_data and isinstance(joint_data, dict):
        for name, val in joint_data.items():
            try:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    addr = model.jnt_qposadr[jid]
                    data.qpos[addr] = float(val)
            except Exception:
                pass
        mujoco.mj_forward(model, data)  # Update kinematics without stepping physics

    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        # Try named camera, fall back to default
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id >= 0:
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data)
        img = renderer.render().copy()
        del renderer

        import io

        from PIL import Image

        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.error(f"Render failed: {e}")
        return None


# ── REST API Handler ──────────────────────────────────────────
class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve static files + REST API for dashboard."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def log_message(self, format, *args):
        if "/api/" in str(args[0]) if args else False:
            logger.info(format % args)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed.path, parse_qs(parsed.query))
        elif parsed.path == "/stream":
            self._serve_mjpeg_stream(parse_qs(parsed.query))
        elif parsed.path.startswith("/assets/"):
            self._serve_asset(parsed.path)
        else:
            super().do_GET()

    def _serve_mjpeg_stream(self, params):
        """MJPEG video stream of MuJoCo sim."""
        from urllib.parse import parse_qs

        qs = parse_qs(params) if isinstance(params, str) else params
        robot_name = qs.get("robot", ["reachy_mini"])[0] if isinstance(qs, dict) else "reachy_mini"
        peer_id = qs.get("peer_id", [None])[0] if isinstance(qs, dict) else None
        # Cap size to avoid framebuffer overflow
        width = min(int(qs.get("width", ["480"])[0]) if isinstance(qs, dict) else 480, 640)
        height = min(int(qs.get("height", ["360"])[0]) if isinstance(qs, dict) else 360, 480)
        if peer_id and peer_id in _peers:
            hw = _peers[peer_id].get("hw")
            if hw:
                robot_name = hw
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        import io
        import math
        import time as _time

        try:
            import mujoco
            from PIL import Image
        except ImportError:
            return
        sim = _get_or_create_sim(robot_name)
        if not sim:
            return
        model, data = sim
        # Create ONE renderer and reuse it
        renderer = mujoco.Renderer(model, height=height, width=width)
        fc = 0
        BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n"
        CRLF = b"\r\n"
        try:
            while True:
                # Apply joint data from mesh or demo animation
                if peer_id and peer_id in _states:
                    for n, v in (_states[peer_id].get("joints") or {}).items():
                        try:
                            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                            if jid >= 0:
                                data.qpos[model.jnt_qposadr[jid]] = float(v)
                        except Exception:
                            pass
                else:
                    t = fc * 0.05
                    for j in range(min(model.njnt, 6)):
                        if model.jnt_type[j] == 3:
                            data.qpos[model.jnt_qposadr[j]] = math.sin(t + j * 0.7) * 0.3
                mujoco.mj_forward(model, data)
                with _render_lock:
                    renderer.update_scene(data)
                img = renderer.render()
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="JPEG", quality=70)
                jpg = buf.getvalue()
                cl = b"Content-Length: " + str(len(jpg)).encode() + CRLF + CRLF
                self.wfile.write(BOUNDARY + cl + jpg + CRLF)
                self.wfile.flush()
                fc += 1
                _time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                del renderer
            except Exception:
                pass

    def _serve_asset(self, path):
        """Serve robot mesh/model files from strands_robots/assets/."""
        try:
            from strands_robots.assets import get_assets_dir

            # /assets/reachy_mini/mjcf/assets/foo.stl → assets_dir/reachy_mini/mjcf/assets/foo.stl
            rel = path[len("/assets/") :]
            file_path = get_assets_dir() / rel
            if not file_path.exists() or not file_path.is_file():
                self.send_error(404)
                return
            # Determine content type
            ext = file_path.suffix.lower()
            ct = {
                ".stl": "model/stl",
                ".xml": "application/xml",
                ".json": "application/json",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".obj": "model/obj",
            }.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(file_path.stat().st_size))
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        except Exception as e:
            self.send_error(500, str(e))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
            self._handle_api(parsed.path, body, method="POST")
        else:
            self.send_error(405)

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def _handle_api(self, path, params, method="GET"):
        try:
            # ── Robot Discovery ──
            if path == "/api/robots":
                return self._api_robots()
            elif path == "/api/peers":
                return self._api_peers()

            # ── Camera ──
            elif path == "/api/cameras":
                return self._api_cameras()
            elif path == "/api/camera/capture":
                return self._api_camera_capture(params)

            # ── Calibration ──
            elif path == "/api/calibrations":
                return self._api_calibrations()

            # ── Datasets ──
            elif path == "/api/datasets":
                return self._api_datasets(params)
            elif path == "/api/datasets/lerobot":
                return self._api_lerobot_datasets(params)

            # ── Policies / Inference ──
            elif path == "/api/policies":
                return self._api_policies()
            elif path == "/api/inference/services":
                return self._api_inference_services()

            # ── Training ──
            elif path == "/api/training/providers":
                return self._api_training_providers()

            # ── Sim Render ──
            elif path == "/api/sim/render":
                return self._api_sim_render(params)

            # ── Robot Scene (kinematic tree + mesh URLs for Three.js) ──
            elif path == "/api/robot/scene":
                return self._api_robot_scene(params)

            # ── System ──
            elif path == "/api/status":
                return self._api_status()

            else:
                self._json_response({"error": f"Unknown endpoint: {path}"}, 404)

        except Exception as e:
            logger.error(f"API error: {e}\n{traceback.format_exc()}")
            self._json_response({"error": str(e)}, 500)

    # ── API Implementations ───────────────────────────────────

    def _api_robots(self):
        """List all available robots from factory."""
        try:
            from strands_robots.factory import _UNIFIED_ROBOTS

            robots = []
            for name, info in sorted(_UNIFIED_ROBOTS.items()):
                robots.append(
                    {
                        "name": name,
                        "description": info.get("description", ""),
                        "has_sim": info.get("sim") is not None,
                        "has_real": info.get("real") is not None,
                    }
                )
            self._json_response({"robots": robots, "count": len(robots)})
        except Exception as e:
            self._json_response({"robots": [], "error": str(e)})

    def _api_peers(self):
        """Get all mesh peers — uses the bridge's local state (not zenoh_mesh globals)."""
        self._json_response({"peers": list(_peers.values()), "local": [], "states": _states})

    def _api_cameras(self):
        """Discover available cameras (non-blocking, probes first 4 indices)."""
        try:
            import cv2

            found = []
            for i in range(4):  # Only probe 0-3 to avoid blocking
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            found.append({"index": i, "width": w, "height": h})
                        cap.release()
                except Exception:
                    continue
            self._json_response({"cameras": found})
        except ImportError:
            self._json_response({"cameras": [], "error": "opencv not installed"})

    def _api_camera_capture(self, params):
        """Capture a frame from a camera."""
        import cv2

        idx = int(params.get("index", [0])[0]) if isinstance(params, dict) and "index" in params else 0
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return self._json_response({"error": f"Camera {idx} not available"}, 400)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return self._json_response({"error": "Capture failed"}, 500)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf.tobytes()).decode()
        self._json_response(
            {"image": f"data:image/jpeg;base64,{b64}", "width": frame.shape[1], "height": frame.shape[0]}
        )

    def _api_calibrations(self):
        """List calibration files."""
        try:
            from strands_robots.tools.lerobot_calibrate import LeRobotCalibrationManager

            mgr = LeRobotCalibrationManager()
            structure = mgr.get_calibration_structure()
            self._json_response({"calibrations": structure})
        except Exception as e:
            self._json_response({"calibrations": {}, "error": str(e)})

    def _api_datasets(self, params):
        """List local LeRobot datasets."""
        try:
            from pathlib import Path

            roots = [
                Path.home() / ".cache/huggingface/lerobot",
                Path.cwd() / "data",
                Path("/data"),
            ]
            datasets = []
            for root in roots:
                if root.exists():
                    for d in root.iterdir():
                        if d.is_dir() and (d / "meta").exists():
                            datasets.append({"name": d.name, "path": str(d), "root": str(root)})
            self._json_response({"datasets": datasets})
        except Exception as e:
            self._json_response({"datasets": [], "error": str(e)})

    def _api_lerobot_datasets(self, params):
        """Search HuggingFace for LeRobot datasets."""
        import urllib.request

        q = params.get("q", ["lerobot"])[0] if isinstance(params, dict) else "lerobot"
        try:
            url = f"https://huggingface.co/api/datasets?search={q}&sort=downloads&direction=-1&limit=50"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            datasets = [{"id": d["id"], "downloads": d.get("downloads", 0), "likes": d.get("likes", 0)} for d in data]
            self._json_response({"datasets": datasets})
        except Exception as e:
            self._json_response({"datasets": [], "error": str(e)})

    def _api_policies(self):
        """List available policy providers."""
        try:
            from strands_robots.tools.inference import PROVIDERS

            self._json_response({"providers": PROVIDERS})
        except Exception as e:
            self._json_response({"providers": {}, "error": str(e)})

    def _api_inference_services(self):
        """List running inference services."""
        try:
            from strands_robots.tools.inference import _RUNNING

            services = {str(k): v for k, v in _RUNNING.items()}
            self._json_response({"services": services})
        except Exception as e:
            self._json_response({"services": {}, "error": str(e)})

    def _api_training_providers(self):
        """List training providers."""
        providers = {
            "groot": {"name": "NVIDIA GR00T N1.6", "desc": "Fine-tune GR00T foundation model"},
            "lerobot": {"name": "LeRobot", "desc": "Train ACT/Pi0/SmolVLA/Diffusion policies"},
            "cosmos_predict": {"name": "Cosmos Predict 2.5", "desc": "Post-train world foundation model"},
            "dreamgen_idm": {"name": "DreamGen IDM", "desc": "Train inverse dynamics model"},
            "dreamgen_vla": {"name": "DreamGen VLA", "desc": "Fine-tune VLA from dreams"},
        }
        self._json_response({"providers": providers})

    def _api_sim_render(self, params):
        """Render a robot sim scene server-side. Creates headless MuJoCo sim on demand.

        Applies live joint positions from mesh state if available.
        """
        robot_name = (params.get("robot", [None])[0] if isinstance(params, dict) else None) or "reachy_mini"
        peer_id = params.get("peer_id", [None])[0] if isinstance(params, dict) else None
        width = int(params.get("width", ["640"])[0]) if isinstance(params, dict) else 640
        height = int(params.get("height", ["480"])[0]) if isinstance(params, dict) else 480
        camera = params.get("camera", ["default"])[0] if isinstance(params, dict) else "default"

        # If peer_id given, try to get robot name from presence
        if peer_id and peer_id in _peers:
            hw = _peers[peer_id].get("hw")
            if hw:
                robot_name = hw

        # Get live joint data from mesh state
        joint_data = None
        if peer_id and peer_id in _states:
            joint_data = _states[peer_id].get("joints")

        # Render
        image_data = _render_sim(robot_name, width, height, joint_data, camera)
        if image_data:
            self._json_response(
                {
                    "image": image_data,
                    "robot": robot_name,
                    "peer_id": peer_id,
                    "width": width,
                    "height": height,
                }
            )
        else:
            self._json_response({"error": f"Failed to render {robot_name}"}, 500)

    def _api_robot_scene(self, params):
        """Export robot kinematic tree + mesh file paths for client-side Three.js rendering."""
        robot_name = params.get("robot", ["reachy_mini"])[0] if isinstance(params, dict) else "reachy_mini"
        try:
            import mujoco

            from strands_robots.assets import resolve_model_path

            model_path = resolve_model_path(robot_name, prefer_scene=True)
            if not model_path:
                return self._json_response({"error": f"No model for {robot_name}"}, 404)

            model = mujoco.MjModel.from_xml_path(str(model_path))

            # Export body tree
            bodies = []
            for i in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
                parent = int(model.body_parentid[i])
                pos = model.body_pos[i].tolist()
                quat = model.body_quat[i].tolist()

                # Find geoms attached to this body
                geoms = []
                for g in range(model.ngeom):
                    if model.geom_bodyid[g] == i:
                        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
                        gtype = int(model.geom_type[g])  # 0=plane,2=sphere,3=capsule,5=cylinder,6=box,7=mesh
                        gsize = model.geom_size[g].tolist()
                        gpos = model.geom_pos[g].tolist()
                        gquat = model.geom_quat[g].tolist()
                        rgba = model.geom_rgba[g].tolist()

                        geom_data = {
                            "name": gname,
                            "type": gtype,
                            "size": gsize,
                            "pos": gpos,
                            "quat": gquat,
                            "rgba": rgba,
                        }

                        # If mesh type, get mesh file info
                        if gtype == 7:  # mesh
                            mesh_id = model.geom_dataid[g]
                            if mesh_id >= 0:
                                mesh_name = (
                                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or f"mesh_{mesh_id}"
                                )
                                geom_data["mesh_name"] = mesh_name

                        geoms.append(geom_data)

                bodies.append({"id": i, "name": name, "parent": parent, "pos": pos, "quat": quat, "geoms": geoms})

            # Export joints
            joints = []
            for j in range(model.njnt):
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
                jtype = int(model.jnt_type[j])
                body_id = int(model.jnt_bodyid[j])
                axis = model.jnt_axis[j].tolist()
                jrange = model.jnt_range[j].tolist()
                joints.append(
                    {"id": j, "name": jname, "type": jtype, "body_id": body_id, "axis": axis, "range": jrange}
                )

            self._json_response(
                {
                    "robot": robot_name,
                    "bodies": bodies,
                    "joints": joints,
                    "nbody": model.nbody,
                    "njnt": model.njnt,
                    "nmesh": model.nmesh,
                    "asset_base": f"/assets/{robot_name}/mjcf/",
                }
            )
        except Exception as e:
            import traceback

            self._json_response({"error": str(e), "trace": traceback.format_exc()}, 500)

    def _api_status(self):
        """System status."""
        import platform

        self._json_response(
            {
                "hostname": os.uname().nodename,
                "platform": platform.system(),
                "arch": platform.machine(),
                "python": platform.python_version(),
                "peers": len(_peers),
                "ws_clients": len(_ws_clients),
            }
        )


# ── WebSocket Handler ─────────────────────────────────────────
async def ws_handler(websocket):
    _ws_clients.add(websocket)
    logger.info(f"Browser connected ({len(_ws_clients)})")
    await websocket.send(json.dumps({"type": "snapshot", "peers": _peers, "states": _states}))
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
                await _handle_ws_msg(msg, websocket)
            except Exception as e:
                await websocket.send(json.dumps({"type": "error", "error": str(e)}))
    finally:
        _ws_clients.discard(websocket)


async def _handle_ws_msg(msg, ws):
    action = msg.get("action")
    if action == "tell":
        target, instruction = msg.get("target"), msg.get("instruction", "")
        try:
            from strands_robots.zenoh_mesh import _LOCAL_ROBOTS

            mesh = next(iter(_LOCAL_ROBOTS.values()), None)
            if mesh:
                result = mesh.send(
                    target,
                    {
                        "action": "execute",
                        "instruction": instruction,
                        "policy_provider": msg.get("policy_provider", "mock"),
                        "duration": msg.get("duration", 30.0),
                    },
                )
                await ws.send(json.dumps({"type": "cmd_result", "target": target, "result": result}))
        except Exception as e:
            await ws.send(json.dumps({"type": "error", "error": str(e)}))
    elif action == "stop":
        try:
            from strands_robots.zenoh_mesh import _LOCAL_ROBOTS

            mesh = next(iter(_LOCAL_ROBOTS.values()), None)
            if mesh:
                mesh.send(msg.get("target"), {"action": "stop"}, timeout=5.0)
        except Exception:
            pass
    elif action == "emergency_stop":
        try:
            from strands_robots.zenoh_mesh import _LOCAL_ROBOTS

            mesh = next(iter(_LOCAL_ROBOTS.values()), None)
            if mesh:
                mesh.emergency_stop()
        except Exception:
            pass
    elif action == "ping":
        await ws.send(json.dumps({"type": "pong", "t": time.time()}))


async def _broadcast(msg: dict):
    if not _ws_clients:
        return
    data = json.dumps(msg, default=str)
    await asyncio.gather(*[ws.send(data) for ws in _ws_clients.copy()], return_exceptions=True)


# ── Zenoh Bridge Thread ──────────────────────────────────────
def zenoh_bridge_thread(loop):
    try:
        # Create a DEDICATED zenoh session for the bridge — LISTEN only
        import zenoh as _z

        MESH_PORT = int(os.getenv("STRANDS_MESH_PORT", "7447"))
        try:
            _bcfg = _z.Config()
            _bcfg.insert_json5("mode", '"router"')
            _bcfg.insert_json5("listen/endpoints", json.dumps([f"tcp/0.0.0.0:{MESH_PORT}"]))
            session = _z.open(_bcfg)
            logger.info(f"Bridge: listening on tcp/0.0.0.0:{MESH_PORT}")
        except Exception as e:
            logger.warning(f"Bridge listen failed ({e}) — trying connect")
            try:
                _bcfg2 = _z.Config()
                _bcfg2.insert_json5("connect/endpoints", json.dumps([f"tcp/127.0.0.1:{MESH_PORT}"]))
                session = _z.open(_bcfg2)
                logger.info(f"Bridge: connected to tcp/127.0.0.1:{MESH_PORT}")
            except Exception as e2:
                logger.warning(f"Bridge zenoh failed: {e2}")
                return

        def on_presence(sample):
            try:
                key = str(sample.key_expr)
                raw = sample.payload.to_bytes().decode()
                logger.info(f"PRESENCE received on {key}: {raw[:100]}")
                d = json.loads(raw)
                pid = d.get("robot_id")
                if pid:
                    _peers[pid] = d
                    asyncio.run_coroutine_threadsafe(_broadcast({"type": "presence", "peer_id": pid, "data": d}), loop)
            except Exception as e:
                logger.error(f"Presence handler error: {e}")

        def on_state(sample):
            try:
                d = json.loads(sample.payload.to_bytes().decode())
                pid = d.get("peer_id")
                if pid:
                    _states[pid] = d
                    asyncio.run_coroutine_threadsafe(_broadcast({"type": "state", "peer_id": pid, "data": d}), loop)
            except Exception:
                pass

        def on_stream(sample):
            try:
                d = json.loads(sample.payload.to_bytes().decode())
                pid = d.get("peer_id")
                if pid:
                    _streams.setdefault(pid, []).append(d)
                    if len(_streams[pid]) > 200:
                        _streams[pid] = _streams[pid][-100:]
                    asyncio.run_coroutine_threadsafe(_broadcast({"type": "stream", "peer_id": pid, "data": d}), loop)
            except Exception:
                pass

        session.declare_subscriber("strands/*/presence", on_presence)
        session.declare_subscriber("strands/*/state", on_state)
        session.declare_subscriber("strands/*/stream", on_stream)

        # RAW debug subscriber
        def on_raw(sample):
            import sys

            key = str(sample.key_expr)
            print(f"ZENOH_RAW: {key}", file=sys.stderr, flush=True)
            logger.info(f"ZENOH_RAW: {key}")

        session.declare_subscriber("**", on_raw)
        logger.info("✅ Zenoh bridge active (with raw ** debug sub)")
    except Exception as e:
        logger.warning(f"Zenoh bridge failed: {e} — REST API still works")


# ── Main ──────────────────────────────────────────────────────
async def main():
    from http.server import ThreadingHTTPServer

    from websockets.asyncio.server import serve as ws_serve

    loop = asyncio.get_event_loop()
    threading.Thread(target=zenoh_bridge_thread, args=(loop,), daemon=True).start()

    # HTTP server (static + API)
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), DashboardHandler)
    httpd.allow_reuse_address = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # WebSocket server
    async with ws_serve(ws_handler, "0.0.0.0", WS_PORT):
        print("🦾 Strands Robots Dashboard")
        print(f"   🌐 http://localhost:{PORT}/index.html")
        print(f"   🔌 ws://localhost:{WS_PORT}")
        print(f"   📡 API: http://localhost:{PORT}/api/")
        print("   Endpoints: /api/robots, /api/peers, /api/cameras,")
        print("              /api/calibrations, /api/datasets, /api/policies,")
        print("              /api/inference/services, /api/training/providers")
        await asyncio.Future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    asyncio.run(main())
