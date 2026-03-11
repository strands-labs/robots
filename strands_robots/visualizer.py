"""Live Dataset Visualization during Recording.

Real-time monitoring of recording sessions with terminal stats, matplotlib preview, and web dashboard.

Provides real-time monitoring of recording sessions with:
- Terminal-based live stats (frame count, FPS, duration, episode progress)
- Optional matplotlib frame preview (camera images)
- Optional web-based dashboard via simple HTTP server
- Integration with RecordSession and DatasetRecorder

Usage:
    from strands_robots.visualizer import RecordingVisualizer

    # Terminal stats only (no extra deps)
    viz = RecordingVisualizer(mode="terminal")
    viz.start()
    viz.update(frame_count=42, fps=29.8, episode=1, cameras={"wrist": img_array})
    viz.stop()

    # With matplotlib preview
    viz = RecordingVisualizer(mode="matplotlib")

    # Web dashboard
    viz = RecordingVisualizer(mode="web", port=8888)
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecordingStats:
    """Live recording statistics."""

    episode: int = 0
    total_episodes: int = 0
    frame_count: int = 0
    fps_actual: float = 0.0
    fps_target: float = 30.0
    duration_s: float = 0.0
    task: str = ""
    repo_id: str = ""
    state_dim: int = 0
    action_dim: int = 0
    cameras: List[str] = field(default_factory=list)
    errors: int = 0
    last_action: Optional[Dict[str, float]] = None
    last_state: Optional[Dict[str, float]] = None
    recording: bool = False


class RecordingVisualizer:
    """Live visualization for recording sessions.

    Modes:
    - "terminal": Print live stats to terminal (ANSI escape codes)
    - "matplotlib": Show camera frames + stats in matplotlib window
    - "web": Serve a simple HTML dashboard with auto-refresh
    - "json": Output JSON stats (for piping to other tools)
    """

    def __init__(
        self,
        mode: str = "terminal",
        refresh_rate: float = 2.0,
        port: int = 8888,
    ):
        self.mode = mode
        self.refresh_rate = refresh_rate
        self.port = port
        self.stats = RecordingStats()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._camera_frames: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._start_time = 0.0
        self._frame_times: List[float] = []
        self._web_server = None

    def start(self):
        """Start the visualization loop."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self.stats.recording = True

        if self.mode == "web":
            self._start_web_server()

        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()
        logger.info(f"Visualizer started (mode={self.mode})")

    def stop(self):
        """Stop the visualization."""
        self._running = False
        self.stats.recording = False
        if self._web_server:
            self._web_server.shutdown()
        logger.info("Visualizer stopped")

    def update(
        self,
        frame_count: Optional[int] = None,
        episode: Optional[int] = None,
        total_episodes: Optional[int] = None,
        task: Optional[str] = None,
        cameras: Optional[Dict[str, np.ndarray]] = None,
        last_action: Optional[Dict[str, float]] = None,
        last_state: Optional[Dict[str, float]] = None,
        error: bool = False,
    ):
        """Update visualization with new data."""
        with self._lock:
            now = time.time()
            if frame_count is not None:
                self.stats.frame_count = frame_count
            if episode is not None:
                self.stats.episode = episode
            if total_episodes is not None:
                self.stats.total_episodes = total_episodes
            if task is not None:
                self.stats.task = task
            if last_action is not None:
                self.stats.last_action = last_action
                self.stats.action_dim = len(last_action)
            if last_state is not None:
                self.stats.last_state = last_state
                self.stats.state_dim = len(last_state)
            if error:
                self.stats.errors += 1

            # Update camera frames
            if cameras:
                self._camera_frames.update(cameras)
                self.stats.cameras = list(self._camera_frames.keys())

            # Track FPS
            self._frame_times.append(now)
            # Keep last 2 seconds of frame times
            cutoff = now - 2.0
            self._frame_times = [t for t in self._frame_times if t > cutoff]
            if len(self._frame_times) >= 2:
                elapsed = self._frame_times[-1] - self._frame_times[0]
                if elapsed > 0:
                    self.stats.fps_actual = (len(self._frame_times) - 1) / elapsed

            self.stats.duration_s = now - self._start_time

    def new_episode(self, episode: int, task: str = ""):
        """Signal start of new episode (resets frame counter)."""
        with self._lock:
            self.stats.episode = episode
            self.stats.task = task
            self.stats.frame_count = 0
            self._frame_times.clear()
            self._start_time = time.time()

    def _display_loop(self):
        """Background thread that refreshes the display."""
        while self._running:
            try:
                if self.mode == "terminal":
                    self._render_terminal()
                elif self.mode == "matplotlib":
                    self._render_matplotlib()
                elif self.mode == "json":
                    self._render_json()
                # web mode is event-driven, no polling needed
            except Exception as e:
                logger.debug(f"Visualizer render error: {e}")

            time.sleep(1.0 / self.refresh_rate)

    def _render_terminal(self):
        """Render stats to terminal using ANSI codes."""
        with self._lock:
            s = self.stats

        # Build display
        bar_width = 40
        if s.fps_target > 0:
            fps_ratio = min(s.fps_actual / s.fps_target, 1.0)
        else:
            fps_ratio = 0

        fps_bar = "█" * int(fps_ratio * bar_width) + "░" * (
            bar_width - int(fps_ratio * bar_width)
        )
        fps_color = (
            "\033[32m"
            if fps_ratio > 0.9
            else "\033[33m" if fps_ratio > 0.7 else "\033[31m"
        )

        mins = int(s.duration_s // 60)
        secs = int(s.duration_s % 60)

        lines = [
            "\033[2J\033[H",  # Clear screen, move to top
            "╔══════════════════════════════════════════════════╗",
            "║  🎬 LIVE RECORDING MONITOR                      ║",
            "╠══════════════════════════════════════════════════╣",
            f"║  Episode: {s.episode}/{s.total_episodes}  │  Task: {s.task[:25]:<25} ║",
            f"║  Frames:  {s.frame_count:<8} │  Duration: {mins:02d}:{secs:02d}          ║",
            f"║  FPS: {fps_color}{s.fps_actual:5.1f}\033[0m/{s.fps_target:.0f}  [{fps_bar}] ║",
            f"║  Cameras: {', '.join(s.cameras) if s.cameras else 'none':<38} ║",
            f"║  State dim: {s.state_dim}  │  Action dim: {s.action_dim}           ║",
        ]

        if s.last_action:
            # Show first 4 action values
            vals = list(s.last_action.values())[:4]
            action_str = " ".join(f"{v:+.2f}" for v in vals)
            if len(s.last_action) > 4:
                action_str += f" +{len(s.last_action)-4} more"
            lines.append(f"║  Action: {action_str:<39} ║")

        if s.errors > 0:
            lines.append(f"║  ⚠️  Errors: {s.errors:<36} ║")

        lines.append("╠══════════════════════════════════════════════════╣")
        lines.append("║  Press Ctrl+C to stop recording                 ║")
        lines.append("╚══════════════════════════════════════════════════╝")

        print("\n".join(lines), flush=True)

    def _render_json(self):
        """Output stats as JSON line."""
        with self._lock:
            data = {
                "episode": self.stats.episode,
                "frame": self.stats.frame_count,
                "fps": round(self.stats.fps_actual, 1),
                "duration_s": round(self.stats.duration_s, 1),
                "task": self.stats.task,
                "cameras": self.stats.cameras,
                "errors": self.stats.errors,
            }
        print(json.dumps(data), flush=True)

    def _render_matplotlib(self):
        """Render camera frames in matplotlib (non-blocking)."""
        try:
            import matplotlib

            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            with self._lock:
                frames = dict(self._camera_frames)
                s = self.stats

            if not frames:
                return

            if not hasattr(self, "_fig") or self._fig is None:
                n_cams = max(len(frames), 1)
                self._fig, self._axes = plt.subplots(1, n_cams, figsize=(5 * n_cams, 4))
                if n_cams == 1:
                    self._axes = [self._axes]
                self._img_plots = {}
                plt.ion()
                plt.show(block=False)

            for idx, (cam_name, frame) in enumerate(frames.items()):
                if idx >= len(self._axes):
                    break
                ax = self._axes[idx]
                if cam_name not in self._img_plots:
                    self._img_plots[cam_name] = ax.imshow(frame)
                    ax.set_title(cam_name)
                    ax.axis("off")
                else:
                    self._img_plots[cam_name].set_data(frame)

            self._fig.suptitle(
                f"Ep {s.episode} | Frame {s.frame_count} | "
                f"FPS {s.fps_actual:.1f}/{s.fps_target:.0f} | {s.task}",
                fontsize=10,
            )
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

        except ImportError:
            logger.warning("matplotlib not available, falling back to terminal mode")
            self.mode = "terminal"
        except Exception as e:
            logger.debug(f"Matplotlib render error: {e}")

    def _start_web_server(self):
        """Start a simple HTTP server for web-based visualization."""
        import http.server
        import socketserver

        visualizer = self

        class DashboardHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/stats":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    with visualizer._lock:
                        data = {
                            "episode": visualizer.stats.episode,
                            "total_episodes": visualizer.stats.total_episodes,
                            "frame_count": visualizer.stats.frame_count,
                            "fps_actual": round(visualizer.stats.fps_actual, 1),
                            "fps_target": visualizer.stats.fps_target,
                            "duration_s": round(visualizer.stats.duration_s, 1),
                            "task": visualizer.stats.task,
                            "cameras": visualizer.stats.cameras,
                            "state_dim": visualizer.stats.state_dim,
                            "action_dim": visualizer.stats.action_dim,
                            "errors": visualizer.stats.errors,
                            "recording": visualizer.stats.recording,
                        }
                    self.wfile.write(json.dumps(data).encode())
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(DASHBOARD_HTML.encode())

            def log_message(self, format, *args):
                pass  # Suppress HTTP logs

        try:
            self._web_server = socketserver.TCPServer(
                ("127.0.0.1", self.port), DashboardHandler
            )
            thread = threading.Thread(
                target=self._web_server.serve_forever, daemon=True
            )
            thread.start()
            logger.info(f"Visualizer web dashboard: http://localhost:{self.port}")
            print(f"🎬 Live dashboard: http://localhost:{self.port}")
        except Exception as e:
            logger.error(f"Web server failed: {e}")

    def get_stats_dict(self) -> Dict[str, Any]:
        """Get current stats as dict (for tool integration)."""
        with self._lock:
            return {
                "episode": self.stats.episode,
                "total_episodes": self.stats.total_episodes,
                "frame_count": self.stats.frame_count,
                "fps_actual": round(self.stats.fps_actual, 1),
                "fps_target": self.stats.fps_target,
                "duration_s": round(self.stats.duration_s, 1),
                "task": self.stats.task,
                "cameras": self.stats.cameras,
                "state_dim": self.stats.state_dim,
                "action_dim": self.stats.action_dim,
                "errors": self.stats.errors,
                "recording": self.stats.recording,
            }


# Minimal HTML dashboard with auto-refresh
DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<title>🎬 Recording Monitor</title>
<style>
  body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
  .card { background: #16213e; border-radius: 8px; padding: 16px; margin: 8px 0; }
  .stat { display: inline-block; min-width: 180px; margin: 8px 16px; }
  .label { color: #7f8c8d; font-size: 12px; }
  .value { font-size: 24px; font-weight: bold; color: #00d4ff; }
  .fps-bar { height: 20px; background: #2c3e50; border-radius: 4px; overflow: hidden; }
  .fps-fill { height: 100%; transition: width 0.3s; border-radius: 4px; }
  .good { background: #2ecc71; } .warn { background: #f39c12; } .bad { background: #e74c3c; }
  h1 { color: #00d4ff; } .task { color: #f39c12; font-style: italic; }
  .recording { animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
</style>
</head>
<body>
<h1>🎬 Live Recording Monitor</h1>
<div class="card">
  <div class="stat"><div class="label">Episode</div><div class="value" id="episode">-</div></div>
  <div class="stat"><div class="label">Frames</div><div class="value" id="frames">-</div></div>
  <div class="stat"><div class="label">Duration</div><div class="value" id="duration">-</div></div>
  <div class="stat"><div class="label">FPS</div><div class="value" id="fps">-</div></div>
</div>
<div class="card">
  <div class="label">FPS</div>
  <div class="fps-bar"><div class="fps-fill" id="fpsbar" style="width:0%"></div></div>
</div>
<div class="card">
  <div class="stat"><div class="label">Task</div><div class="value task" id="task">-</div></div>
  <div class="stat"><div class="label">Cameras</div><div class="value" id="cameras">-</div></div>
  <div class="stat"><div class="label">State/Action</div><div class="value" id="dims">-</div></div>
  <div class="stat"><div class="label">Status</div><div class="value recording" id="status">●</div></div>
</div>
<script>
async function refresh() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    document.getElementById('episode').textContent = d.episode + '/' + d.total_episodes;
    document.getElementById('frames').textContent = d.frame_count;
    const m = Math.floor(d.duration_s/60), s = Math.floor(d.duration_s%60);
    document.getElementById('duration').textContent = String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
    document.getElementById('fps').textContent = d.fps_actual + '/' + d.fps_target;
    document.getElementById('task').textContent = d.task || '-';
    document.getElementById('cameras').textContent = d.cameras.join(', ') || 'none';
    document.getElementById('dims').textContent = d.state_dim + ' / ' + d.action_dim;
    document.getElementById('status').textContent = d.recording ? '● REC' : '■ STOP';
    document.getElementById('status').style.color = d.recording ? '#e74c3c' : '#7f8c8d';
    const ratio = d.fps_target > 0 ? Math.min(d.fps_actual/d.fps_target, 1) : 0;
    const bar = document.getElementById('fpsbar');
    bar.style.width = (ratio*100)+'%';
    bar.className = 'fps-fill ' + (ratio>0.9?'good':ratio>0.7?'warn':'bad');
  } catch(e) {}
}
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>"""
