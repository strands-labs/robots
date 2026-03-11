"""Motion library for strands-robots.

Download and play pre-recorded robot motions from HuggingFace datasets.
Inspired by Reachy Mini's recorded_move + HF dataset pattern.

Usage:
    from strands_robots.motion_library import MotionLibrary

    lib = MotionLibrary()
    lib.preload("pollen-robotics/reachy-mini-emotions-library")

    # List available motions
    motions = lib.list_motions()

    # Get a motion as trajectory
    traj = lib.get_motion("happy")

    # Play in simulation
    lib.play_in_sim(sim, "arm1", "happy")
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default cache directory
MOTION_CACHE_DIR = Path.home() / ".strands_robots" / "motions"
MOTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class Motion:
    """A single recorded motion (trajectory)."""

    def __init__(self, name: str, data: Dict[str, Any], source: str = ""):
        self.name = name
        self.data = data
        self.source = source

        # Extract trajectory info
        self.timestamps = data.get("time", data.get("timestamps", []))
        self.description = data.get("description", name)
        self.duration = self.timestamps[-1] if self.timestamps else 0.0
        self.n_steps = len(self.timestamps)

        # Joint trajectories
        self.joint_positions = data.get(
            "joint_positions", data.get("trajectory", data.get("actions", []))
        )

    def get_action_at(self, t: float) -> Optional[Dict[str, float]]:
        """Get interpolated action at time t."""
        if not self.timestamps or not self.joint_positions:
            return None

        # Clamp
        t = max(0, min(t, self.duration))

        # Find bracketing indices
        idx = 0
        for i, ts in enumerate(self.timestamps):
            if ts > t:
                break
            idx = i

        if idx >= len(self.joint_positions):
            idx = len(self.joint_positions) - 1

        positions = self.joint_positions[idx]

        if isinstance(positions, dict):
            return positions
        elif isinstance(positions, list):
            return {f"joint_{i}": v for i, v in enumerate(positions)}
        else:
            return None

    def as_numpy(self) -> Optional[np.ndarray]:
        """Get trajectory as numpy array (n_steps x n_joints)."""
        if not self.joint_positions:
            return None

        if isinstance(self.joint_positions[0], dict):
            keys = sorted(self.joint_positions[0].keys())
            return np.array([[step[k] for k in keys] for step in self.joint_positions])
        elif isinstance(self.joint_positions[0], list):
            return np.array(self.joint_positions)
        return None

    def __repr__(self):
        return f"Motion('{self.name}', steps={self.n_steps}, duration={self.duration:.1f}s)"


class MotionLibrary:
    """Library of pre-recorded robot motions from HuggingFace datasets.

    Supports:
    - HuggingFace dataset repos (auto-downloaded with huggingface_hub)
    - Local JSON/JSONL files
    - In-memory registration
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = Path(cache_dir) if cache_dir else MOTION_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._motions: Dict[str, Motion] = {}
        self._loaded_repos: set = set()

    def preload(self, repo_id: str, repo_type: str = "dataset") -> int:
        """Download and load motions from a HuggingFace dataset.

        Args:
            repo_id: HuggingFace repo ID (e.g., "pollen-robotics/reachy-mini-emotions-library")
            repo_type: "dataset" or "model"

        Returns:
            Number of motions loaded
        """
        if repo_id in self._loaded_repos:
            logger.info("Already loaded: %s", repo_id)
            return 0

        try:
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(repo_id, repo_type=repo_type)
            count = self.load_directory(local_path, source=repo_id)
            self._loaded_repos.add(repo_id)
            logger.info("Loaded %s motions from %s", count, repo_id)
            return count
        except ImportError:
            logger.warning("huggingface_hub not installed — can't download motions")
            return 0
        except Exception as e:
            logger.warning("Failed to load motions from %s: %s", repo_id, e)
            return 0

    def load_directory(self, path: str, source: str = "") -> int:
        """Load all JSON motion files from a directory."""
        path = Path(path)
        count = 0

        for json_file in sorted(path.rglob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Handle single motion or list of motions
                if isinstance(data, list):
                    for item in data:
                        name = item.get("name", item.get("description", json_file.stem))
                        self._motions[name] = Motion(name, item, source=source)
                        count += 1
                elif isinstance(data, dict):
                    # Could be a single motion or a collection
                    if (
                        "name" in data
                        or "trajectory" in data
                        or "joint_positions" in data
                    ):
                        name = data.get("name", json_file.stem)
                        self._motions[name] = Motion(name, data, source=source)
                        count += 1
                    else:
                        # Dict of motions keyed by name
                        for name, motion_data in data.items():
                            if isinstance(motion_data, dict):
                                self._motions[name] = Motion(
                                    name, motion_data, source=source
                                )
                                count += 1
            except Exception as e:
                logger.debug("Could not load %s: %s", json_file, e)

        return count

    def load_file(self, path: str, name: Optional[str] = None) -> Optional[Motion]:
        """Load a single JSON motion file."""
        path = Path(path)
        try:
            with open(path) as f:
                data = json.load(f)
            motion_name = name or data.get("name", path.stem)
            motion = Motion(motion_name, data, source=str(path))
            self._motions[motion_name] = motion
            return motion
        except Exception as e:
            logger.warning("Failed to load motion from %s: %s", path, e)
            return None

    def register(self, name: str, data: Dict[str, Any]) -> Motion:
        """Register a motion in-memory."""
        motion = Motion(name, data, source="in-memory")
        self._motions[name] = motion
        return motion

    def get(self, name: str) -> Optional[Motion]:
        """Get a motion by name."""
        return self._motions.get(name)

    def list_motions(self) -> List[Dict[str, Any]]:
        """List all available motions."""
        return [
            {
                "name": m.name,
                "description": m.description,
                "duration": m.duration,
                "steps": m.n_steps,
                "source": m.source,
            }
            for m in self._motions.values()
        ]

    def search(self, query: str) -> List[Motion]:
        """Search motions by name or description."""
        query_lower = query.lower()
        return [
            m
            for m in self._motions.values()
            if query_lower in m.name.lower() or query_lower in m.description.lower()
        ]

    def play_in_sim(
        self,
        simulation,
        robot_name: str,
        motion_name: str,
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        """Play a motion in the simulation.

        Args:
            simulation: Simulation backend instance (MujocoBackend, etc.)
            robot_name: Name of the robot in the sim
            motion_name: Name of the motion to play
            speed: Playback speed multiplier

        Returns:
            Dict with playback stats
        """
        motion = self.get(motion_name)
        if not motion:
            return {"status": "error", "message": f"Motion '{motion_name}' not found"}

        trajectory = motion.as_numpy()
        if trajectory is None:
            return {"status": "error", "message": "Motion has no valid trajectory data"}

        try:
            import mujoco

            world = simulation._world
            if world is None:
                return {"status": "error", "message": "No world created"}

            model = world._model
            data = world._data

            # Apply each step
            steps_applied = 0
            for step_idx, actions in enumerate(trajectory):
                # Apply to actuators
                n_act = min(len(actions), model.nu)
                data.ctrl[:n_act] = actions[:n_act]

                # Step physics
                n_substeps = max(1, int(1.0 / (speed * 50 * model.opt.timestep)))
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)

                steps_applied += 1

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Played '{motion_name}' on '{robot_name}': "
                        f"{steps_applied} steps, {motion.duration / speed:.1f}s"
                    }
                ],
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def save(self, name: str, path: Optional[str] = None) -> str:
        """Save a motion to JSON file."""
        motion = self.get(name)
        if not motion:
            raise ValueError(f"Motion '{name}' not found")

        if path is None:
            path = str(self._cache_dir / f"{name}.json")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(motion.data, f, indent=2, default=str)

        return path

    @property
    def count(self) -> int:
        return len(self._motions)

    def __repr__(self):
        return f"MotionLibrary({self.count} motions, {len(self._loaded_repos)} repos)"
