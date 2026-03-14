"""Domain randomization mixin."""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from strands_robots.simulation._backend import _ensure_mujoco

logger = logging.getLogger(__name__)


class RandomizationMixin:
    """Domain randomization for Simulation. Expects self._world."""

    def randomize(
        self,
        randomize_colors: bool = True,
        randomize_lighting: bool = True,
        randomize_physics: bool = False,
        randomize_positions: bool = False,
        position_noise: float = 0.02,
        color_range: Tuple[float, float] = (0.1, 1.0),
        friction_range: Tuple[float, float] = (0.5, 1.5),
        mass_range: Tuple[float, float] = (0.5, 2.0),
        seed: int = None,
    ) -> Dict[str, Any]:
        """Apply domain randomization to the scene."""
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        rng = np.random.default_rng(seed)
        mj = _ensure_mujoco()
        model = self._world._model
        data = self._world._data
        changes = []

        if randomize_colors:
            for i in range(model.ngeom):
                geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
                if geom_name and geom_name != "ground":
                    model.geom_rgba[i, :3] = rng.uniform(color_range[0], color_range[1], size=3)
            changes.append(f"🎨 Colors: {model.ngeom} geoms randomized")

        if randomize_lighting:
            for i in range(model.nlight):
                model.light_pos[i] += rng.uniform(-0.5, 0.5, size=3)
                model.light_diffuse[i] = rng.uniform(0.3, 1.0, size=3)
            changes.append(f"💡 Lighting: {model.nlight} lights randomized")

        if randomize_physics:
            for i in range(model.ngeom):
                model.geom_friction[i, 0] *= rng.uniform(*friction_range)
            for i in range(model.nbody):
                if model.body_mass[i] > 0:
                    model.body_mass[i] *= rng.uniform(*mass_range)
            changes.append(f"⚙️ Physics: friction×[{friction_range}], mass×[{mass_range}]")

        if randomize_positions:
            for obj_name, obj in self._world.objects.items():
                if not obj.is_static:
                    jnt_name = f"{obj_name}_joint"
                    jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
                    if jnt_id >= 0:
                        qpos_addr = model.jnt_qposadr[jnt_id]
                        noise = rng.uniform(-position_noise, position_noise, size=3)
                        data.qpos[qpos_addr : qpos_addr + 3] += noise
            mj.mj_forward(model, data)
            changes.append(f"📍 Positions: ±{position_noise}m noise on dynamic objects")

        return {
            "status": "success",
            "content": [{"text": "🎲 Domain Randomization applied:\n" + "\n".join(changes)}],
        }
