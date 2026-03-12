"""Sensor management — contact, IMU, and tiled camera sensors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ._core import NewtonBackend


def add_sensor(
    backend: NewtonBackend,
    name: str,
    sensor_type: str = "contact",
    body_name: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Add a sensor to the simulation.

    Args:
        name: Sensor name.
        sensor_type: "contact", "imu", or "tiled_camera".
        body_name: Body to attach sensor to.
    """
    if backend._model is None:
        from ._scene import finalize_model

        finalize_model(backend)

    try:
        from newton.sensors import SensorContact, SensorIMU, SensorTiledCamera

        if sensor_type == "contact":
            shapes = kwargs.get("shapes", [0])
            sensor = SensorContact(
                sensing_obj_shapes=shapes, verbose=kwargs.get("verbose", False)
            )
        elif sensor_type == "imu":
            sensor = SensorIMU(**kwargs)
        elif sensor_type == "tiled_camera":
            sensor = SensorTiledCamera(
                width=kwargs.get("width", 640),
                height=kwargs.get("height", 480),
                **{k: v for k, v in kwargs.items() if k not in ("width", "height")},
            )
        else:
            return {
                "success": False,
                "message": f"Unknown sensor type: {sensor_type}",
            }

        backend._sensors[name] = {"sensor": sensor, "type": sensor_type}
        return {
            "success": True,
            "message": f"Sensor '{name}' added ({sensor_type})",
        }
    except ImportError:
        return {"success": False, "message": "newton.sensors not available"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def read_sensor(backend: NewtonBackend, name: str) -> Dict[str, Any]:
    """Read data from a named sensor."""
    if name not in backend._sensors:
        return {"success": False, "message": f"Sensor '{name}' not found."}

    try:
        sensor_info = backend._sensors[name]
        sensor = sensor_info["sensor"]
        data = sensor.evaluate(backend._model, backend._state_0, backend._contacts)
        return {"success": True, "data": data}
    except Exception as exc:
        return {"success": False, "message": str(exc)}
