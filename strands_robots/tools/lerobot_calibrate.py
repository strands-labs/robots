"""LeRobot Calibration Management Tool

This tool provides comprehensive management of LeRobot calibration files,
allowing users to list, view, backup, restore, and analyze calibration data
for use with LeRobot teleoperation and robot control systems.

Features:
- List all available calibrations organized by device type and model
- View detailed calibration data including motor configurations
- Backup and restore calibration files
- Delete unwanted calibrations
- Analyze calibration statistics
- Search and filter calibrations

Based on LeRobot's calibration system:
- Calibrations stored in ~/.cache/huggingface/lerobot/calibration/
- Organized by device type: teleoperators/ and robots/
- Each device type has subfolders for specific models
- Calibration files are JSON files named by device ID
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from strands import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LeRobot calibration paths
try:
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    # Fallback path
    HF_LEROBOT_CALIBRATION = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"

# Session storage for backups
BACKUP_DIR = Path.cwd() / ".strands_robots/.calibration_backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


class LeRobotCalibrationManager:
    """Independent LeRobot calibration management class"""

    def __init__(self, base_path: Path | None = None):
        self.base_path = Path(base_path) if base_path else HF_LEROBOT_CALIBRATION
        self.teleop_path = self.base_path / "teleoperators"
        self.robot_path = self.base_path / "robots"

        # Ensure paths exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.teleop_path.mkdir(parents=True, exist_ok=True)
        self.robot_path.mkdir(parents=True, exist_ok=True)

    def get_calibration_structure(self) -> dict[str, dict[str, list[str]]]:
        """Get the complete structure of calibration files"""
        structure: dict[str, dict[str, list[str]]] = {"teleoperators": {}, "robots": {}}

        for device_type in ["teleoperators", "robots"]:
            device_path = self.base_path / device_type
            if not device_path.exists():
                continue

            for model_dir in device_path.iterdir():
                if model_dir.is_dir():
                    calibrations = []
                    for calib_file in model_dir.glob("*.json"):
                        calibrations.append(calib_file.stem)  # filename without .json
                    if calibrations:
                        structure[device_type][model_dir.name] = sorted(calibrations)

        return structure

    def get_calibration_path(self, device_type: str, device_model: str, device_id: str) -> Path:
        """Get the full path to a calibration file"""
        return self.base_path / device_type / device_model / f"{device_id}.json"

    def calibration_exists(self, device_type: str, device_model: str, device_id: str) -> bool:
        """Check if a calibration file exists"""
        return self.get_calibration_path(device_type, device_model, device_id).exists()

    def load_calibration(self, device_type: str, device_model: str, device_id: str) -> dict[str, Any] | None:
        """Load calibration data from file"""
        calib_path = self.get_calibration_path(device_type, device_model, device_id)

        if not calib_path.exists():
            return None

        try:
            with open(calib_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading calibration {calib_path}: {e}")
            return None

    def save_calibration(self, device_type: str, device_model: str, device_id: str, data: dict[str, Any]) -> bool:
        """Save calibration data to file"""
        calib_path = self.get_calibration_path(device_type, device_model, device_id)

        # Ensure parent directory exists
        calib_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(calib_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving calibration {calib_path}: {e}")
            return False

    def delete_calibration(self, device_type: str, device_model: str, device_id: str) -> bool:
        """Delete a calibration file"""
        calib_path = self.get_calibration_path(device_type, device_model, device_id)

        if not calib_path.exists():
            return False

        try:
            calib_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error deleting calibration {calib_path}: {e}")
            return False

    def get_calibration_info(self, device_type: str, device_model: str, device_id: str) -> dict[str, Any] | None:
        """Get detailed information about a calibration file"""
        calib_path = self.get_calibration_path(device_type, device_model, device_id)

        if not calib_path.exists():
            return None

        try:
            stat = calib_path.stat()
            data = self.load_calibration(device_type, device_model, device_id)

            info = {
                "path": str(calib_path),
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime),
                "device_type": device_type,
                "device_model": device_model,
                "device_id": device_id,
                "data": data,
            }

            if data and isinstance(data, dict):
                info["motor_count"] = len(data)
                info["motor_names"] = list(data.keys())

            return info

        except Exception as e:
            logger.error(f"Error getting calibration info {calib_path}: {e}")
            return None

    def search_calibrations(
        self, query: str = "", device_type: str | None = None, device_model: str | None = None
    ) -> list[dict[str, Any]]:
        """Search calibrations by various criteria"""
        results = []
        structure = self.get_calibration_structure()

        for dev_type, models in structure.items():
            if device_type and device_type != dev_type:
                continue

            for model, calibrations in models.items():
                if device_model and device_model != model:
                    continue

                for calib_id in calibrations:
                    # Check if query matches device_id, model, or type
                    if not query or (
                        query.lower() in calib_id.lower()
                        or query.lower() in model.lower()
                        or query.lower() in dev_type.lower()
                    ):
                        info = self.get_calibration_info(dev_type, model, calib_id)
                        if info:
                            results.append(info)

        return sorted(results, key=lambda x: x["modified_time"], reverse=True)

    def backup_calibrations(
        self,
        output_dir: Path | None = None,
        device_type: str | None = None,
        device_model: str | None = None,
        device_id: str | None = None,
    ) -> tuple[bool, str, int]:
        """Backup calibration files"""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = BACKUP_DIR / f"backup_{timestamp}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        structure = self.get_calibration_structure()
        copied_count = 0

        try:
            for dev_type, models in structure.items():
                if device_type and device_type != dev_type:
                    continue

                for model, calibrations in models.items():
                    if device_model and device_model != model:
                        continue

                    for calib_id in calibrations:
                        if device_id and device_id != calib_id:
                            continue

                        source_file = self.get_calibration_path(dev_type, model, calib_id)
                        dest_dir = output_dir / dev_type / model
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_file = dest_dir / f"{calib_id}.json"

                        shutil.copy2(source_file, dest_file)
                        copied_count += 1

            # Create backup manifest
            manifest = {
                "backup_date": datetime.now().isoformat(),
                "source_path": str(self.base_path),
                "filter_device_type": device_type,
                "filter_device_model": device_model,
                "filter_device_id": device_id,
                "files_count": copied_count,
                "structure": structure,
            }

            with open(output_dir / "backup_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            return True, str(output_dir), copied_count

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False, str(e), copied_count

    def restore_calibrations(self, backup_dir: Path, overwrite: bool = False) -> tuple[bool, str, int]:
        """Restore calibrations from backup"""
        backup_dir = Path(backup_dir)

        if not backup_dir.exists():
            return False, f"Backup directory not found: {backup_dir}", 0

        restored_count = 0

        try:
            for device_type in ["teleoperators", "robots"]:
                type_dir = backup_dir / device_type
                if not type_dir.exists():
                    continue

                for model_dir in type_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    for calib_file in model_dir.glob("*.json"):
                        dest_file = self.get_calibration_path(device_type, model_dir.name, calib_file.stem)

                        if dest_file.exists() and not overwrite:
                            continue  # Skip existing files unless overwrite is True

                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(calib_file, dest_file)
                        restored_count += 1

            return True, f"Successfully restored {restored_count} calibrations", restored_count

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False, str(e), restored_count


@tool
def lerobot_calibrate(
    action: str = "list",
    device_type: str | None = None,
    device_model: str | None = None,
    device_id: str | None = None,
    query: str | None = None,
    output_dir: str | None = None,
    backup_dir: str | None = None,
    overwrite: bool = False,
    base_path: str | None = None,
    format_output: str = "rich",
) -> dict[str, Any]:
    """
    Advanced LeRobot calibration management tool.

    This tool provides comprehensive management of LeRobot calibration files,
    allowing users to list, view, backup, restore, and analyze calibration data.

    Actions:
        list: List all calibrations with filtering options
        view: View detailed calibration information
        search: Search calibrations by query
        backup: Backup calibrations to specified directory
        restore: Restore calibrations from backup
        delete: Delete a specific calibration
        analyze: Analyze calibration statistics
        path: Show calibration file paths

    Device Types:
        - teleoperators: Teleoperation leader devices
        - robots: Robot follower devices

    Common Device Models:
        - so101_leader: SO-101 leader arm
        - so101_follower: SO-101 follower arm
        - bi_so100_leader: Dual SO-100 leader arms
        - bi_so100_follower: Dual SO-100 follower arms
        - koch_leader/koch_follower: Koch arms
        - hope_jr: HOPE Jr robot

    Examples:
        # List all calibrations
        lerobot_calibrate(action="list")

        # List only robot calibrations
        lerobot_calibrate(action="list", device_type="robots")

        # View specific calibration
        lerobot_calibrate(
            action="view",
            device_type="robots",
            device_model="so101_follower",
            device_id="orange_arm"
        )

        # Search calibrations
        lerobot_calibrate(action="search", query="orange")

        # Backup all calibrations
        lerobot_calibrate(action="backup", output_dir="./my_backup")

        # Backup specific device type
        lerobot_calibrate(
            action="backup",
            device_type="teleoperators",
            output_dir="./teleop_backup"
        )

        # Restore from backup
        lerobot_calibrate(action="restore", backup_dir="./my_backup")

        # Delete calibration
        lerobot_calibrate(
            action="delete",
            device_type="robots",
            device_model="so101_follower",
            device_id="old_calibration"
        )

    Args:
        action: Action to perform (list, view, search, backup, restore, delete, analyze, path)
        device_type: Filter by device type (teleoperators, robots)
        device_model: Filter by device model (so101_leader, so101_follower, etc.)
        device_id: Specific device ID for view/delete operations
        query: Search query for search action
        output_dir: Output directory for backup action
        backup_dir: Backup directory for restore action
        overwrite: Whether to overwrite existing files during restore
        base_path: Custom base path for calibrations (default: ~/.cache/huggingface/lerobot/calibration)
        format_output: Output format (rich, simple, json)

    Returns:
        Dict with operation status and results
    """

    try:
        # Initialize calibration manager
        manager = LeRobotCalibrationManager(Path(base_path) if base_path else None)

        if action == "list":
            structure = manager.get_calibration_structure()

            if not any(structure.values()):
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"ℹ️ **No calibration files found.**\n\nCalibrations are stored in: `{manager.base_path}`"
                        }
                    ],
                    "calibrations": structure,
                    "count": 0,
                }

            # Format output
            content_lines = ["🔧 **LeRobot Calibrations**", f"📍 Location: `{manager.base_path}`", ""]
            total_count = 0

            for dev_type, models in structure.items():
                if device_type and device_type != dev_type:
                    continue

                if not models:
                    continue

                content_lines.append(f"## 📁 **{dev_type.title()}**")

                for model, calibrations in models.items():
                    if device_model and device_model != model:
                        continue

                    content_lines.append(f"### 🤖 **{model}** ({len(calibrations)} calibrations)")

                    for calib_id in calibrations:
                        info = manager.get_calibration_info(dev_type, model, calib_id)
                        if info:
                            modified = info["modified_time"].strftime("%Y-%m-%d %H:%M:%S")
                            size_kb = info["size_bytes"] / 1024
                            motor_info = f"{info.get('motor_count', 0)} motors" if info.get("motor_count") else ""
                            content_lines.append(f"  - `{calib_id}` *({modified}, {size_kb:.1f}KB, {motor_info})*")
                            total_count += 1
                        else:
                            content_lines.append(f"  - `{calib_id}` *(error reading file)*")

                    content_lines.append("")

            return {
                "status": "success",
                "content": [{"text": "\n".join(content_lines)}],
                "calibrations": structure,
                "count": total_count,
            }

        elif action == "view":
            if not all([device_type, device_model, device_id]):
                return {
                    "status": "error",
                    "content": [{"text": "❌ **view** action requires: device_type, device_model, and device_id"}],
                }

            assert device_type is not None and device_model is not None and device_id is not None
            info = manager.get_calibration_info(device_type, device_model, device_id)
            if not info:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Calibration not found: `{device_type}/{device_model}/{device_id}`"}],
                }

            content_lines = [
                f"🔧 **Calibration Details: `{device_type}/{device_model}/{device_id}`**",
                f"📍 **Path:** `{info['path']}`",
                f"📅 **Modified:** {info['modified_time'].strftime('%Y-%m-%d %H:%M:%S')}",
                f"📏 **Size:** {info['size_bytes']} bytes ({info['size_bytes'] / 1024:.1f} KB)",
                "",
            ]

            if info.get("data") and isinstance(info["data"], dict):
                content_lines.extend([f"🤖 **Motor Configuration** ({info.get('motor_count', 0)} motors)", ""])

                for motor_name, motor_data in info["data"].items():
                    if isinstance(motor_data, dict):
                        content_lines.extend(
                            [
                                f"### ⚙️ **{motor_name}**",
                                f"  - **ID:** {motor_data.get('id', 'N/A')}",
                                f"  - **Drive Mode:** {motor_data.get('drive_mode', 'N/A')}",
                                f"  - **Homing Offset:** {motor_data.get('homing_offset', 'N/A')}",
                                f"  - **Range:** {motor_data.get('range_min', 'N/A')}"
                                f" to {motor_data.get('range_max', 'N/A')}",
                                "",
                            ]
                        )

            return {"status": "success", "content": [{"text": "\n".join(content_lines)}], "calibration_info": info}

        elif action == "search":
            results = manager.search_calibrations(query or "", device_type, device_model)

            if not results:
                search_desc = f"query '{query}'" if query else "specified criteria"
                return {
                    "status": "success",
                    "content": [{"text": f"🔍 **No calibrations found** matching {search_desc}"}],
                    "results": [],
                    "count": 0,
                }

            content_lines = [f"🔍 **Search Results** ({len(results)} found)", f"📍 Query: `{query or 'all'}`", ""]

            for result in results:
                modified = result["modified_time"].strftime("%Y-%m-%d %H:%M:%S")
                size_kb = result["size_bytes"] / 1024
                motor_info = f"{result.get('motor_count', 0)} motors" if result.get("motor_count") else ""

                content_lines.extend(
                    [
                        f"### 🤖 **{result['device_type']}/{result['device_model']}/{result['device_id']}**",
                        f"  - **Modified:** {modified}",
                        f"  - **Size:** {size_kb:.1f} KB",
                        f"  - **Motors:** {motor_info}",
                        f"  - **Path:** `{result['path']}`",
                        "",
                    ]
                )

            return {
                "status": "success",
                "content": [{"text": "\n".join(content_lines)}],
                "results": results,
                "count": len(results),
            }

        elif action == "backup":
            output_path = Path(output_dir) if output_dir else None
            success, message, count = manager.backup_calibrations(output_path, device_type, device_model, device_id)

            if success:
                content_lines = [
                    "💾 **Backup Completed Successfully**",
                    f"📁 **Location:** `{message}`",
                    f"📊 **Files copied:** {count}",
                    "",
                ]

                if device_type or device_model or device_id:
                    content_lines.append("🔍 **Filters applied:**")
                    if device_type:
                        content_lines.append(f"  - Device Type: `{device_type}`")
                    if device_model:
                        content_lines.append(f"  - Device Model: `{device_model}`")
                    if device_id:
                        content_lines.append(f"  - Device ID: `{device_id}`")

                return {
                    "status": "success",
                    "content": [{"text": "\n".join(content_lines)}],
                    "backup_path": message,
                    "files_count": count,
                }
            else:
                return {"status": "error", "content": [{"text": f"❌ **Backup failed:** {message}"}]}

        elif action == "restore":
            if not backup_dir:
                return {"status": "error", "content": [{"text": "❌ **restore** action requires: backup_dir"}]}

            success, message, count = manager.restore_calibrations(Path(backup_dir), overwrite)

            if success:
                return {
                    "status": "success",
                    "content": [
                        {"text": f"✅ **{message}**\n📁 From: `{backup_dir}`\n🔄 Overwrite mode: `{overwrite}`"}
                    ],
                    "restored_count": count,
                }
            else:
                return {"status": "error", "content": [{"text": f"❌ **Restore failed:** {message}"}]}

        elif action == "delete":
            if not all([device_type, device_model, device_id]):
                return {
                    "status": "error",
                    "content": [{"text": "❌ **delete** action requires: device_type, device_model, and device_id"}],
                }

            assert device_type is not None and device_model is not None and device_id is not None
            if not manager.calibration_exists(device_type, device_model, device_id):
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Calibration not found: `{device_type}/{device_model}/{device_id}`"}],
                }

            success = manager.delete_calibration(device_type, device_model, device_id)

            if success:
                return {
                    "status": "success",
                    "content": [{"text": f"🗑️ **Successfully deleted:** `{device_type}/{device_model}/{device_id}`"}],
                }
            else:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ **Failed to delete:** `{device_type}/{device_model}/{device_id}`"}],
                }

        elif action == "analyze":
            structure = manager.get_calibration_structure()

            if not any(structure.values()):
                return {"status": "success", "content": [{"text": "📊 **No calibrations to analyze**"}], "analysis": {}}

            total_calibrations = 0
            device_counts = {"teleoperators": 0, "robots": 0}
            model_stats = {}
            motor_stats = {}

            for dev_type, models in structure.items():
                for model, calibrations in models.items():
                    device_counts[dev_type] += len(calibrations)
                    total_calibrations += len(calibrations)

                    key = f"{dev_type}/{model}"
                    model_stats[key] = len(calibrations)

                    # Analyze motor configurations
                    motor_counts = []
                    for calib_id in calibrations:
                        info = manager.get_calibration_info(dev_type, model, calib_id)
                        if info and info.get("motor_count"):
                            motor_counts.append(info["motor_count"])

                    if motor_counts:
                        motor_stats[key] = {
                            "min": min(motor_counts),
                            "max": max(motor_counts),
                            "avg": sum(motor_counts) / len(motor_counts),
                        }

            content_lines = [
                "📊 **Calibration Analysis**",
                f"📍 **Base Path:** `{manager.base_path}`",
                "",
                "### 📈 **Summary Statistics**",
                f"  - **Total Calibrations:** {total_calibrations}",
                f"  - **Teleoperators:** {device_counts['teleoperators']}",
                f"  - **Robots:** {device_counts['robots']}",
                f"  - **Device Models:** {len(model_stats)}",
                "",
            ]

            if model_stats:
                content_lines.extend(["### 🤖 **Device Model Breakdown**"])
                for model_key, count in sorted(model_stats.items()):
                    motor_info = ""
                    if model_key in motor_stats:
                        stats = motor_stats[model_key]
                        motor_info = f"(avg {stats['avg']:.1f} motors)"
                    content_lines.append(f"  - **{model_key}:** {count} calibrations {motor_info}")

            analysis = {
                "total_calibrations": total_calibrations,
                "device_counts": device_counts,
                "model_stats": model_stats,
                "motor_stats": motor_stats,
                "base_path": str(manager.base_path),
            }

            return {"status": "success", "content": [{"text": "\n".join(content_lines)}], "analysis": analysis}

        elif action == "path":
            if device_type and device_model and device_id:
                # Show specific calibration path
                calib_path = manager.get_calibration_path(device_type, device_model, device_id)
                exists = calib_path.exists()

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"📍 **Calibration Path**\n`{calib_path}`\n\n"
                            f"{'✅ File exists' if exists else '❌ File does not exist'}"
                        }
                    ],
                    "path": str(calib_path),
                    "exists": exists,
                }
            else:
                # Show base paths
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"📍 **LeRobot Calibration Paths**\n\n"
                            f"**Base:** `{manager.base_path}`\n"
                            f"**Teleoperators:** `{manager.teleop_path}`\n"
                            f"**Robots:** `{manager.robot_path}`"
                        }
                    ],
                    "base_path": str(manager.base_path),
                    "teleop_path": str(manager.teleop_path),
                    "robot_path": str(manager.robot_path),
                }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"❌ **Unknown action:** `{action}`\n\n"
                        "Available actions: list, view, search, backup, restore, delete, analyze, path"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"LeRobot calibrate tool error: {e}")
        return {"status": "error", "content": [{"text": f"❌ **Tool execution failed:** {str(e)}"}]}
