import importlib
import re
import subprocess
import sys
from pathlib import Path


def test_the_package_imports_with_nothing_heavy() -> None:
    heavy = ("mujoco", "torch", "lerobot", "zenoh", "isaacsim", "warp", "onnxruntime", "huggingface_hub")
    for name in [m for m in sys.modules if m.split(".")[0] in heavy]:
        del sys.modules[name]
    pkg = importlib.import_module("strands_robots")
    assert pkg.__version__.startswith("1.0.0")
    assert not [m for m in sys.modules if m.split(".")[0] in heavy]


def test_simulation_without_mujoco_says_how_to_install() -> None:
    probe = (
        "import sys; sys.modules['mujoco'] = None\n"
        "from strands_robots import Robot\n"
        "try:\n    Robot('so101')\nexcept ImportError as e:\n    print(e)\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert out.startswith("Simulation() needs MuJoCo: pip install 'strands-robots[mujoco]'"), out


def test_the_wheel_is_built_with_the_version_the_package_reports() -> None:
    # One copy: a static [project].version beside __version__ is a second one, and the usual bump
    # (`hatch version 1.0.0rc1`) edits only that, so the 1.0.0rc1 wheel's strands_robots.__version__ says 1.0.0.dev0.
    # Read as text: tomllib is 3.11+, and the package supports 3.10.
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^dynamic = \["version"\]$', text, re.M) and not re.search(r"^version = ", text, re.M)
    assert '\n[tool.hatch.version]\npath = "strands_robots/__init__.py"\n' in text
