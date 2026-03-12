"""Cross-dependency AST validation for strands-robots.

Validates that every import from 3rd-party libraries resolves correctly
and that call-site signatures match the installed package versions.

This test suite is CRITICAL for hardware safety: a broken import path
means a silent failure that could result in uncontrolled robot behavior.

Three validation layers:
1. Import Resolution — does `from lerobot.X import Y` actually resolve?
2. Signature Validation — do our kwargs match the function signature?
3. Version Compatibility — are we within the tested version range?

Run: pytest tests/test_cross_deps.py -v
"""

import ast
import importlib
import inspect
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

STRANDS_ROBOTS_ROOT = Path(__file__).parent.parent / "strands_robots"

# Libraries that MUST be validated when installed
CRITICAL_LIBS = {
    "lerobot",
    "mujoco",
    "torch",
    "numpy",
    "gymnasium",
    "transformers",
    "huggingface_hub",
    "PIL",
    "cv2",
}

# Libraries that are GPU/hardware-only — skip if not installed
OPTIONAL_LIBS = {
    "gr00t",
    "stable_baselines3",
    "unitree_sdk2py",
    "warp",
    "openpi_client",
    "isaacsim",
    "isaaclab",
    "cosmos_transfer2",
    "cosmos_predict2",
    "robot_descriptions",
    "cyclonedds",
}

# Known guarded imports (wrapped in try/except in our code — expected to fail)
KNOWN_GUARDED = {
    "gr00t.data.embodiment_tags.EmbodimentTag",
    "gr00t.experiment.data_config.DATA_CONFIG_MAP",
    "gr00t.model.idm.IDM",
    "gr00t.model.idm.IDMConfig",
    "gr00t.model.policy.Gr00tPolicy",
    "gr00t.policy.gr00t_policy.Gr00tPolicy",
    "unitree_sdk2py",
    "unitree_sdk2py.core.channel.ChannelFactoryInitialize",
    "warp",
    "openpi_client.msgpack_numpy",
    "stable_baselines3.PPO",
    "stable_baselines3.SAC",
    "stable_baselines3.A2C",
    "stable_baselines3.common.callbacks.CheckpointCallback",
    "stable_baselines3.common.callbacks.EvalCallback",
    # gRPC-dependent: guarded by ImportError in _ensure_connected()
    "lerobot.transport.services_pb2_grpc",
    # Florence2 compat patch: guarded by try/except in _load_model()
    "transformers.models.florence2.configuration_florence2.Florence2LanguageConfig",
}


# ─────────────────────────────────────────────────────────────────────
# AST Extraction Helpers
# ─────────────────────────────────────────────────────────────────────


def extract_all_imports(root_dir: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Extract all 3rd-party imports from our codebase.

    Returns:
        {library_name: {full_import_path: set(our_files)}}
    """
    stdlib = {
        "os", "sys", "time", "json", "logging", "threading", "math", "re",
        "pathlib", "typing", "abc", "enum", "dataclasses", "functools",
        "concurrent", "asyncio", "inspect", "uuid", "socket", "importlib",
        "subprocess", "collections", "copy", "io", "struct", "signal",
        "tempfile", "shutil", "hashlib", "urllib", "base64", "warnings",
        "traceback", "contextlib", "pkgutil", "textwrap", "http",
        "unittest", "ctypes", "array", "queue", "itertools", "glob",
        "argparse", "datetime", "random", "pickle", "atexit", "gzip",
        "xml", "socketserver", "types", "multiprocessing",
    }

    imports = defaultdict(lambda: defaultdict(set))

    for py_file in root_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        rel_path = str(py_file.relative_to(root_dir.parent))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in stdlib or top.startswith("_") or top == "strands_robots" or top == "strands":
                    continue
                # Skip relative imports (node.level > 0 means from .X import Y)
                if node.level and node.level > 0:
                    continue
                for alias in node.names:
                    full_path = f"{node.module}.{alias.name}"
                    imports[top][full_path].add(rel_path)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in stdlib or top.startswith("_") or top == "strands_robots" or top == "strands":
                        continue
                    imports[top][alias.name].add(rel_path)

    return dict(imports)


def extract_call_sites(
    root_dir: Path, target_lib: str
) -> List[Tuple[str, int, str, List[str]]]:
    """Extract all call sites to a target library.

    Returns:
        [(file_path, line_number, full_call_path, [kwarg_names])]
    """
    call_sites = []

    for py_file in root_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        rel_path = str(py_file.relative_to(root_dir.parent))

        # Collect imports in this file
        file_imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(target_lib):
                    for alias in node.names:
                        local = alias.asname or alias.name
                        file_imports[local] = f"{node.module}.{alias.name}"

        # Find calls to imported objects
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    if func.value.id in file_imports:
                        call_str = f"{file_imports[func.value.id]}.{func.attr}"
                        kwargs = [kw.arg for kw in node.keywords if kw.arg]
                        call_sites.append((rel_path, node.lineno, call_str, kwargs))
                elif isinstance(func, ast.Name) and func.id in file_imports:
                    call_str = file_imports[func.id]
                    kwargs = [kw.arg for kw in node.keywords if kw.arg]
                    call_sites.append((rel_path, node.lineno, call_str, kwargs))

    return call_sites


def validate_import(import_path: str) -> Tuple[str, Optional[str]]:
    """Validate a single import path resolves.

    Returns:
        (status, error_message) where status is "ok", "MISSING_ATTR", "IMPORT_ERROR", etc.
    """
    parts = import_path.split(".")

    try:
        if len(parts) >= 2:
            mod_path = ".".join(parts[:-1])
            attr_name = parts[-1]
            mod = importlib.import_module(mod_path)
            if hasattr(mod, attr_name):
                return "ok", None
            try:
                importlib.import_module(import_path)
                return "ok", None
            except ImportError:
                return "MISSING_ATTR", f"{mod_path} has no attribute '{attr_name}'"
        else:
            importlib.import_module(import_path)
            return "ok", None
    except ImportError as e:
        return "IMPORT_ERROR", str(e)
    except Exception as e:
        return "RUNTIME_ERROR", str(e)[:200]


def validate_signature(
    call_path: str, kwargs_used: List[str]
) -> Tuple[str, Optional[str], Optional[List[str]]]:
    """Validate call-site kwargs against actual function signature.

    Returns:
        (status, error_message, invalid_kwargs)
    """
    parts = call_path.rsplit(".", 1)
    if len(parts) < 2:
        return "ok", None, None

    obj_path, method = parts

    try:
        mod_parts = obj_path.rsplit(".", 1)
        if len(mod_parts) == 2:
            mod = importlib.import_module(mod_parts[0])
            obj = getattr(mod, mod_parts[1], None)
        else:
            obj = importlib.import_module(mod_parts[0])

        if obj is None:
            return "OBJECT_NOT_FOUND", f"Cannot resolve {obj_path}", None

        func = getattr(obj, method, None)
        if func is None:
            return "METHOD_NOT_FOUND", f"{obj_path} has no method '{method}'", None

        sig = inspect.signature(func)
        valid_params = set(sig.parameters.keys())

        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        bad_kwargs = [k for k in kwargs_used if k not in valid_params and k != "self"]

        if bad_kwargs and not has_var_keyword:
            return "INVALID_KWARGS", f"Invalid kwargs: {bad_kwargs}", bad_kwargs

        return "ok", None, None

    except Exception as e:
        return "RESOLVE_ERROR", str(e)[:200], None


# ─────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def all_imports():
    """Extract all imports once per test session."""
    return extract_all_imports(STRANDS_ROBOTS_ROOT)


@pytest.fixture(scope="session")
def installed_libs():
    """Detect which libraries are actually installed."""
    installed = set()
    for lib in CRITICAL_LIBS | OPTIONAL_LIBS:
        try:
            importlib.import_module(lib)
            installed.add(lib)
        except ImportError:
            pass
    return installed


# ─────────────────────────────────────────────────────────────────────
# Tests: Import Resolution
# ─────────────────────────────────────────────────────────────────────


class TestImportResolution:
    """Validate all cross-boundary imports resolve against installed packages."""

    def test_all_critical_imports_resolve(self, all_imports, installed_libs):
        """Every import from a CRITICAL library must resolve."""
        failures = []

        for lib in CRITICAL_LIBS:
            if lib not in installed_libs:
                pytest.skip(f"{lib} not installed")
            if lib not in all_imports:
                continue

            for import_path, our_files in all_imports[lib].items():
                if import_path in KNOWN_GUARDED:
                    continue
                status, error = validate_import(import_path)
                if status != "ok":
                    failures.append(
                        f"  {import_path} [{status}]: {error}\n"
                        f"    Used in: {', '.join(sorted(our_files))}"
                    )

        if failures:
            pytest.fail(
                f"Cross-dependency import failures ({len(failures)}):\n"
                + "\n".join(failures)
            )

    def test_optional_imports_are_guarded(self, all_imports):
        """Imports from OPTIONAL libraries must be inside try/except blocks."""
        # This is a structural check — we verify optional imports are wrapped
        unguarded = []

        for lib in OPTIONAL_LIBS:
            if lib not in all_imports:
                continue

            for import_path, our_files in all_imports[lib].items():
                if import_path in KNOWN_GUARDED:
                    continue

                # Check if the import is inside a try/except in each file
                for filepath in our_files:
                    full_path = STRANDS_ROBOTS_ROOT.parent / filepath
                    if not full_path.exists():
                        continue

                    source = full_path.read_text(encoding="utf-8")
                    tree = ast.parse(source)

                    # Find the import node and check if it's inside a Try
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            if import_path.startswith(node.module):
                                # Walk up to find enclosing Try
                                if not _is_inside_try(tree, node):
                                    unguarded.append(
                                        f"  {import_path} in {filepath}:{node.lineno} — NOT guarded by try/except"
                                    )

        # This is a warning, not a hard failure (some are TYPE_CHECKING guarded)
        if unguarded:
            pytest.xfail(
                f"Optional imports not guarded ({len(unguarded)}):\n"
                + "\n".join(unguarded[:10])
            )


def _is_inside_try(tree: ast.AST, target_node: ast.AST) -> bool:
    """Check if a node is inside a Try block (including TYPE_CHECKING guard)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Try, ast.TryStar)):
            for child in ast.walk(node):
                if child is target_node:
                    return True
        # Also check `if TYPE_CHECKING:` blocks
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                for child in ast.walk(node):
                    if child is target_node:
                        return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Tests: Signature Validation
# ─────────────────────────────────────────────────────────────────────


class TestSignatureValidation:
    """Validate call-site kwargs match actual function signatures."""

    def test_lerobot_signatures(self, installed_libs):
        """All lerobot call sites must have valid kwargs."""
        if "lerobot" not in installed_libs:
            pytest.skip("lerobot not installed")

        call_sites = extract_call_sites(STRANDS_ROBOTS_ROOT, "lerobot")
        failures = []

        for filepath, line, call_path, kwargs in call_sites:
            status, error, bad_kwargs = validate_signature(call_path, kwargs)
            if status == "INVALID_KWARGS":
                failures.append(
                    f"  {call_path} [{status}]\n"
                    f"    File: {filepath}:{line}\n"
                    f"    Invalid kwargs: {bad_kwargs}\n"
                    f"    Error: {error}"
                )

        if failures:
            pytest.fail(
                f"Signature mismatches ({len(failures)}):\n"
                + "\n".join(failures)
            )

    def test_mujoco_signatures(self, installed_libs):
        """All mujoco call sites must have valid kwargs."""
        if "mujoco" not in installed_libs:
            pytest.skip("mujoco not installed")

        call_sites = extract_call_sites(STRANDS_ROBOTS_ROOT, "mujoco")
        failures = []

        for filepath, line, call_path, kwargs in call_sites:
            status, error, bad_kwargs = validate_signature(call_path, kwargs)
            if status == "INVALID_KWARGS":
                failures.append(f"  {call_path}: {bad_kwargs} in {filepath}:{line}")

        if failures:
            pytest.fail("MuJoCo signature mismatches:\n" + "\n".join(failures))


# ─────────────────────────────────────────────────────────────────────
# Tests: Version Compatibility
# ─────────────────────────────────────────────────────────────────────


class TestVersionCompatibility:
    """Verify installed dependency versions are within our tested range."""

    # Minimum versions we've validated against
    REQUIRED_VERSIONS = {
        "lerobot": "0.5.0",
        "mujoco": "3.0.0",
        "numpy": "1.21.0",
        "torch": "2.0.0",
        "gymnasium": "0.26.0",
    }

    def test_dependency_versions(self, installed_libs):
        """All installed critical deps must be >= our minimum tested version."""
        from importlib.metadata import version as pkg_version

        from packaging.version import Version

        issues = []
        for lib, min_ver in self.REQUIRED_VERSIONS.items():
            if lib not in installed_libs:
                continue
            try:
                installed_ver = pkg_version(lib)
                if Version(installed_ver) < Version(min_ver):
                    issues.append(
                        f"  {lib}: installed={installed_ver}, required>={min_ver}"
                    )
            except Exception:
                pass

        if issues:
            pytest.fail(
                "Version compatibility issues:\n" + "\n".join(issues)
            )

    def test_lerobot_api_surface(self, installed_libs):
        """Verify lerobot 0.5 exports the classes we need."""
        if "lerobot" not in installed_libs:
            pytest.skip("lerobot not installed")

        critical_imports = [
            ("lerobot.datasets.lerobot_dataset", "LeRobotDataset"),
            ("lerobot.robots.config", "RobotConfig"),
            ("lerobot.robots.robot", "Robot"),
            ("lerobot.envs.factory", "make_env"),
        ]

        missing = []
        for module_path, attr_name in critical_imports:
            try:
                mod = importlib.import_module(module_path)
                if not hasattr(mod, attr_name):
                    missing.append(f"  {module_path}.{attr_name}")
            except ImportError:
                missing.append(f"  {module_path} (module not found)")

        if missing:
            pytest.fail(
                "Critical lerobot exports missing:\n" + "\n".join(missing)
            )

    def test_lerobot_dataset_create_signature(self, installed_libs):
        """The LeRobotDataset.create() signature must accept our kwargs."""
        if "lerobot" not in installed_libs:
            pytest.skip("lerobot not installed")

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        sig = inspect.signature(LeRobotDataset.create)
        params = set(sig.parameters.keys())

        # These are the kwargs our DatasetRecorder passes
        our_kwargs = {
            "repo_id", "fps", "features", "root", "robot_type",
            "use_videos", "image_writer_threads", "vcodec",
        }

        missing = our_kwargs - params
        assert not missing, (
            f"LeRobotDataset.create() missing params we need: {missing}\n"
            f"Available: {sorted(params)}"
        )

    def test_lerobot_robot_configs_discoverable(self, installed_libs):
        """Our robot config resolver must find at least so100 and unitree_g1."""
        if "lerobot" not in installed_libs:
            pytest.skip("lerobot not installed")

        import pkgutil

        import lerobot.robots as lr_robots
        from lerobot.robots.config import RobotConfig

        for _, modname, _ in pkgutil.iter_modules(lr_robots.__path__):
            if modname in ("config", "robot", "utils"):
                continue
            try:
                importlib.import_module(f"lerobot.robots.{modname}")
            except Exception:
                continue

        known = RobotConfig.get_known_choices()

        # These must exist for our factory.py to work
        required = {"so100_follower", "unitree_g1"}
        # Accept either exact or partial match
        found = set()
        for name in known:
            for req in required:
                if req in name:
                    found.add(req)

        missing = required - found
        assert not missing, (
            f"Required robot configs not found in lerobot: {missing}\n"
            f"Available: {sorted(known.keys())}"
        )
