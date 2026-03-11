"""Base classes and utilities for training providers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _discover_nvidia_cuda_lib_paths() -> list:
    """Discover pip-installed NVIDIA CUDA shared library directories.

    On systems where the system CUDA toolkit version differs from what
    packages like ``transformer_engine`` or ``megatron-core`` were compiled
    against, the required shared libraries (e.g. ``libcublas.so.12``) are
    often pip-installed under ``site-packages/nvidia/*/lib/``.

    This function discovers those directories so they can be added to
    ``LD_LIBRARY_PATH`` for subprocesses.

    Returns:
        List of directory paths containing NVIDIA CUDA shared libraries.
    """
    import glob
    import os
    import sys

    nvidia_lib_dirs = []

    # Check all site-packages directories (system, user, venv)
    for site_dir in sys.path:
        if not os.path.isdir(site_dir):
            continue
        nvidia_base = os.path.join(site_dir, "nvidia")
        if os.path.isdir(nvidia_base):
            # Pattern: nvidia/*/lib/ (e.g. nvidia/cublas/lib/, nvidia/cuda_runtime/lib/)
            for lib_dir in glob.glob(os.path.join(nvidia_base, "*", "lib")):
                if os.path.isdir(lib_dir):
                    nvidia_lib_dirs.append(lib_dir)

    # Also check ~/.local/lib/pythonX.Y/site-packages/nvidia/*/lib/
    user_sp = os.path.expanduser(
        f"~/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}"
        f"/site-packages/nvidia"
    )
    if os.path.isdir(user_sp):
        for lib_dir in glob.glob(os.path.join(user_sp, "*", "lib")):
            if os.path.isdir(lib_dir) and lib_dir not in nvidia_lib_dirs:
                nvidia_lib_dirs.append(lib_dir)

    return nvidia_lib_dirs


@dataclass
class TrainConfig:
    """Universal training configuration."""

    dataset_path: str = ""
    output_dir: str = "./outputs"
    max_steps: int = 10000
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    num_gpus: int = 1
    save_steps: int = 1000
    save_total_limit: int = 5
    use_wandb: bool = False
    dataloader_num_workers: int = 2
    seed: int = 42
    resume: bool = False


class Trainer(ABC):
    """Abstract base class for VLA model trainers."""

    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """Run the training loop.

        Returns:
            Dict with training results (loss, steps, checkpoint_path, etc.)
        """
        pass

    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """Run evaluation.

        Returns:
            Dict with evaluation metrics
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get trainer provider name."""
        pass


def _build_cosmos_subprocess_env(
    train_script_path: Optional[str],
    extra_env_vars: Optional[list] = None,
) -> dict:
    """Build environment variables for Cosmos training subprocesses.

    Shared by CosmosTrainer and CosmosTransferTrainer. Ensures the repo
    root and sub-packages are on PYTHONPATH, and CUDA shared libraries
    are findable.

    Args:
        train_script_path: Resolved path to the training script.
        extra_env_vars: Additional env var names to check for repo root.
    """
    import os

    env = os.environ.copy()

    # Derive repo root from script path
    repo_root = None
    if train_script_path and os.path.isfile(train_script_path):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(train_script_path)))

    if repo_root is None and extra_env_vars:
        for env_var in extra_env_vars:
            repo_root = os.environ.get(env_var)
            if repo_root:
                break

    if repo_root and os.path.isdir(repo_root):
        existing = env.get("PYTHONPATH", "")
        extra_paths = [repo_root]
        for subpkg in ("packages/cosmos-oss", "packages/cosmos-cuda"):
            subpkg_path = os.path.join(repo_root, subpkg)
            if os.path.isdir(subpkg_path):
                extra_paths.append(subpkg_path)
        new_pythonpath = os.pathsep.join(extra_paths)
        if existing:
            new_pythonpath = new_pythonpath + os.pathsep + existing
        env["PYTHONPATH"] = new_pythonpath
        logger.info("✅ Cosmos subprocess PYTHONPATH prepended: %s", repo_root)

    # Ensure CUDA shared libraries are findable
    nvidia_pip_paths = _discover_nvidia_cuda_lib_paths()

    cuda_lib_candidates = [
        os.environ.get("CUDA_HOME", "/usr/local/cuda") + "/lib64",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    system_cuda_paths = [p for p in cuda_lib_candidates if os.path.isdir(p)]
    cuda_paths = nvidia_pip_paths + system_cuda_paths

    if cuda_paths:
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        new_ld = os.pathsep.join(cuda_paths)
        if existing_ld:
            new_ld = new_ld + os.pathsep + existing_ld
        env["LD_LIBRARY_PATH"] = new_ld

    return env
