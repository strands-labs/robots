"""Single source of truth for Isaac Sim install metadata.

Centralises the docker image tag, Omniverse Launcher hint, and Isaac Lab
bootstrap command so they don't drift across docstrings and error
messages whenever the upstream image is bumped (security update or
otherwise). Update :data:`ISAAC_SIM_DOCKER_IMAGE` here when the
supported image tag changes; everything that surfaces install hints --
``IsaacSimulation.is_available()``, the ``ImportError`` raised by
``_get_or_create_simulation_app``, and the package docstring -- composes
its message from these constants.

Maintainers only: bumping these values is a deliberate compatibility
decision. Bump the constant, run the test suite (the
``test_install_constants`` cases pin format expectations), and note the
change in the release notes.
"""

from __future__ import annotations

# --- Canonical install sources -------------------------------------------------

#: Lowest Isaac Sim major.minor we attempt to support. Surfaced in
#: error messages so a user on an older Omniverse Launcher install knows
#: to upgrade. 6.0 is the floor because it's the only Isaac Sim release
#: targeting Python 3.12, which ``strands-robots>=0.4.0`` requires (#98).
ISAAC_SIM_MIN_VERSION: str = "6.0"

#: Pinned NVIDIA NGC docker image. Bump when CI / docs validate a newer
#: tag (security patches, kit-sdk minor bumps, etc.).
ISAAC_SIM_DOCKER_IMAGE: str = "nvcr.io/nvidia/isaac-sim:6.0"

#: One-liner to bootstrap an Isaac Lab checkout. Kept as a single
#: string so callers don't have to assemble it.
ISAAC_LAB_BOOTSTRAP: str = "git clone IsaacLab && ./isaaclab.sh -i"

#: Pip install command for the Isaac Sim runtime itself. Viable since the
#: cp312 wheels shipped for Isaac Sim 6.0.x (#1803); the ``extscache``
#: extra is REQUIRED - the bare ``isaacsim[all]`` metapackage omits the
#: ``isaacsim-extscache-*`` packages and ``SimulationApp`` aborts resolving
#: its extension graph without them.
ISAAC_SIM_PIP_INSTALL: str = "pip install 'isaacsim[all,extscache]==6.0.*' --extra-index-url https://pypi.nvidia.com"

#: Caveats that apply only to the pip route (#1803): isaacsim-kernel
#: downgrades ``coverage`` to 7.4.4, which breaks numba (and hence
#: robosuite/LIBERO) with a red-herring ``coverage.types.Tracer``
#: AttributeError; and the first non-interactive import hangs on the EULA
#: prompt without the env var.
ISAAC_SIM_PIP_CAVEATS: str = (
    "then reinstall 'pip install coverage>=7.6.1' and set OMNI_KIT_ACCEPT_EULA=YES for the first import"
)

#: Pip extra users install to pull our Python helpers alongside an
#: out-of-band Isaac Sim install.
PIP_EXTRA: str = "pip install 'strands-robots[sim-isaac]'"


# --- Composed messages ---------------------------------------------------------


def install_options_block(indent: str = "  - ") -> str:
    """Return a multi-line bullet block enumerating supported install paths.

    Used by :meth:`IsaacSimulation.is_available` as the body of the
    "not importable" reason string. Single source so docstring and
    runtime error stay in lockstep.
    """
    lines = [
        f"{indent}pip (Python 3.12): {ISAAC_SIM_PIP_INSTALL} ({ISAAC_SIM_PIP_CAVEATS})",
        f"{indent}NVIDIA Omniverse Launcher (Isaac Sim {ISAAC_SIM_MIN_VERSION}+)",
        f"{indent}Isaac Lab: {ISAAC_LAB_BOOTSTRAP}",
        f"{indent}Docker: {ISAAC_SIM_DOCKER_IMAGE}",
    ]
    return "\n".join(lines)


def install_options_inline() -> str:
    """Return a one-line variant of the install options.

    Used by the ``ImportError`` raised in
    ``_get_or_create_simulation_app`` where a multi-line block reads
    awkwardly inside a single sentence.
    """
    return (
        "Isaac Sim must be installed first - via pip "
        "(isaacsim[all,extscache], Python 3.12), Omniverse Launcher, "
        f"Isaac Lab ({ISAAC_LAB_BOOTSTRAP.split(' && ')[-1]}), "
        f"or Docker ({ISAAC_SIM_DOCKER_IMAGE})."
    )


def not_importable_reason() -> str:
    """Full reason string returned by ``is_available()`` when neither
    the legacy ``omni.isaac.kit`` nor the modern ``isaacsim`` SimulationApp
    entry point can be located.
    """
    return (
        "omni.isaac.kit.SimulationApp / isaacsim.SimulationApp not importable. "
        f"Install Isaac Sim {ISAAC_SIM_MIN_VERSION}+ via one of:\n"
        f"{install_options_block()}\n"
        f"Then install the Python helpers: {PIP_EXTRA}"
    )


def not_available_import_error() -> str:
    """Message for the ``ImportError`` raised when SimulationApp can't
    be constructed at runtime.
    """
    return f"omni.isaac.kit.SimulationApp / isaacsim.SimulationApp not available. {install_options_inline()}"


__all__ = [
    "ISAAC_SIM_MIN_VERSION",
    "ISAAC_SIM_DOCKER_IMAGE",
    "ISAAC_LAB_BOOTSTRAP",
    "ISAAC_SIM_PIP_INSTALL",
    "ISAAC_SIM_PIP_CAVEATS",
    "PIP_EXTRA",
    "install_options_block",
    "install_options_inline",
    "not_importable_reason",
    "not_available_import_error",
]
