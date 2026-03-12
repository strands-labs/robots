"""Differentiable simulation — gradient-based optimization via wp.Tape."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ._core import NewtonBackend


def run_diffsim(
    backend: NewtonBackend,
    num_steps: int = 100,
    loss_fn: Optional[Callable] = None,
    lr: float = 0.02,
    iterations: int = 20,
    optimize_param: str = "joint_qd",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run differentiable simulation with gradient-based optimization.

    Phase 1.2: Full wp.Tape integration following Newton's diffsim_ball pattern.

    Args:
        num_steps: Sim steps per iteration.
        loss_fn: Loss function taking list of states.
        lr: Learning rate.
        iterations: Optimisation iterations.
        optimize_param: Parameter to optimise.
        verbose: Print progress.
    """
    if backend._model is None:
        from ._scene import finalize_model

        finalize_model(backend)

    if not backend._config.enable_differentiable:
        return {"success": False, "message": "enable_differentiable=True required."}

    wp = backend._wp
    newton = backend._newton
    dt = backend._config.physics_dt

    param = _resolve_param(backend, optimize_param)
    if param is None:
        return {"success": False, "message": f"Unknown parameter: {optimize_param}"}

    param.requires_grad = True
    loss_history = []

    for iteration in range(iterations):
        tape = wp.Tape()
        states = [backend._model.state()]

        # Create collision pipeline once per iteration
        diffsim_pipeline = None
        if backend._collision_pipeline is not None:
            try:
                diffsim_pipeline = newton.CollisionPipeline(
                    backend._model,
                    soft_contact_margin=backend._config.soft_contact_margin,
                )
            except Exception:
                pass

        with tape:
            for t in range(num_steps):
                current = states[-1]
                next_state = backend._model.state()

                if hasattr(current, "clear_forces"):
                    current.clear_forces()

                if diffsim_pipeline is not None:
                    try:
                        contacts = diffsim_pipeline.contacts()
                        diffsim_pipeline.collide(backend._model, current, contacts)
                    except Exception:
                        pass

                backend._solver.step(
                    backend._model, current, next_state, dt, backend._control
                )
                states.append(next_state)

            if loss_fn is not None:
                loss = loss_fn(states)
            else:
                loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        tape.backward(loss)
        current_loss = (
            float(loss.numpy()[0]) if hasattr(loss, "numpy") else float(loss)
        )
        loss_history.append(current_loss)

        # Gradient descent
        param_np = param.numpy()
        grad_np = (
            tape.grad(param).numpy()
            if tape.grad(param) is not None
            else np.zeros_like(param_np)
        )
        param_np -= lr * grad_np
        param.assign(
            wp.array(param_np, dtype=wp.float32, device=backend._config.device)
        )

        if verbose:
            logger.info("Iteration %d: loss=%.6f", iteration, current_loss)

        tape.zero()

    return {
        "success": True,
        "iterations": iterations,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "loss_history": loss_history,
    }


def _resolve_param(backend: NewtonBackend, optimize_param: str):
    """Resolve the parameter array to optimize."""
    param_map = {
        "joint_qd": lambda: backend._state_0.joint_qd,
        "joint_q": lambda: backend._state_0.joint_q,
        "particle_qd": lambda: backend._state_0.particle_qd,
        "particle_q": lambda: backend._state_0.particle_q,
        "joint_act": lambda: (
            backend._control.joint_act
            if hasattr(backend._control, "joint_act")
            else backend._state_0.joint_act
        ),
    }
    resolver = param_map.get(optimize_param)
    if resolver:
        return resolver()
    return getattr(backend._state_0, optimize_param, None)
