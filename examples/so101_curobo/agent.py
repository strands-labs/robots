# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strands agent driving the SO-101 cuRobo synthetic-data demo (issue #67 T9).

Exposes a small set of custom ``@tool`` functions over :class:`SO101CuroboDemo`
(natural-language -> plan/execute/record). Returns ``None`` (chat disabled) if
``strands-agents`` or the LLM backend isn't available, so the Gradio app still
runs buttons-only.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("so101_curobo.agent")


def build_agent(demo, model_id: str | None = None) -> Any:
    """Build a Strands ``Agent`` bound to a built :class:`SO101CuroboDemo`."""
    try:
        from strands import Agent, tool
    except Exception as exc:  # noqa: BLE001
        logger.warning("strands-agents unavailable (%s); agent chat disabled.", exc)
        return None

    cameras = list(getattr(demo.scene, "cameras", []) or [])

    @tool
    def plan_and_execute(task: str = "pick up the red cube and place it in the bin") -> str:
        """Plan a pick-and-place for the described task, execute it, and record one episode."""
        return demo.plan_and_execute(task=task)

    @tool
    def record_dataset(n_episodes: int = 5) -> str:
        """Generate ``n_episodes`` of synthetic pick-and-place data into a LeRobot dataset."""
        n = max(1, min(int(n_episodes), 100))
        summary = demo.record_dataset(n_episodes=n)
        if summary.get("status") != "success":
            return f"Could not record dataset: {summary.get('message', summary)}"
        return (
            f"Recorded {summary['episodes']} episodes ({summary['total_frames']} frames, "
            f"planner={summary['planner']}, success_rate={summary['success_rate']:.0%}) "
            f"to LeRobot dataset {summary['repo_id']}."
        )

    @tool
    def move_camera(view: str) -> str:
        """Switch the preview camera. ``view`` is one of the scene's camera presets."""
        return demo.set_camera((view or "").strip().lower())

    @tool
    def describe_scene() -> str:
        """Describe the scene, planner, and which optional runtimes are present."""
        return demo.describe()

    system_prompt = (
        "You orchestrate an SO-101 tabletop pick-and-place + synthetic-data demo "
        "(MuJoCo backend today; Isaac Sim + cuRobo when installed). The scene is "
        "ALREADY built (an SO-101 arm, a red cube, a bin, and cameras) -- never "
        "recreate or reset it.\n\n"
        "Tools (use ONLY these):\n"
        "  - plan_and_execute(task): plan + run a pick-and-place, record one episode.\n"
        "  - record_dataset(n_episodes): generate N episodes of training data.\n"
        f"  - move_camera(view): one of {cameras}.\n"
        "  - describe_scene(): scene + planner + dependency status.\n\n"
        "If cuRobo/Isaac aren't installed the demo uses a scripted planner on "
        "MuJoCo (motion is demonstrative, grasps aren't guaranteed) -- say so "
        "briefly if asked. Keep replies to one or two sentences. Be concise."
    )

    kwargs: dict = {
        "tools": [plan_and_execute, record_dataset, move_camera, describe_scene],
        "system_prompt": system_prompt,
    }
    if model_id:
        kwargs["model"] = model_id
    try:
        agent = Agent(**kwargs)
        logger.info("SO-101 cuRobo agent ready (model=%s).", model_id or "default")
        return agent
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build agent (%s); chat disabled.", exc)
        return None


def extract_text(result: Any) -> str:
    """Best-effort extraction of the assistant's text from a Strands result."""
    if result is None:
        return ""
    msg = getattr(result, "message", None)
    if msg is not None:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
            if texts:
                return "\n".join(t for t in texts if t)
    if isinstance(result, str):
        return result
    return str(result)
