"""A Robot is a body in a world or on a bus: observe / act / run, the same verbs for both."""

from __future__ import annotations

from .body import Body, SimBody
from .robot import Robot
from .run.result import RunResult
from .run.source import from_agent
from .run.task import Task

__all__ = ["Robot", "Body", "SimBody", "Task", "RunResult", "from_agent"]
