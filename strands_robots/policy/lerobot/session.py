"""One admitted LeRobot session for a ``ZenohPolicy``: connect, read the contract, admit, build the runtime.

LeRobot's ``RemoteClient`` and engine are single-use, so a policy that has lost its server opens a new
``Session`` the same way it opened the first one. A reopen is pinned to what the first session was: the same
checkpoint (LeRobot's own ``expected_artifact`` check), the same features, semantics, rate and execution.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..base import PolicyFault
from .contract import Contract
from .sequential import Sequential, ledgered


@dataclass
class Session:
    """What one admitted session runs on: its client, contract and execution, and the runtime of that execution."""

    client: Any
    contract: Contract
    execution: str
    interval: float
    seq: Sequential | None = None  # sequential: ours
    engine: Any = None  # splice: LeRobot's engine
    interp: Any = None

    @property
    def artifact(self) -> str | None:
        return self.client.descriptor.get("artifact_identity")

    @property
    def present(self) -> bool:
        """The server's presence token is still alive (LeRobot's liveliness subscription)."""
        return bool(self.client.present)

    def close(self) -> None:
        if self.seq is not None:
            self.seq.close()  # a reply still in flight belongs to this session
        if self.engine is not None:
            self.engine.stop()
        else:
            self.client.close()


def execution_of(descriptor: dict[str, Any], asked: str) -> str:
    """The execution a session runs: the caller's word, else the server's, else what its contract implies."""
    if asked != "auto":
        return asked
    said = descriptor.get("execution")
    if isinstance(said, str) and said in ("sequential", "splice"):
        return said
    caps = descriptor.get("capabilities") or {}
    contracts = descriptor.get("execution_contracts") or ()
    rtc = int(caps.get("rtc_horizon") or 0) > 0 or "rtc_model_space_v1" in contracts
    return "splice" if rtc else "sequential"


def changed(old: Session, new: Session) -> str | None:
    """What a reopened session changed against the one it replaces, or None when it is the same deployment."""
    a, b = old.contract, new.contract
    if a.features != b.features or a.action_feature != b.action_feature:
        return "its features"
    if a.semantics != b.semantics:
        return f"its semantics ({a.semantics!r} -> {b.semantics!r})"
    if not math.isclose(old.interval, new.interval, rel_tol=1e-6):
        return f"its rate ({1 / old.interval:g} Hz -> {1 / new.interval:g} Hz)"
    if old.execution != new.execution:
        return f"its execution ({old.execution!r} -> {new.execution!r})"
    return None


def open_session(
    policy: Any,
    robot: Any,
    *,
    check: Callable[[Contract], None] | None = None,
    like: Session | None = None,
) -> Session:
    """Connect to ``policy``'s deployment, read its contract against ``robot`` and admit one session.

    ``check`` runs against the contract before admission (the robot's requirements, on attach). ``like``: the
    session this one replaces; the server must serve the same checkpoint and contract, else ``PolicyFault``.
    """
    pin = {} if like is None or like.artifact is None else {"expected_artifact": like.artifact}
    try:
        client = policy._connect(**pin)
    except Exception as e:
        if like is not None and getattr(e, "code", None) == "incompatible":  # LeRobot's expected_artifact check
            raise PolicyFault(f"{policy!r}: the server came back with a different deployment: {e}") from e
        raise
    try:
        caps, sem = client.capabilities, client.descriptor["semantics"]
        contract = Contract(robot.description, tuple(caps.features), caps.action_feature, sem)
        execution = execution_of(client.descriptor, policy.asked)
        out = Session(client, contract, execution, float(caps.action_interval))
        if like is not None and (what := changed(like, out)) is not None:
            raise PolicyFault(f"{policy!r}: the server came back with a different deployment: {what} changed")
        contract.check()
        contract.units()  # unknown semantics -> refused by name, before a session exists
        if check is not None:
            check(contract)
        if execution == "sequential":  # whole chunks, appended: an aligned session would demand RTC cursors
            from lerobot.inference import chunk_settings

            config = policy._config(sem, client.descriptor, mode="chunk", chunk_merge="append", **pin)
            client.chunk_settings = chunk_settings(  # the client derived them from the describe-time config
                config.chunk_merge, config.blend_steps, config.blend_weight, config.blend_components
            )
        else:
            config = policy._config(sem, client.descriptor, **pin)  # the hello's limits widen the action timeout
        client.config = config
        client.admit(
            features=caps.features,
            action_feature=caps.action_feature,
            semantics=sem,
            action_interval=caps.action_interval,
            mode=config.mode,
        )
    except BaseException as e:
        client.close()
        if like is not None and getattr(e, "code", None) == "incompatible":  # the server's own admission check
            raise PolicyFault(f"{policy!r}: the server came back with a different deployment: {e}") from e
        raise
    if execution == "sequential":  # no engine: ``act`` asks for a chunk only when the queue is empty
        out.seq = Sequential(client, contract, repr(policy), policy.calls)
        return out
    from lerobot.remote_inference import RemoteInferenceEngine
    from lerobot.utils.action_interpolator import ActionInterpolator

    client.infer = ledgered(client.infer, contract, policy.calls)  # LeRobot's engine asks; the ledger is ours
    out.engine = RemoteInferenceEngine(client, config, task="")
    out.interp = ActionInterpolator(multiplier=1)
    out.engine.start()
    out.engine.resume()
    return out
