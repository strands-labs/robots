"""The rules engine for one tenant (MAPPING rows 2.1–2.6): the mesh's events as topics with SQL rules.

Rules are data (``rules_for``), applied idempotently (``apply_rules``: create, or replace when the document
changed). The library never creates IAM: the caller passes the ``role_arn`` the actions assume, and that role must
be older than the rule (measured: a rule whose role was created seconds before drops messages silently until IAM
propagates). Presence and the live ``state`` stream never go through rules by default (3,600–108,000 triggers per
robot per hour); state storage is opt-in through Basic Ingest at ``telemetry_hz`` ≤ 1, and because a Basic Ingest
message has no MQTT topic, the tenant and device id travel in the envelope itself (they already do: ``wire``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from .. import wire

MAX_TELEMETRY_HZ = 1.0
_NAME = re.compile(r"[^A-Za-z0-9_]")


def rule_name(kind: str, tenant: str) -> str:
    """``strands_<kind>_<tenant>``: a rule name allows only ``[A-Za-z0-9_]``; other id characters become ``_``."""
    return f"strands_{kind}_{_NAME.sub('_', wire.check_id(tenant, 'tenant'))}"[:128]


def basic_ingest_topic(tenant: str, device_id: str) -> str:
    """Where a device publishes its ≤ 1 Hz telemetry: no messaging charge and no subscriber ever sees it."""
    return f"$aws/rules/{rule_name('telemetry', tenant)}/{wire.key('state', tenant, device_id)}"


def _republish(topic: str, role_arn: str, qos: int = 1) -> dict[str, Any]:
    return {"republish": {"topic": topic, "roleArn": role_arn, "qos": qos}}


def rules_for(
    tenant: str,
    role_arn: str,
    *,
    sns_topic_arn: str | None = None,
    s3_bucket: str | None = None,
    timestream: tuple[str, str] | None = None,
    telemetry: bool = False,
) -> dict[str, dict[str, Any]]:
    """The tenant's rules as ``{name: topicRulePayload}``.

    ``estop``: every tenant-wide e-stop republished on ``…/@estop/latest`` (and to SNS when given). ``events``:
    ``fault``/``estop``/``run.*`` events into Timestream ``(database, table)`` and/or S3 ``recordings/<tenant>/…``.
    ``fault``: high-severity faults to SNS. ``telemetry``: the Basic Ingest sink for ``basic_ingest_topic``.
    """
    t = wire.check_id(tenant, "tenant")
    out: dict[str, dict[str, Any]] = {}
    estop_actions = [_republish(f"{wire.key('estop', t)}/latest", role_arn)]
    if sns_topic_arn:
        estop_actions.append({"sns": {"targetArn": sns_topic_arn, "roleArn": role_arn, "messageFormat": "JSON"}})
    out[rule_name("estop", t)] = {
        "sql": f"SELECT *, topic(2) AS tenant, timestamp() AS at FROM '{wire.key('estop', t)}'",
        "description": f"strands-robots tenant {t}: e-stop fan-out",
        "actions": estop_actions,
        "ruleDisabled": False,
        "awsIotSqlVersion": "2016-03-23",
    }
    sinks: list[dict[str, Any]] = []
    if timestream:
        db, table = timestream
        sinks.append({"timestream": {"roleArn": role_arn, "databaseName": db, "tableName": table,
                                     "dimensions": [{"name": "tenant", "value": "${tenant}"},
                                                    {"name": "device_id", "value": "${device_id}"}]}})  # fmt: skip
    if s3_bucket:
        sinks.append({"s3": {"roleArn": role_arn, "bucketName": s3_bucket,
                             "key": "recordings/${tenant}/${device_id}/${event}/${timestamp()}.json"}})  # fmt: skip
    if sinks:
        out[rule_name("events", t)] = {
            "sql": f"SELECT *, topic(3) AS device_id, topic(5) AS event, timestamp() AS at "
            f"FROM '{wire.key('event', t, '*', '*').replace('*', '+')}'",
            "description": f"strands-robots tenant {t}: device events to storage",
            "actions": sinks,
            "ruleDisabled": False,
            "awsIotSqlVersion": "2016-03-23",
        }
    if sns_topic_arn:
        out[rule_name("fault", t)] = {
            "sql": f"SELECT *, topic(3) AS device_id FROM '{wire.key('event', t, '*', 'fault').replace('*', '+')}' "
            f"WHERE result.severity = 'high'",
            "description": f"strands-robots tenant {t}: high-severity faults",
            "actions": [{"sns": {"targetArn": sns_topic_arn, "roleArn": role_arn, "messageFormat": "JSON"}}],
            "ruleDisabled": False,
            "awsIotSqlVersion": "2016-03-23",
        }
    if telemetry:
        if not sinks:
            raise ValueError("telemetry=True needs a sink: s3_bucket= and/or timestream=(database, table)")
        out[rule_name("telemetry", t)] = {
            "sql": "SELECT tenant, device_id, seq, t, result FROM 'strands/basic-ingest'",  # ignored: Basic Ingest
            "description": f"strands-robots tenant {t}: <= 1 Hz state telemetry (Basic Ingest, no messaging charge)",
            "actions": [dict(s) for s in sinks],
            "ruleDisabled": False,
            "awsIotSqlVersion": "2016-03-23",
        }
    return out


def apply_rules(iot: Any, rules: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Create each rule, or replace it when its document differs. ``iot`` is a boto3 ``iot`` client. Returns
    ``{name: "created" | "replaced" | "unchanged"}``."""
    from botocore.exceptions import ClientError

    out: dict[str, str] = {}
    for name, payload in rules.items():
        try:
            current = iot.get_topic_rule(ruleName=name)["rule"]
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in ("ResourceNotFoundException", "UnauthorizedException"):
                raise
            iot.create_topic_rule(ruleName=name, topicRulePayload=dict(payload))
            out[name] = "created"
            continue
        same = current.get("sql") == payload["sql"] and _norm(current.get("actions")) == _norm(payload["actions"])
        if same and not current.get("ruleDisabled", False):
            out[name] = "unchanged"
        else:
            iot.replace_topic_rule(ruleName=name, topicRulePayload=dict(payload))
            out[name] = "replaced"
    return out


def delete_rules(iot: Any, tenant: str) -> list[str]:
    """Delete every ``strands_*_<tenant>`` rule; the names deleted."""
    suffix = rule_name("x", tenant)[len("strands_x") :]
    names = [r["ruleName"] for r in iot.list_topic_rules().get("rules", []) if r["ruleName"].endswith(suffix)]
    for name in names:
        iot.delete_topic_rule(ruleName=name)
    return sorted(names)


def _norm(actions: Any) -> str:
    return json.dumps(actions or [], sort_keys=True, separators=(",", ":"))


def telemetry(link: Any, tenant: str, device_id: str, robot: Any, hz: float) -> Callable[[], None]:
    """Publish ``robot``'s state envelope at ``hz`` ≤ 1 on the Basic Ingest topic. Returns the undo."""
    if not 0 < hz <= MAX_TELEMETRY_HZ:
        raise ValueError(f"telemetry_hz={hz} is over {MAX_TELEMETRY_HZ:g} Hz; the mesh state stream is the live path")
    topic = basic_ingest_topic(tenant, device_id)
    seq = [0]

    def send(payload: Mapping[str, Any]) -> None:
        seq[0] += 1
        link.publish(topic, wire.encode(wire.envelope(tenant, device_id, "state", "state", seq=seq[0], result=payload)))

    undo: Callable[[], None] = robot.subscribe("state", send, hz=hz)
    return undo
