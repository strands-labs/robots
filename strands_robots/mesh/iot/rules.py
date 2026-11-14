"""The rules engine for one tenant (MAPPING rows 2.1–2.6): the mesh's events as topics with SQL rules.

Rules are data (``rules_for``), applied idempotently (``apply_rules``: create, or replace when the document
changed). The library never creates IAM: the caller passes the ``role_arn`` the actions assume, and that role must
be older than the rule (measured: a rule whose role was created seconds before drops messages silently until IAM
propagates). Presence and the live ``state`` stream never go through rules by default (3,600–108,000 triggers per
robot per hour); state storage is opt-in through Basic Ingest at ``telemetry_hz`` ≤ 1.

What a rule stores is filed under the identity the broker checked, never one the message body names: the tenant is
the rule's own (each tenant has its rules), the device is the key segment the device policy pins to the publishing
thing (``topic(3)``), or for Basic Ingest the MQTT client id, which that policy pins to the thing name. A body that
says ``"tenant": "other"`` still lands under its own tenant.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from .. import wire

MAX_TELEMETRY_HZ = 1.0
KINDS = ("estop", "events", "fault", "telemetry", "lifecycle")
"""Every kind of rule a tenant can have: ``rules_for`` gives the first four, ``lifecycle.clear_presence_rule`` the
last."""
_ROOM = 128 - len("strands__") - max(len(kind) for kind in KINDS)
"""110: what a rule name, at most 128 characters, leaves for the tenant after the longest kind."""
_NAME = re.compile(r"[^A-Za-z0-9_]")
_PLAIN = re.compile(r"[A-Za-z0-9]+")


def rule_name(kind: str, tenant: str) -> str:
    """``strands_<kind>_<tenant>``, within what a rule name allows: ``[A-Za-z0-9_]``, at most 128 characters. A tenant
    of letters and digits, at most 110 of them, is used as it is. Any other tenant keeps at most its first 97
    characters, with ``_`` for ``- . :``, then ``_`` and the first 12 hex digits of its SHA-256. So ``lab-a`` and
    ``lab_a``, or two long tenants that start alike, get rules of their own: that tenant part has a ``_``, which a
    plain one never has, and two of them are equal only if the two SHA-256 digests begin with the same 48 bits."""
    if kind not in KINDS:
        raise ValueError(f"unknown rule kind {kind!r}; known: {list(KINDS)}")
    return f"strands_{kind}_{tenant_key(tenant, _ROOM)}"


def tenant_key(tenant: str, room: int) -> str:
    """``tenant`` in at most ``room`` characters of ``[A-Za-z0-9_]``: as it is if it is letters and digits, else its
    head with ``_`` for ``- . :``, ``_`` and 12 hex of its SHA-256. It never holds a ``-``, so a name built as
    ``<prefix><key>-...`` ends the tenant at the first ``-``, and no tenant's key is a prefix of another's."""
    t = wire.check_id(tenant, "tenant")
    if len(t) <= room and _PLAIN.fullmatch(t):
        return t
    head = _NAME.sub("_", t)[: room - 13]  # room for "_" and the 12 hex digits
    return f"{head}_{hashlib.sha256(t.encode()).hexdigest()[:12]}"


def basic_ingest_topic(tenant: str, device_id: str) -> str:
    """Where a device publishes its ≤ 1 Hz telemetry: no messaging charge and no subscriber ever sees it."""
    return f"$aws/rules/{rule_name('telemetry', tenant)}/{wire.key('state', tenant, device_id)}"


def _sns(topic_arn: str, role_arn: str) -> dict[str, Any]:
    """An SNS action that sends the rule's record as the message body. ``RAW``, not ``JSON``: with ``JSON`` SNS reads
    the body as a per-protocol envelope and refuses one with no ``"default"`` key, which every record is, and a rule
    with no error action drops that refusal silently (measured: 0 messages in 90 s after a real e-stop)."""
    return {"sns": {"targetArn": topic_arn, "roleArn": role_arn, "messageFormat": "RAW"}}


def _republish(topic: str, role_arn: str, qos: int = 1) -> dict[str, Any]:
    return {"republish": {"topic": topic, "roleArn": role_arn, "qos": qos}}


def _error_action(role_arn: str, tenant: str, kind: str, s3_bucket: str | None) -> dict[str, Any]:
    """The ``errorAction`` for one rule. A rule with none drops an action's failure silently (measured: 0 SNS
    messages in 90 s after a real e-stop), which for the e-stop and fault rules is the wrong default. Into S3 under
    an ``errors/<tenant>/<kind>/`` prefix when a bucket is given, else republished to the tenant's ``@error`` topic
    so the failure is visible to anyone the role lets subscribe. The caller's ``error_action=`` overrides this."""
    if s3_bucket:
        return {"s3": {"roleArn": role_arn, "bucketName": s3_bucket,
                       "key": f"errors/{tenant}/{kind}/${{timestamp()}}.json"}}  # fmt: skip
    return _republish(wire.key("error", tenant), role_arn)


def rules_for(
    tenant: str,
    role_arn: str,
    *,
    sns_topic_arn: str | None = None,
    s3_bucket: str | None = None,
    timestream: tuple[str, str] | None = None,
    telemetry: bool = False,
    error_action: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """The tenant's rules as ``{name: topicRulePayload}``.

    ``estop``: every tenant-wide e-stop republished on ``…/@estop/latest`` (and to SNS when given). ``events``:
    ``fault``/``estop``/``run.*`` events into Timestream ``(database, table)`` and/or S3 ``recordings/<tenant>/…``.
    ``fault``: high-severity faults to SNS. ``telemetry``: the Basic Ingest sink for ``basic_ingest_topic``.

    Every rule carries an ``errorAction`` so a failing action is recorded, not dropped: ``error_action`` when
    given, else into S3 under ``errors/<tenant>/<kind>/`` when ``s3_bucket`` is set, else republished to the
    tenant's ``@error`` topic.
    """
    t = wire.check_id(tenant, "tenant")
    out: dict[str, dict[str, Any]] = {}

    def err(kind: str) -> dict[str, Any]:
        return dict(error_action) if error_action is not None else _error_action(role_arn, t, kind, s3_bucket)

    estop_actions = [_republish(f"{wire.key('estop', t)}/latest", role_arn)]
    if sns_topic_arn:
        estop_actions.append(_sns(sns_topic_arn, role_arn))
    out[rule_name("estop", t)] = {
        "sql": f"SELECT *, topic(2) AS tenant, timestamp() AS at FROM '{wire.key('estop', t)}'",
        "description": f"strands-robots tenant {t}: e-stop fan-out",
        "actions": estop_actions,
        "errorAction": err("estop"),
        "ruleDisabled": False,
        "awsIotSqlVersion": "2016-03-23",
    }
    sinks = _sinks(role_arn, t, s3_bucket, timestream, device="${topic(3)}", event="${topic(5)}")
    if sinks:
        out[rule_name("events", t)] = {
            "sql": f"SELECT *, topic(2) AS tenant, topic(3) AS device_id, topic(5) AS event, timestamp() AS at "
            f"FROM '{wire.key('event', t, '*', '*').replace('*', '+')}'",
            "description": f"strands-robots tenant {t}: device events to storage",
            "actions": sinks,
            "errorAction": err("events"),
            "ruleDisabled": False,
            "awsIotSqlVersion": "2016-03-23",
        }
    if sns_topic_arn:
        out[rule_name("fault", t)] = {
            "sql": f"SELECT *, topic(2) AS tenant, topic(3) AS device_id "
            f"FROM '{wire.key('event', t, '*', 'fault').replace('*', '+')}' "
            f"WHERE result.severity = 'high'",
            "description": f"strands-robots tenant {t}: high-severity faults",
            "actions": [_sns(sns_topic_arn, role_arn)],
            "errorAction": err("fault"),
            "ruleDisabled": False,
            "awsIotSqlVersion": "2016-03-23",
        }
    if telemetry:
        if not sinks:
            raise ValueError("telemetry=True needs a sink: s3_bucket= and/or timestream=(database, table)")
        # Basic Ingest ignores FROM; only seq, t and result come from the body, the identity never does
        out[rule_name("telemetry", t)] = {
            "sql": f"SELECT seq, t, result, '{t}' AS tenant, clientid() AS device_id FROM 'strands/basic-ingest'",
            "description": f"strands-robots tenant {t}: <= 1 Hz state telemetry (Basic Ingest, no messaging charge)",
            "actions": _sinks(role_arn, t, s3_bucket, timestream, device="${clientid()}", event="state"),
            "errorAction": err("telemetry"),
            "ruleDisabled": False,
            "awsIotSqlVersion": "2016-03-23",
        }
    return out


def _sinks(
    role_arn: str, tenant: str, s3_bucket: str | None, timestream: tuple[str, str] | None, *, device: str, event: str
) -> list[dict[str, Any]]:
    """The storage actions, keyed by ``tenant`` itself and the ``device``/``event`` substitutions the broker fills from
    the topic or connection, so no body field can choose where a record is filed."""
    sinks: list[dict[str, Any]] = []
    if timestream:
        db, table = timestream
        sinks.append({"timestream": {"roleArn": role_arn, "databaseName": db, "tableName": table,
                                     "dimensions": [{"name": "tenant", "value": tenant},
                                                    {"name": "device_id", "value": device}]}})  # fmt: skip
    if s3_bucket:
        sinks.append({"s3": {"roleArn": role_arn, "bucketName": s3_bucket,
                             "key": f"recordings/{tenant}/{device}/{event}/${{timestamp()}}.json"}})  # fmt: skip
    return sinks


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
        same = (
            current.get("sql") == payload["sql"]
            and _norm(current.get("actions")) == _norm(payload["actions"])
            and _norm(current.get("errorAction")) == _norm(payload.get("errorAction"))
        )
        if same and not current.get("ruleDisabled", False):
            out[name] = "unchanged"
        else:
            iot.replace_topic_rule(ruleName=name, topicRulePayload=dict(payload))
            out[name] = "replaced"
    return out


def delete_rules(iot: Any, tenant: str) -> list[str]:
    """Delete THIS tenant's rules, by the names ``rule_name`` gives it (one per kind in ``KINDS``); the names
    deleted. Only those exact names match, so another tenant's rules never do: not ``prod-lab``'s next to ``lab``,
    nor ``lab_a``'s next to ``lab-a``, however long the tenants. The account's rules come a page at a time
    (``nextToken``); every page is read before the first delete."""
    own = {rule_name(kind, tenant) for kind in KINDS}
    found: set[str] = set()
    token: str | None = None
    while True:
        page = iot.list_topic_rules(**({"nextToken": token} if token else {}))
        found.update(r["ruleName"] for r in page.get("rules", []) if r["ruleName"] in own)
        token = page.get("nextToken")
        if not token:
            break
    names = sorted(found)
    for name in names:
        iot.delete_topic_rule(ruleName=name)
    return names


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
