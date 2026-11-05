"""The IoT/IAM policy documents (MAPPING 3.2): one fleet policy written with policy variables so every device
certificate carries the same text, a small controller add-on, and the operator IAM policy. The documents are data;
``provision`` writes them and ``claim`` names them. ``json`` is the only import."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .. import wire

POLICY_NAME = "strands-robots-device"
"""The one policy every device certificate carries; the tenant and thing come from the connection, not the text."""
POLICY_CAP = 2048
"""IoT refuses a policy document over 2,048 bytes."""
IAM_POLICY_CAP = 6144
"""IAM refuses a managed policy over 6,144 characters; each device adds ten ARNs to the operator policy, so one
document names about six short device ids and a larger tenant attaches several."""
_THING, _TENANT = "${iot:Connection.Thing.ThingName}", "${iot:Connection.Thing.Attributes[tenant]}"
_RULE = "${iot:Connection.Thing.Attributes[telemetry_rule]}"
"""The thing's own Basic Ingest rule (``rules.rule_name("telemetry", tenant)``), written by ``provision``/``claim``."""


def group_name(tenant: str) -> str:
    return f"strands-{wire.check_id(tenant, 'tenant')}"


def operator_client_prefix(tenant: str) -> str:
    """``strands-op-<tenant>-``: every operator MQTT client id of ``tenant`` starts with it (the IAM link makes one)."""
    return f"strands-op-{wire.check_id(tenant, 'tenant')}-"


def policy_text(doc: Mapping[str, Any]) -> str:
    """The document as it is sent: compact JSON (IoT counts every byte of the text against the cap)."""
    return json.dumps(doc, separators=(",", ":"))


def device_policy(region: str, account: str) -> dict[str, Any]:
    """The fleet policy: connect only as the attached thing; publish/subscribe only inside its own tenant and only
    on its own device keys, presence, reply and shadows; ``@estop`` and ``@peer`` are read-only for a device (a
    commander's liveliness is its own: ``controller_policy``, or the operator IAM policy). Telemetry goes only to the
    rule its thing names (the ``telemetry_rule`` attribute), never to another tenant's rule."""
    arn = f"arn:aws:iot:{region}:{account}"
    own, tenant, shadow = f"strands/{_TENANT}/{_THING}", f"strands/{_TENANT}", f"$aws/things/{_THING}/shadow"
    attached = {"Bool": {"iot:Connection.Thing.IsAttached": "true"}}
    keep = [f"{arn}:topic/{own}/*", f"{arn}:topic/{shadow}/*"]
    pub = [f"{arn}:topic/{tenant}/@presence/{_THING}", f"{arn}:topic/{tenant}/@reply/*",
           f"{arn}:topic/$aws/rules/{_RULE}/{own}/state"]  # fmt: skip
    sub = [f"{arn}:topicfilter/{own}/*", f"{arn}:topicfilter/{tenant}/@estop", f"{arn}:topicfilter/{tenant}/@peer/*",
           f"{arn}:topicfilter/{tenant}/@presence/*", f"{arn}:topicfilter/{shadow}/*"]  # fmt: skip
    recv = [f"{arn}:topic/{tenant}/*", f"{arn}:topic/{shadow}/*"]
    doc = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "iot:Connect", "Resource": f"{arn}:client/{_THING}", "Condition": attached},
        # RetainPublish only on its own keys (manifest, lockout) and shadow; presence, replies and telemetry are
        # never retained, and a device has no right to publish a command or a liveliness token anywhere.
        {"Effect": "Allow", "Action": ["iot:Publish", "iot:RetainPublish"], "Resource": keep},
        {"Effect": "Allow", "Action": "iot:Publish", "Resource": pub},
        {"Effect": "Allow", "Action": "iot:Subscribe", "Resource": sub},
        # Receive is coarse on purpose: Subscribe is the gate, and every byte here counts against the 2,048 cap.
        {"Effect": "Allow", "Action": "iot:Receive", "Resource": recv},
    ]}  # fmt: skip
    if len(policy_text(doc)) > POLICY_CAP:
        raise ValueError(f"device policy is {len(policy_text(doc))} bytes; IoT accepts at most {POLICY_CAP}")
    return doc


def controller_policy(region: str, account: str) -> dict[str, Any]:
    """Attached IN ADDITION to the fleet policy on a certificate that also *commands* (a Greengrass core, a
    controller box): it may send rpcs and fetch manifests inside its tenant and hear its own replies. A plain robot
    never gets it; a certificate with it is still pinned to its own thing by the fleet policy. (Two small policies
    instead of one: a single document would be 2,249 bytes, over IoT's 2,048 cap.) A commander also beats its own
    ``@peer`` liveliness, which the devices it moves watch."""
    arn, t = f"arn:aws:iot:{region}:{account}", f"strands/{_TENANT}"
    pub = [f"{arn}:topic/{t}/*/rpc/*", f"{arn}:topic/{t}/*/manifest", f"{arn}:topic/{t}/@peer/*"]
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "iot:Publish", "Resource": pub},
        {"Effect": "Allow", "Action": "iot:Subscribe", "Resource": f"{arn}:topicfilter/{t}/@reply/{_THING}"},
    ]}  # fmt: skip


def operator_policy(
    region: str, account: str, tenant: str, *, things: list[str] | None = None, all_things: bool = False
) -> dict[str, Any]:
    """An IAM policy for operators of ``tenant`` (SigV4 WebSocket): rpcs, e-stop, shadow targets, reads; no
    publish on presence, manifest, lockout or state (those are the device's word). The operator connects only with a
    client id under ``operator_client_prefix(tenant)``, so it can neither take over another tenant's operator id nor
    a device's; it direct-messages only the tenant's devices, and only on their rpc topics, and hears only their
    lifecycle events. The shadow grants name the
    tenant's own device ids (``things``, e.g. the ids ``registry.list_things`` yields): a shadow document carries
    no author and a thing name has no tenant segment, so an account-wide shadow grant would let an operator of one
    tenant move another tenant's robot through ``update_thing_shadow``. An empty ``things`` is refused rather than
    widened to ``*``; ``all_things=True`` keeps the account-wide ``$aws/things/*`` shadow grant, which crosses the
    tenant boundary and is only safe in a single-tenant account. A named shadow's IAM resource is
    ``thing/<id>/<shadow>``: operators update ``control`` only and read every shadow (``thing/<id>/*``)."""
    arn, t = f"arn:aws:iot:{region}:{account}", f"strands/{wire.check_id(tenant, 'tenant')}"
    ids = [wire.check_id(i, "device_id") for i in (things or [])]
    if not ids and not all_things:
        raise ValueError(
            "operator_policy needs the tenant's device ids (things=[...] from registry.list_things); "
            "pass all_things=True only in a single-tenant account, where it crosses no tenant boundary"
        )
    marks = ["*"] if all_things else ids
    sh_pub = [f"{arn}:topic/$aws/things/{m}/shadow/name/control/update" for m in marks]
    sh_pub += [f"{arn}:topic/$aws/things/{m}/shadow/name/*/get" for m in marks]
    sh_sub = [f"{arn}:topicfilter/$aws/things/{m}/shadow/name/*/*" for m in marks]
    sh_recv = [f"{arn}:topic/$aws/things/{m}/shadow/*" for m in marks]
    # lifecycle events of this tenant's devices only: the event topic ends in the client id (= the thing name)
    life_sub = [f"{arn}:topicfilter/$aws/events/presence/*/{m}" for m in marks]
    life_recv = [f"{arn}:topic/$aws/events/presence/*/{m}" for m in marks]
    direct = [f"{arn}:client/{m}" for m in marks]  # a device's MQTT client id is its thing name
    thing_arns: Any = "*" if all_things else [f"{arn}:thing/{m}" for m in marks]
    read_arns: Any = "*" if all_things else [f"{arn}:thing/{m}/*" for m in marks]
    write_arns: Any = "*" if all_things else [f"{arn}:thing/{m}/control" for m in marks]
    group = f"{arn}:thinggroup/{group_name(tenant)}"
    pub = [f"{arn}:topic/{t}/*/rpc/*", f"{arn}:topic/{t}/@estop", f"{arn}:topic/{t}/@peer/*", *sh_pub]
    sub = [f"{arn}:topicfilter/{t}/*", f"{arn}:topicfilter/{t}/*/*", f"{arn}:topicfilter/{t}/*/*/*",
           *sh_sub, *life_sub]  # fmt: skip
    recv = [f"{arn}:topic/{t}/*", *sh_recv, *life_recv]
    # SendDirectMessage names the TARGET client as its resource (a connection/* grant matches nothing). The topic
    # condition keeps it to the two keys Transport.query sends on: a direct message needs no subscription, so a wider
    # topic would let an operator forge a device's word.
    queried = {"StringLike": {"iot:Topic": [f"{t}/*/rpc/*", f"{t}/*/manifest"]}}
    doc = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "iot:Connect", "Resource": f"{arn}:client/{operator_client_prefix(tenant)}*"},
        {"Effect": "Allow", "Action": "iot:SendDirectMessage", "Resource": direct, "Condition": queried},
        {"Effect": "Allow", "Action": "iot:Publish", "Resource": pub},
        {"Effect": "Allow", "Action": "iot:RetainPublish", "Resource": f"{arn}:topic/{t}/@estop"},
        {"Effect": "Allow", "Action": "iot:Subscribe", "Resource": sub},
        {"Effect": "Allow", "Action": "iot:Receive", "Resource": recv},
        {"Effect": "Allow", "Action": "iot:DescribeEndpoint", "Resource": "*"},
        {"Effect": "Allow", "Action": "iot:ListThingsInThingGroup", "Resource": group},
        {"Effect": "Allow", "Action": "iot:DescribeThing", "Resource": thing_arns},
        {"Effect": "Allow", "Action": "iot:GetThingShadow", "Resource": read_arns},
        {"Effect": "Allow", "Action": "iot:UpdateThingShadow", "Resource": write_arns},
    ]}  # fmt: skip
    if len(policy_text(doc)) > IAM_POLICY_CAP:
        raise ValueError(
            f"operator policy for {len(ids)} things is {len(policy_text(doc))} characters; IAM accepts at most "
            f"{IAM_POLICY_CAP}, about six devices: attach several documents to the operator"
        )
    return doc
