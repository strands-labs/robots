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
"""IAM refuses a managed policy over 6,144 characters; each device adds six ARNs to the operator policy, so one
document names about nine devices and a larger tenant attaches several."""
_THING, _TENANT = "${iot:Connection.Thing.ThingName}", "${iot:Connection.Thing.Attributes[tenant]}"


def group_name(tenant: str) -> str:
    return f"strands-{wire.check_id(tenant, 'tenant')}"


def policy_text(doc: Mapping[str, Any]) -> str:
    """The document as it is sent: compact JSON (IoT counts every byte of the text against the cap)."""
    return json.dumps(doc, separators=(",", ":"))


def device_policy(region: str, account: str) -> dict[str, Any]:
    """The fleet policy: connect only as the attached thing; publish/subscribe only inside its own tenant and only
    on its own device keys, presence, reply and shadows; ``@estop`` and ``@peer`` are read-only for a device."""
    arn = f"arn:aws:iot:{region}:{account}"
    own, tenant, shadow = f"strands/{_TENANT}/{_THING}", f"strands/{_TENANT}", f"$aws/things/{_THING}/shadow"
    attached = {"Bool": {"iot:Connection.Thing.IsAttached": "true"}}
    pub = [f"{arn}:topic/{own}/*", f"{arn}:topic/{tenant}/@presence/{_THING}", f"{arn}:topic/{tenant}/@reply/*",
           f"{arn}:topic/{tenant}/@peer/*", f"{arn}:topic/{shadow}/*",
           f"{arn}:topic/$aws/rules/*/{own}/state"]  # fmt: skip
    sub = [f"{arn}:topicfilter/{own}/*", f"{arn}:topicfilter/{tenant}/@estop", f"{arn}:topicfilter/{tenant}/@peer/*",
           f"{arn}:topicfilter/{tenant}/@presence/*", f"{arn}:topicfilter/{shadow}/*"]  # fmt: skip
    recv = [f"{arn}:topic/{tenant}/*", f"{arn}:topic/{shadow}/*"]
    doc = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "iot:Connect", "Resource": f"{arn}:client/{_THING}", "Condition": attached},
        # RetainPublish rides with Publish: a device may retain only on its own keys (manifest, lockout), and it
        # has no right to publish a command anywhere, so a retained command from a device is impossible.
        {"Effect": "Allow", "Action": ["iot:Publish", "iot:RetainPublish"], "Resource": pub},
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
    instead of one: a single document would be 2,249 bytes, over IoT's 2,048 cap.)"""
    arn, t = f"arn:aws:iot:{region}:{account}", f"strands/{_TENANT}"
    pub = [f"{arn}:topic/{t}/*/rpc/*", f"{arn}:topic/{t}/*/manifest"]
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "iot:Publish", "Resource": pub},
        {"Effect": "Allow", "Action": "iot:Subscribe", "Resource": f"{arn}:topicfilter/{t}/@reply/{_THING}"},
    ]}  # fmt: skip


def operator_policy(
    region: str, account: str, tenant: str, *, things: list[str] | None = None, all_things: bool = False
) -> dict[str, Any]:
    """An IAM policy for operators of ``tenant`` (SigV4 WebSocket): rpcs, e-stop, shadow targets, reads; no
    publish on presence, manifest, lockout or state (those are the device's word). The shadow grants name the
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
    thing_arns: Any = "*" if all_things else [f"{arn}:thing/{m}" for m in marks]
    read_arns: Any = "*" if all_things else [f"{arn}:thing/{m}/*" for m in marks]
    write_arns: Any = "*" if all_things else [f"{arn}:thing/{m}/control" for m in marks]
    group = f"{arn}:thinggroup/{group_name(tenant)}"
    pub = [f"{arn}:topic/{t}/*/rpc/*", f"{arn}:topic/{t}/@estop", f"{arn}:topic/{t}/@peer/*", *sh_pub]
    sub = [f"{arn}:topicfilter/{t}/*", f"{arn}:topicfilter/{t}/*/*", f"{arn}:topicfilter/{t}/*/*/*",
           *sh_sub, f"{arn}:topicfilter/$aws/events/presence/*/*"]  # fmt: skip
    recv = [f"{arn}:topic/{t}/*", *sh_recv, f"{arn}:topic/$aws/events/presence/*"]
    doc = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "iot:Connect", "Resource": f"{arn}:client/strands-op-*"},
        {"Effect": "Allow", "Action": "iot:SendDirectMessage", "Resource": f"{arn}:connection/*"},
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
            f"{IAM_POLICY_CAP}, about nine devices: attach several documents to the operator"
        )
    return doc
