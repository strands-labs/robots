"""X.509 identity for devices (MAPPING rows 3.1–3.4, 3.6, 3.7): thing = device id, type = device type, group and
attribute = tenant, ONE fleet policy written with policy variables, one certificate per device from a LOCAL CSR.
The certificate's CN is the device id (``csr``): on MQTT the broker knows the thing through
``${iot:Connection.Thing.ThingName}`` (proven: SUBACK/PUBACK 135 outside the thing's own scope, link stays up); on
the HTTPS path that variable is empty, so ``${iot:Certificate.Subject.CommonName}`` is the identity that still
holds (v0.5 fact (a)); AWS-generated certificates all say ``CN=AWS IoT Certificate``, so the key never leaves the
device and only the CSR travels. ``boto3`` and ``cryptography`` are imported where used."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import wire

POLICY_NAME = "strands-robots-device"
"""The one policy every device certificate carries; the tenant and thing come from the connection, not the text."""
POLICY_CAP = 2048
"""IoT refuses a policy document over 2,048 bytes."""
_THING, _TENANT = "${iot:Connection.Thing.ThingName}", "${iot:Connection.Thing.Attributes[tenant]}"


def group_name(tenant: str) -> str:
    return f"strands-{wire.check_id(tenant, 'tenant')}"


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


def policy_text(doc: Mapping[str, Any]) -> str:
    """The document as it is sent: compact JSON (IoT counts every byte of the text against the cap)."""
    return json.dumps(doc, separators=(",", ":"))


def operator_policy(region: str, account: str, tenant: str) -> dict[str, Any]:
    """An IAM policy for operators of ``tenant`` (SigV4 WebSocket): rpcs, e-stop, shadow targets, reads; no
    publish on presence, manifest, lockout or state (those are the device's word)."""
    arn, t = f"arn:aws:iot:{region}:{account}", f"strands/{wire.check_id(tenant, 'tenant')}"
    shadow = "$aws/things/*/shadow"
    pub = [f"{arn}:topic/{t}/*/rpc/*", f"{arn}:topic/{t}/@estop", f"{arn}:topic/{t}/@peer/*",
           f"{arn}:topic/{shadow}/name/control/update", f"{arn}:topic/{shadow}/name/*/get"]  # fmt: skip
    sub = [f"{arn}:topicfilter/{t}/*", f"{arn}:topicfilter/{t}/*/*", f"{arn}:topicfilter/{t}/*/*/*",
           f"{arn}:topicfilter/{shadow}/name/*/*", f"{arn}:topicfilter/$aws/events/presence/*/*"]  # fmt: skip
    recv = [f"{arn}:topic/{t}/*", f"{arn}:topic/{shadow}/*", f"{arn}:topic/$aws/events/presence/*"]
    api = ["iot:DescribeEndpoint", "iot:ListThingsInThingGroup", "iot:DescribeThing", "iot:GetThingShadow",
           "iot:UpdateThingShadow"]  # fmt: skip
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "iot:Connect", "Resource": f"{arn}:client/strands-op-*"},
        {"Effect": "Allow", "Action": "iot:SendDirectMessage", "Resource": f"{arn}:connection/*"},
        {"Effect": "Allow", "Action": "iot:Publish", "Resource": pub},
        {"Effect": "Allow", "Action": "iot:RetainPublish", "Resource": f"{arn}:topic/{t}/@estop"},
        {"Effect": "Allow", "Action": "iot:Subscribe", "Resource": sub},
        {"Effect": "Allow", "Action": "iot:Receive", "Resource": recv},
        {"Effect": "Allow", "Action": api, "Resource": "*"},
    ]}  # fmt: skip


def csr(device_id: str) -> tuple[str, str]:
    """A fresh RSA-2048 key and a CSR with ``CN=<device_id>``, both PEM. The key is made here and never sent.
    RSA in the traditional PEM form is what every awscrt TLS backend loads (measured: an EC/PKCS#8 key is refused
    on macOS with AWS_IO_FILE_VALIDATION_FAILURE)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    wire.check_id(device_id, "device_id")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, device_id)])
    req = x509.CertificateSigningRequestBuilder().subject_name(name).sign(key, hashes.SHA256())
    key_pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
    ).decode()
    return key_pem, req.public_bytes(serialization.Encoding.PEM).decode()


def _exists(call: Any, **kw: Any) -> bool:
    from botocore.exceptions import ClientError

    try:
        call(**kw)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ResourceAlreadyExistsException":
            return False
        raise


def setup_tenant(
    iot: Any,
    tenant: str,
    *,
    region: str,
    account: str,
    policy: str = POLICY_NAME,
    tags: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Once per tenant: the thing group, the fleet policy and its ``<policy>-controller`` add-on (idempotent).
    ``iot`` is a boto3 ``iot`` client."""
    tag = [{"Key": k, "Value": v} for k, v in (tags or {}).items()]
    _exists(iot.create_thing_group, thingGroupName=group_name(tenant), tags=tag)
    docs = ((policy, device_policy(region, account)), (f"{policy}-controller", controller_policy(region, account)))
    for name, doc in docs:
        _exists(iot.create_policy, policyName=name, policyDocument=policy_text(doc), tags=tag)
    return {"thing_group": group_name(tenant), "policy": policy, "controller_policy": f"{policy}-controller"}


def teardown_tenant(iot: Any, tenant: str, *, policy: str = POLICY_NAME) -> None:
    """Delete the tenant's thing group and both fleet policies (after every device of the tenant is torn down)."""
    for name in (policy, f"{policy}-controller"):
        for v in iot.list_policy_versions(policyName=name).get("policyVersions", []):
            if not v["isDefaultVersion"]:
                iot.delete_policy_version(policyName=name, policyVersionId=v["versionId"])
        iot.delete_policy(policyName=name)
    iot.delete_thing_group(thingGroupName=group_name(tenant))


def register(
    iot: Any, record: Mapping[str, Any], out_dir: str | Path, *, policy: str = POLICY_NAME, controller: bool = False
) -> dict[str, Any]:
    """A device's identity from its mesh ``record`` (id, device_type, tenant, labels): thing type, thing with
    attributes, group membership, a certificate from a local CSR with the fleet policy (plus the controller policy
    when ``controller``). Writes ``<out_dir>/<device_id>/{cert.pem,key.pem}`` (0600) and returns ids plus the
    ``iot://`` query to use."""
    did, dtype, tenant = str(record["id"]), str(record["device_type"]), str(record["tenant"])
    attrs = {
        "tenant": tenant,
        "device_type": dtype,
        **{str(k): str(v)[:800] for k, v in (record.get("labels") or {}).items()},
    }
    _exists(iot.create_thing_type, thingTypeName=_segment(dtype))
    _exists(iot.create_thing, thingName=did, thingTypeName=_segment(dtype), attributePayload={"attributes": attrs})
    iot.update_thing(
        thingName=did, thingTypeName=_segment(dtype), attributePayload={"attributes": attrs, "merge": True}
    )
    iot.add_thing_to_thing_group(thingGroupName=group_name(tenant), thingName=did)
    key_pem, csr_pem = csr(did)
    cert = iot.create_certificate_from_csr(certificateSigningRequest=csr_pem, setAsActive=True)
    iot.attach_thing_principal(thingName=did, principal=cert["certificateArn"])  # first: never an orphan cert
    for name in (policy, f"{policy}-controller") if controller else (policy,):
        iot.attach_policy(policyName=name, target=cert["certificateArn"])
    paths = _write(Path(out_dir) / did, cert["certificatePem"], key_pem)
    query = f"client={did}&cert={paths['cert']}&key={paths['key']}"
    return {"thing": did, "thing_type": _segment(dtype), "thing_group": group_name(tenant),
            "certificate_id": cert["certificateId"], **paths, "query": query}  # fmt: skip


def rotate(iot: Any, device_id: str, out_dir: str | Path, *, policy: str = POLICY_NAME) -> dict[str, Any]:
    """A new certificate for ``device_id`` (same CN, same policy) over the old files; the previous certificates
    stay ACTIVE until ``retire``, so the device reconnects first and nothing breaks."""
    old = [p.rsplit("/", 1)[-1] for p in _principals(iot, device_id)]
    key_pem, csr_pem = csr(device_id)
    cert = iot.create_certificate_from_csr(certificateSigningRequest=csr_pem, setAsActive=True)
    iot.attach_policy(policyName=policy, target=cert["certificateArn"])
    iot.attach_thing_principal(thingName=device_id, principal=cert["certificateArn"])
    paths = _write(Path(out_dir) / device_id, cert["certificatePem"], key_pem)
    return {"thing": device_id, "certificate_id": cert["certificateId"], "retire": old, **paths}


def retire(iot: Any, device_id: str, certificate_ids: list[str]) -> list[str]:
    """INACTIVE, detach, delete the given certificates of ``device_id`` (the old ones after a rotation)."""
    done: list[str] = []
    for arn in _principals(iot, device_id):
        cid = arn.rsplit("/", 1)[-1]
        if cid in certificate_ids:
            _drop_cert(iot, device_id, arn, cid)
            done.append(cid)
    return done


def revoke(iot: Any, device_id: str) -> list[str]:
    """Every certificate of ``device_id`` → INACTIVE and detached: its next connect fails, nothing else changes."""
    arns = _principals(iot, device_id)
    for arn in arns:
        _drop_cert(iot, device_id, arn, arn.rsplit("/", 1)[-1], delete=False)
    return [a.rsplit("/", 1)[-1] for a in arns]


def teardown(iot: Any, device_id: str) -> dict[str, Any]:
    """Delete the certificates and the thing (type and group stay: other devices use them)."""
    for arn in _principals(iot, device_id):
        _drop_cert(iot, device_id, arn, arn.rsplit("/", 1)[-1])
    for g in iot.list_thing_groups_for_thing(thingName=device_id).get("thingGroups", []):
        iot.remove_thing_from_thing_group(thingGroupName=g["groupName"], thingName=device_id)
    iot.delete_thing(thingName=device_id)
    return {"thing": device_id, "deleted": True}


def _principals(iot: Any, device_id: str) -> list[str]:
    return list(iot.list_thing_principals(thingName=device_id).get("principals", []))


def _drop_cert(iot: Any, device_id: str, arn: str, cid: str, *, delete: bool = True) -> None:
    iot.update_certificate(certificateId=cid, newStatus="INACTIVE")
    iot.detach_thing_principal(thingName=device_id, principal=arn)
    for p in iot.list_attached_policies(target=arn).get("policies", []):
        iot.detach_policy(policyName=p["policyName"], target=arn)
    if delete:
        iot.delete_certificate(certificateId=cid, forceDelete=True)


def _segment(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-." else "-" for ch in text)[:128] or "device"


def _write(folder: Path, cert_pem: str, key_pem: str) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    cert, key = folder / "cert.pem", folder / "key.pem"
    cert.write_text(cert_pem)
    key.write_text(key_pem)
    key.chmod(0o600)
    return {"cert": str(cert), "key": str(key)}
