"""X.509 identity for devices (MAPPING rows 3.1–3.4, 3.6, 3.7): thing = device id, type = device type, group and
attribute = tenant, ONE fleet policy written with policy variables, one certificate per device from a LOCAL CSR.
The certificate's CN is the device id (``csr``): on MQTT the broker knows the thing through
``${iot:Connection.Thing.ThingName}`` (proven: SUBACK/PUBACK 135 outside the thing's own scope, link stays up); on
the HTTPS path that variable is empty, so ``${iot:Certificate.Subject.CommonName}`` is the identity that still
holds (v0.5 fact (a)); AWS-generated certificates all say ``CN=AWS IoT Certificate``, so the key never leaves the
device and only the CSR travels. ``boto3`` and ``cryptography`` are imported where used."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .. import wire
from ..transports.pkcs11 import refuse_pin
from .policy import POLICY_NAME, controller_policy, device_policy, group_name, policy_text


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
    iot: Any,
    record: Mapping[str, Any],
    out_dir: str | Path,
    *,
    policy: str = POLICY_NAME,
    controller: bool = False,
    csr_pem: str | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """A device's identity from its mesh ``record`` (id, device_type, tenant, labels): thing type, thing with
    attributes, group membership, a certificate from a CSR with the fleet policy (plus the controller policy when
    ``controller``). By default the CSR is made locally and the key written to ``<out_dir>/<device_id>/key.pem``
    (0600). Give ``csr_pem`` (a CSR whose key was generated in an HSM) and its ``key`` (a ``pkcs11:`` URI) to
    register an identity whose private key is never a PEM: only ``cert.pem`` is written and the query's ``key=``
    is the token URI. Returns ids plus the ``iot://`` query to use."""
    _pair(csr_pem, key)
    did, dtype, tenant = str(record["id"]), str(record["device_type"]), str(record["tenant"])
    labels = {str(k): str(v)[:800] for k, v in (record.get("labels") or {}).items()}
    attrs, ttype = {"tenant": tenant, "device_type": dtype, **labels}, _segment(dtype)
    _exists(iot.create_thing_type, thingTypeName=ttype)
    _exists(iot.create_thing, thingName=did, thingTypeName=ttype, attributePayload={"attributes": attrs})
    iot.update_thing(thingName=did, thingTypeName=ttype, attributePayload={"attributes": attrs, "merge": True})
    iot.add_thing_to_thing_group(thingGroupName=group_name(tenant), thingName=did)
    policies = [policy, f"{policy}-controller"] if controller else [policy]
    cert, paths = _issue(iot, did, out_dir, policies, csr_pem=csr_pem, key=key)
    query = f"client={did}&cert={paths['cert']}&key={quote(paths['key'], safe='') if key else paths['key']}"
    return {"thing": did, "thing_type": ttype, "thing_group": group_name(tenant),
            "certificate_id": cert["certificateId"], **paths, "query": query}  # fmt: skip


def rotate(
    iot: Any,
    device_id: str,
    out_dir: str | Path,
    *,
    policy: str = POLICY_NAME,
    csr_pem: str | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """A new certificate for ``device_id`` with the same CN and every policy its current ones carry (``policy`` if
    none) over the old files; those stay ACTIVE until ``retire``, so the device reconnects first and nothing breaks.
    An identity registered with ``csr_pem``/``key`` rotates the same way (a CSR signed by its token key and the
    unchanged URI); without them it is refused, so a software key never replaces the token's."""
    _pair(csr_pem, key)
    folder = Path(out_dir) / device_id
    if csr_pem is None and (folder / "cert.pem").exists() and not (folder / "key.pem").exists():
        raise ValueError(f"{device_id}'s key lives in a token: give rotate csr_pem (signed by it) and key (its URI)")
    arns = _principals(iot, device_id)
    names = {p["policyName"] for a in arns for p in iot.list_attached_policies(target=a).get("policies", [])}
    cert, paths = _issue(iot, device_id, out_dir, sorted(names or {policy}), csr_pem=csr_pem, key=key)
    old = [a.rsplit("/", 1)[-1] for a in arns]
    return {"thing": device_id, "certificate_id": cert["certificateId"], "retire": old, **paths}


def _pair(csr_pem: str | None, key: str | None) -> None:
    if (csr_pem is None) != (key is None):
        raise ValueError("give both csr_pem (an HSM-generated CSR) and key (its pkcs11: URI), or neither")
    refuse_pin(key or "")


def _issue(
    iot: Any,
    device_id: str,
    out_dir: str | Path,
    policies: list[str],
    *,
    csr_pem: str | None = None,
    key: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """A certificate from a CSR (made locally, or ``csr_pem`` when the key is an HSM-born ``key``), attached to
    the thing first (never an orphan), then to ``policies``."""
    key_pem: str | None
    if csr_pem is None:
        key_pem, csr_pem = csr(device_id)
    else:
        key_pem = None
    cert = iot.create_certificate_from_csr(certificateSigningRequest=csr_pem, setAsActive=True)
    iot.attach_thing_principal(thingName=device_id, principal=cert["certificateArn"])
    for name in policies:
        iot.attach_policy(policyName=name, target=cert["certificateArn"])
    return cert, _write(Path(out_dir) / device_id, cert["certificatePem"], key_pem, key)


def retire(iot: Any, device_id: str, certificate_ids: list[str]) -> list[str]:
    """INACTIVE, detach, delete the given certificates of ``device_id`` (the old ones after a rotation)."""
    done: list[str] = []
    for arn in _principals(iot, device_id):
        cid = arn.rsplit("/", 1)[-1]
        if cid in certificate_ids:
            _drop_cert(iot, device_id, arn, cid)
            done.append(cid)
    return done


def revoke(iot: Any, device_id: str, *, mesh: Any = None, deadline: float = 1.0) -> dict[str, Any]:
    """Certificates of ``device_id`` → INACTIVE, detached: its next connect fails, but a live session stays up until
    it drops, so ``mesh`` (an operator's ``Mesh``) e-stops it first; ``stopped`` is its ack within ``deadline``."""
    stopped = mesh is not None and device_id in mesh.broadcast("estop", selector=device_id, deadline=deadline)
    arns = _principals(iot, device_id)
    for arn in arns:
        _drop_cert(iot, device_id, arn, arn.rsplit("/", 1)[-1], delete=False)
    return {"thing": device_id, "stopped": stopped, "certs": [a.rsplit("/", 1)[-1] for a in arns]}


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


def _write(folder: Path, cert_pem: str, key_pem: str | None, key_uri: str | None = None) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    cert = folder / "cert.pem"
    cert.write_text(cert_pem)
    if key_pem is None:  # HSM-born key: nothing private is ever a PEM on disk
        assert key_uri is not None  # register requires key with csr_pem
        return {"cert": str(cert), "key": key_uri}
    key = folder / "key.pem"
    key.write_text(key_pem)
    key.chmod(0o600)
    return {"cert": str(cert), "key": str(key)}
