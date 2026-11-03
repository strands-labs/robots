"""Fleet provisioning by claim (MAPPING row 3.5): a fresh robot gets its own certificate on first mount.

The device generates a key and a CSR with ``CN=<device_id>`` (``provision.csr``), connects with the fleet's CLAIM
certificate, asks IoT for a certificate from that CSR and registers its thing through ``TEMPLATE``. The template
creates thing, type, group membership and attributes and attaches the fleet policy. The pre-provisioning hook
(``HOOK_SOURCE``, a Lambda the user deploys) is a first-come binding of a NEW name: it accepts a request only when
the thing does not exist yet and the certificate's subject CN equals that thing name, so a claim holder can name a
new robot but never mint a certificate for one that already exists (measured: forged CN → 403 in 552 ms, honest
path claim connect → connected as the thing in ~5 s). Which serials may claim which names is a product choice
the hook does not make. The hook's role is the user's (it needs ``iot:DescribeThing``); the library creates no
IAM. Files land in ``<out_dir>/<device_id>/{cert.pem,key.pem}`` (0600).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Collection
from pathlib import Path
from typing import Any

from .. import wire
from .policy import POLICY_NAME, group_name
from .provision import _write, csr

TEMPLATE_NAME = "strands-robots-claim"
TEMPLATE: dict[str, Any] = {
    "Parameters": {
        "device_id": {"Type": "String"},
        "device_type": {"Type": "String"},
        "tenant": {"Type": "String"},
        "AWS::IoT::Certificate::Id": {"Type": "String"},
    },
    "Resources": {
        "thing": {
            "Type": "AWS::IoT::Thing",
            "OverrideSettings": {"AttributePayload": "MERGE", "ThingGroups": "DO_NOTHING", "ThingTypeName": "REPLACE"},
            "Properties": {
                "ThingName": {"Ref": "device_id"},
                "ThingTypeName": {"Ref": "device_type"},
                "AttributePayload": {"tenant": {"Ref": "tenant"}, "device_type": {"Ref": "device_type"}},
                "ThingGroups": [{"Fn::Join": ["", ["strands-", {"Ref": "tenant"}]]}],
            },
        },
        "certificate": {
            "Type": "AWS::IoT::Certificate",
            "Properties": {"CertificateId": {"Ref": "AWS::IoT::Certificate::Id"}, "Status": "Active"},
        },
        "policy": {"Type": "AWS::IoT::Policy", "Properties": {"PolicyName": POLICY_NAME}},
    },
}
"""The provisioning template body (≤ 10 KB): the thing IS the device id, the group IS the tenant."""

HOOK_SOURCE = '''\
"""Pre-provisioning hook (python3.12 Lambda, boto3 from the runtime): allow a RegisterThing only for a thing that does
not exist yet, and only when the certificate's subject has exactly one CN and it is that thing's name. The CN is read
by walking the DER to the TBS subject Name (never the issuer, never bytes inside another attribute's value). The
hook's role needs iot:DescribeThing; any other answer than ResourceNotFoundException denies."""
import base64


def _tlv(der, i):
    tag, n = der[i], der[i + 1]
    i += 2
    if n & 0x80:
        k = n & 0x7F
        n, i = int.from_bytes(der[i : i + k], "big"), i + k
    return tag, i, i + n  # tag, start of value, end of value


def _items(der, start, end):
    while start < end:
        tag, v, start = _tlv(der, start)
        yield tag, v, start


def subject_cns(pem):
    der = base64.b64decode("".join(line for line in pem.splitlines() if "-----" not in line))
    _, c, _ = _tlv(der, 0)  # Certificate
    _, t, t_end = _tlv(der, c)  # tbsCertificate
    fields = [f for f in _items(der, t, t_end) if f[0] != 0xA0]  # drop [0] version
    _, s, s_end = fields[4]  # serial, signature, issuer, validity, SUBJECT
    cns = []
    for _, r, r_end in _items(der, s, s_end):  # RDN (SET)
        for _, a, a_end in _items(der, r, r_end):  # AttributeTypeAndValue
            (_, o, o_end), (tag, v, v_end) = list(_items(der, a, a_end))[:2]
            if der[o:o_end] == bytes.fromhex("550403"):  # 2.5.4.3 commonName
                cns.append(der[v:v_end].decode("utf-8", "replace") if tag in (0x0C, 0x13) else None)
    return cns


def thing_exists(name):
    import boto3

    iot = boto3.client("iot")
    try:
        iot.describe_thing(thingName=name)
        return True
    except iot.exceptions.ResourceNotFoundException:
        return False


def handler(event, _context):
    params = event.get("parameters") or {}
    thing = str(params.get("device_id") or "")
    try:
        ok = bool(thing) and subject_cns(event.get("certificatePem", "")) == [thing] and not thing_exists(thing)
    except Exception:  # a malformed certificate or any AWS error denies
        ok = False
    return {"allowProvisioning": ok, "parameterOverrides": {"device_id": thing} if ok else {}}
'''
"""Source of the pre-provisioning Lambda: a new thing, named by the certificate's one subject CN."""


def hook_decision(parameters: dict[str, Any], certificate_pem: str, *, existing: Collection[str] = ()) -> bool:
    """The hook's rule, callable without AWS: accept iff ``parameters["device_id"]`` is not in ``existing`` (the
    things IoT already has) and is the certificate's one subject CN."""
    ns: dict[str, Any] = {}
    exec(HOOK_SOURCE, ns)  # noqa: S102 — the shipped source is the rule under test
    ns["thing_exists"] = lambda name: name in existing
    return bool(ns["handler"]({"parameters": parameters, "certificatePem": certificate_pem}, None)["allowProvisioning"])


def template_body() -> str:
    text = json.dumps(TEMPLATE, separators=(",", ":"))
    if len(text) > 10 * 1024:
        raise ValueError("provisioning template over 10 KB")
    return text


def create_template(iot: Any, provisioning_role_arn: str, *, hook_arn: str | None = None) -> str:
    """Create or update ``TEMPLATE_NAME`` (idempotent). ``provisioning_role_arn`` is the user's IoT provisioning
    role; ``hook_arn`` the deployed ``HOOK_SOURCE`` Lambda (strongly recommended: without it, CN is unchecked)."""
    from botocore.exceptions import ClientError

    kw: dict[str, Any] = dict(templateBody=template_body(), enabled=True, provisioningRoleArn=provisioning_role_arn)
    if hook_arn:
        kw["preProvisioningHook"] = {"targetArn": hook_arn, "payloadVersion": "2020-04-01"}
    try:
        iot.create_provisioning_template(
            templateName=TEMPLATE_NAME, description="strands-robots: a robot's identity on first mount", **kw
        )
        return "created"
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceAlreadyExistsException":
            raise
        iot.create_provisioning_template_version(
            templateName=TEMPLATE_NAME, templateBody=kw["templateBody"], setAsDefault=True
        )
        return "updated"


def claim(
    endpoint: str,
    claim_cert: str,
    claim_key: str,
    record: dict[str, Any],
    out_dir: str | Path,
    *,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Run the claim flow for ``record`` (id, device_type, tenant) against ``endpoint`` with the claim certificate.
    Returns the thing name, certificate id and the ``iot://`` query (``client=&cert=&key=``) for ``Mesh``."""
    from awscrt import mqtt5
    from awsiot import iotidentity, mqtt5_client_builder

    did = wire.check_id(str(record["id"]), "device_id")
    key_pem, csr_pem = csr(did)
    up, done = threading.Event(), threading.Event()
    result: dict[str, Any] = {}
    client = mqtt5_client_builder.mtls_from_path(
        endpoint=endpoint,
        cert_filepath=claim_cert,
        pri_key_filepath=claim_key,
        client_id=f"strands-claim-{did}",
        on_lifecycle_connection_success=lambda *_: up.set(),
    )
    client.start()
    if not up.wait(timeout):
        raise TimeoutError(f"claim: did not connect to {endpoint} within {timeout}s")
    ident = iotidentity.IotIdentityClient(client)
    qos = mqtt5.QoS.AT_LEAST_ONCE

    def rejected(resp: Any) -> None:
        result["error"] = f"{getattr(resp, 'error_code', '?')}: {getattr(resp, 'error_message', resp)}"
        done.set()

    def on_cert(resp: Any) -> None:
        result.update(
            certificate_id=resp.certificate_id,
            certificate_pem=resp.certificate_pem,
            token=resp.certificate_ownership_token,
        )
        req = iotidentity.RegisterThingRequest(
            template_name=TEMPLATE_NAME,
            certificate_ownership_token=resp.certificate_ownership_token,
            parameters={"device_id": did, "device_type": str(record["device_type"]), "tenant": str(record["tenant"])},
        )
        ident.publish_register_thing(req, qos)

    def on_thing(resp: Any) -> None:
        result["thing"] = resp.thing_name
        done.set()

    sub = iotidentity.CreateCertificateFromCsrSubscriptionRequest()
    ident.subscribe_to_create_certificate_from_csr_accepted(sub, qos, on_cert)[0].result(timeout)
    ident.subscribe_to_create_certificate_from_csr_rejected(sub, qos, rejected)[0].result(timeout)
    reg = iotidentity.RegisterThingSubscriptionRequest(template_name=TEMPLATE_NAME)
    ident.subscribe_to_register_thing_accepted(reg, qos, on_thing)[0].result(timeout)
    ident.subscribe_to_register_thing_rejected(reg, qos, rejected)[0].result(timeout)
    ident.publish_create_certificate_from_csr(
        iotidentity.CreateCertificateFromCsrRequest(certificate_signing_request=csr_pem), qos
    ).result(timeout)
    ok = done.wait(timeout)
    client.stop()
    if not ok:
        raise TimeoutError(f"claim: no answer for {did} within {timeout}s")
    if "error" in result:
        raise PermissionError(f"claim for {did} refused: {result['error']}")
    paths = _write(Path(out_dir) / did, result["certificate_pem"], key_pem)
    query = f"client={did}&cert={paths['cert']}&key={paths['key']}"
    group = group_name(str(record["tenant"]))
    return {
        "thing": result["thing"],
        "certificate_id": result["certificate_id"],
        "thing_group": group,
        **paths,
        "query": query,
    }
