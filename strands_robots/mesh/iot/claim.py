"""Fleet provisioning by claim (MAPPING row 3.5): a fresh robot gets its own certificate on first mount.

The device generates a key and a CSR with ``CN=<device_id>`` (``provision.csr``), connects with the fleet's CLAIM
certificate, asks IoT for a certificate from that CSR and registers its thing through ``TEMPLATE``. The template
creates thing, type, group membership and attributes and attaches the fleet policy. The pre-provisioning hook
(``HOOK_SOURCE``, a Lambda the user deploys) accepts a request only when the certificate's CN equals the thing name
it is about to create: a claim holder can never mint a certificate that speaks for another robot (v0.5 fact (b);
measured: forged CN → 403 in 552 ms, honest path claim connect → connected as the thing in ~5 s). The hook's role is
the user's; the library creates no IAM. Files land in ``<out_dir>/<device_id>/{cert.pem,key.pem}`` (0600).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .. import wire
from .provision import POLICY_NAME, _write, csr, group_name

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
"""Pre-provisioning hook (python3.12 Lambda, no dependencies): allow a RegisterThing only when the certificate's CN
is the thing being created. The hook event carries the certificate PEM, so the CN is read from the DER itself
(measured: a forged CSR with another robot's CN is denied 403 in 552 ms and its certificate is never activated)."""
import base64


def common_name(pem):
    der = base64.b64decode("".join(line for line in pem.splitlines() if "-----" not in line))
    i = der.find(bytes.fromhex("0603550403"))  # OID 2.5.4.3 commonName
    if i < 0:
        return None
    j = i + 5
    tag, length = der[j], der[j + 1]
    return der[j + 2 : j + 2 + length].decode("utf-8", "replace") if tag in (0x0C, 0x13) else None


def handler(event, _context):
    params = event.get("parameters") or {}
    thing = str(params.get("device_id") or "")
    ok = bool(thing) and common_name(event.get("certificatePem", "")) == thing
    return {"allowProvisioning": ok, "parameterOverrides": {"device_id": thing} if ok else {}}
'''
"""Source of the pre-provisioning Lambda. The template only runs for a NEW thing (``OverrideSettings`` never
replaces an existing thing's certificate attachment), and the hook binds CN to that thing."""


def hook_decision(parameters: dict[str, Any], certificate_pem: str) -> bool:
    """The hook's rule, callable without AWS: accept iff the certificate's CN equals ``parameters["device_id"]``."""
    ns: dict[str, Any] = {}
    exec(HOOK_SOURCE, ns)  # noqa: S102 — the shipped source is the rule under test
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
