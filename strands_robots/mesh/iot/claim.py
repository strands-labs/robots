"""Fleet provisioning by claim (MAPPING row 3.5): a fresh robot gets its own certificate on first mount.

The device generates a key and a CSR with ``CN=<device_id>`` (``provision.csr``), connects with the fleet's CLAIM
certificate, asks IoT for a certificate from that CSR and registers its thing through ``TEMPLATE``. The template
creates thing, type, group membership and attributes and attaches the fleet policy. The pre-provisioning hook
(``claim_hook.HOOK_SOURCE``, a Lambda the user deploys, required by ``create_template``) accepts only a NEW name the
operator approved with ``claim_hook.allow``, with that approval's one-time secret, tenant and device type, and the
certificate's subject CN equal to that name; the approval is used up. So a leaked claim certificate names no robot
and joins no tenant on its own (measured before the allowlist: forged CN → 403 in 552 ms, honest path claim connect
→ connected as the thing in ~5 s). Files land in ``<out_dir>/<device_id>/{cert.pem,key.pem}`` (0600).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .. import wire
from .claim_hook import HOOK_SOURCE, allow, hook_decision
from .policy import POLICY_NAME, group_name
from .provision import _write, csr

__all__ = ["HOOK_SOURCE", "TEMPLATE", "TEMPLATE_NAME", "allow", "claim", "create_template", "hook_decision"]

TEMPLATE_NAME = "strands-robots-claim"
TEMPLATE: dict[str, Any] = {
    "Parameters": {
        "device_id": {"Type": "String"},
        "device_type": {"Type": "String"},
        "tenant": {"Type": "String"},
        "claim_secret": {"Type": "String"},  # read by the hook only; the hook overrides tenant and device_type
        "AWS::IoT::Certificate::Id": {"Type": "String"},
    },
    "Resources": {
        "thing": {
            "Type": "AWS::IoT::Thing",
            "OverrideSettings": {"AttributePayload": "MERGE", "ThingGroups": "DO_NOTHING", "ThingTypeName": "REPLACE"},
            "Properties": {
                "ThingName": {"Ref": "device_id"},
                "ThingTypeName": {"Ref": "device_type"},
                "AttributePayload": {
                    "tenant": {"Ref": "tenant"},
                    "device_type": {"Ref": "device_type"},
                    # rules.rule_name("telemetry", tenant) for a plain (letters and digits) tenant; any other tenant
                    # gets a hashed rule name the template cannot compute, so its telemetry is refused until
                    # provision.register (or update-thing) writes the attribute
                    "telemetry_rule": {"Fn::Join": ["", ["strands_telemetry_", {"Ref": "tenant"}]]},
                },
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


def template_body() -> str:
    text = json.dumps(TEMPLATE, separators=(",", ":"))
    if len(text) > 10 * 1024:
        raise ValueError("provisioning template over 10 KB")
    return text


def create_template(iot: Any, provisioning_role_arn: str, *, hook_arn: str) -> str:
    """Create ``TEMPLATE_NAME``, or bring an existing one up to date (body, role, enabled AND the hook: a template
    created earlier without one gets it now). ``hook_arn`` is the deployed ``HOOK_SOURCE`` Lambda and is required:
    without it a claim certificate could register any name in any tenant."""
    from botocore.exceptions import ClientError

    if not hook_arn:
        raise ValueError("create_template needs hook_arn= (the deployed claim_hook.HOOK_SOURCE Lambda)")
    hook = {"targetArn": hook_arn, "payloadVersion": "2020-04-01"}
    kw: dict[str, Any] = dict(enabled=True, provisioningRoleArn=provisioning_role_arn, preProvisioningHook=hook)
    try:
        iot.create_provisioning_template(
            templateName=TEMPLATE_NAME,
            description="strands-robots: a robot's identity on first mount",
            templateBody=template_body(),
            **kw,
        )
        return "created"
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceAlreadyExistsException":
            raise
        iot.create_provisioning_template_version(
            templateName=TEMPLATE_NAME, templateBody=template_body(), setAsDefault=True
        )
        iot.update_provisioning_template(templateName=TEMPLATE_NAME, **kw)  # the only call that sets the hook later
        return "updated"


def claim(
    endpoint: str,
    claim_cert: str,
    claim_key: str,
    record: dict[str, Any],
    out_dir: str | Path,
    *,
    secret: str,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Run the claim flow for ``record`` (id, device_type, tenant) against ``endpoint`` with the claim certificate.
    ``secret`` is the one-time secret ``claim_hook.allow`` returned when the operator approved this robot.
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
            parameters={
                "device_id": did,
                "device_type": str(record["device_type"]),
                "tenant": str(record["tenant"]),
                "claim_secret": secret,
            },  # fmt: skip
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
