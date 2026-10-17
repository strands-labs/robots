"""X.509 identity from the record (MAPPING 3.1–3.4, 3.6, 3.7) on moto; the policy text has no literal tenant/thing."""

from __future__ import annotations

import json

from strands_robots.mesh.iot import provision

ACCOUNT, REGION = "123456789012", "us-west-2"  # placeholders, as in conftest

RECORD = {"id": "arm-1", "device_type": "robot_arm", "tenant": "lab", "labels": {"role": "arm"}, "manifest": {}}


def test_fleet_policy_fits_the_cap_and_names_no_tenant_or_thing():
    text = provision.policy_text(provision.device_policy(REGION, ACCOUNT))
    assert len(text) <= provision.POLICY_CAP, len(text)
    assert "${iot:Connection.Thing.ThingName}" in text and "${iot:Connection.Thing.Attributes[tenant]}" in text
    assert "lab" not in text and "arm" not in text
    doc = json.loads(text)
    actions = {a for s in doc["Statement"] for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])}
    assert actions == {"iot:Connect", "iot:Publish", "iot:RetainPublish", "iot:Subscribe", "iot:Receive"}
    pub = [s for s in doc["Statement"] if "iot:Publish" in s["Action"]][0]["Resource"]
    assert not any("@estop" in r for r in pub), "a device never publishes the tenant e-stop"
    assert any(r.endswith("@presence/${iot:Connection.Thing.ThingName}") for r in pub)


def test_operator_policy_has_no_publish_on_the_devices_word():
    text = provision.policy_text(provision.operator_policy(REGION, ACCOUNT, "lab"))
    doc = json.loads(text)
    pub = [s for s in doc["Statement"] if s["Action"] == "iot:Publish"][0]["Resource"]
    assert any(r.endswith("strands/lab/@estop") for r in pub)
    for word in ("@presence", "/manifest", "/lockout", "/state"):
        assert not any(r.endswith(word) for r in pub), word
    assert any(s["Action"] == "iot:SendDirectMessage" for s in doc["Statement"])


def test_csr_carries_the_device_id_as_cn():
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    key_pem, csr_pem = provision.csr("arm-1")
    req = x509.load_pem_x509_csr(csr_pem.encode())
    assert req.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "arm-1"
    assert key_pem.startswith("-----BEGIN RSA PRIVATE KEY-----")


def test_register_rotate_retire_revoke_teardown(aws, tmp_path):
    provision.setup_tenant(aws, "lab", region=REGION, account=ACCOUNT)
    provision.setup_tenant(aws, "lab", region=REGION, account=ACCOUNT)  # idempotent
    out = provision.register(aws, RECORD, tmp_path)
    assert out["thing"] == "arm-1" and out["thing_type"] == "robot_arm" and out["thing_group"] == "strands-lab"
    assert out["query"] == f"client=arm-1&cert={tmp_path}/arm-1/cert.pem&key={tmp_path}/arm-1/key.pem"
    assert oct((tmp_path / "arm-1" / "key.pem").stat().st_mode)[-3:] == "600"
    thing = aws.describe_thing(thingName="arm-1")
    assert thing["attributes"] == {"tenant": "lab", "device_type": "robot_arm", "role": "arm"}
    assert aws.list_things_in_thing_group(thingGroupName="strands-lab")["things"] == ["arm-1"]  # names
    first = out["certificate_id"]
    assert aws.describe_certificate(certificateId=first)["certificateDescription"]["status"] == "ACTIVE"
    policies = aws.list_attached_policies(
        target=aws.describe_certificate(certificateId=first)["certificateDescription"]["certificateArn"]
    )
    assert [p["policyName"] for p in policies["policies"]] == ["strands-robots-device"]

    rot = provision.rotate(aws, "arm-1", tmp_path)
    assert rot["retire"] == [first] and rot["certificate_id"] != first
    assert len(aws.list_thing_principals(thingName="arm-1")["principals"]) == 2  # both ACTIVE until retire
    assert provision.retire(aws, "arm-1", [first]) == [first]
    assert len(aws.list_thing_principals(thingName="arm-1")["principals"]) == 1

    assert provision.revoke(aws, "arm-1") == [rot["certificate_id"]]
    assert (
        aws.describe_certificate(certificateId=rot["certificate_id"])["certificateDescription"]["status"] == "INACTIVE"
    )
    assert aws.list_thing_principals(thingName="arm-1")["principals"] == []

    provision.teardown(aws, "arm-1")
    assert aws.list_things()["things"] == []
    assert aws.list_things_in_thing_group(thingGroupName="strands-lab")["things"] == []
