"""X.509 identity from the record (MAPPING 3.1–3.4, 3.6, 3.7) on moto; the policy text has no literal tenant/thing."""

from __future__ import annotations

import pytest

from strands_robots.mesh.iot import provision

ACCOUNT, REGION = "123456789012", "us-west-2"  # placeholders, as in conftest

RECORD = {"id": "arm-1", "device_type": "robot_arm", "tenant": "lab", "labels": {"role": "arm"}, "manifest": {}}


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

    assert provision.revoke(aws, "arm-1") == {"thing": "arm-1", "stopped": False, "certs": [rot["certificate_id"]]}
    assert (
        aws.describe_certificate(certificateId=rot["certificate_id"])["certificateDescription"]["status"] == "INACTIVE"
    )
    assert aws.list_thing_principals(thingName="arm-1")["principals"] == []

    provision.teardown(aws, "arm-1")
    assert aws.list_things()["things"] == []
    assert aws.list_things_in_thing_group(thingGroupName="strands-lab")["things"] == []


def _policies(aws, cert_id):
    arn = aws.describe_certificate(certificateId=cert_id)["certificateDescription"]["certificateArn"]
    return sorted(p["policyName"] for p in aws.list_attached_policies(target=arn)["policies"])


def test_a_rotated_controller_certificate_is_still_a_controller(aws, tmp_path):
    provision.setup_tenant(aws, "lab", region=REGION, account=ACCOUNT)
    old = provision.register(aws, RECORD, tmp_path, controller=True)["certificate_id"]
    both = ["strands-robots-device", "strands-robots-device-controller"]
    assert _policies(aws, old) == both
    rot = provision.rotate(aws, "arm-1", tmp_path)
    provision.retire(aws, "arm-1", rot["retire"])
    assert _policies(aws, rot["certificate_id"]) == both


def test_revoke_with_a_mesh_estops_the_connected_device_before_its_certificate_goes(aws, tmp_path):
    from strands_robots import Robot
    from strands_robots.mesh import Mesh

    provision.setup_tenant(aws, "lab", region=REGION, account=ACCOUNT)
    provision.register(aws, RECORD, tmp_path)
    url = f"iot://fake?broker=revoke-{tmp_path.name}"
    device, operator = Mesh(url, tenant="lab"), Mesh(url, tenant="lab")
    arm = Robot("so101", id="arm-1")
    try:
        device.mount(arm, device_id="arm-1")
        assert operator.discover("arm-1", min_count=1, timeout=3.0)
        out = provision.revoke(aws, "arm-1", mesh=operator)
        assert out["stopped"] is True and arm.estopped, "a connected session outlives revoke(); the e-stop stops it"
        assert len(out["certs"]) == 1 and aws.list_thing_principals(thingName="arm-1")["principals"] == []
    finally:
        operator.close()
        device.close()
        arm.close()


def test_register_with_an_hsm_csr_writes_no_key_pem(aws, tmp_path):
    from urllib.parse import parse_qs

    provision.setup_tenant(aws, "lab", region=REGION, account=ACCOUNT)
    _, csr_pem = provision.csr("arm-9")  # stands in for a CSR whose key was generated in an HSM
    rec = {**RECORD, "id": "arm-9"}
    key_uri = "pkcs11:token=lab;object=arm-9?module-path=/opt/hsm.so"
    out = provision.register(aws, rec, tmp_path, csr_pem=csr_pem, key=key_uri)
    assert not (tmp_path / "arm-9" / "key.pem").exists()  # nothing private is a PEM on disk
    assert (tmp_path / "arm-9" / "cert.pem").exists()
    assert out["key"] == key_uri
    assert parse_qs(out["query"])["key"][0] == key_uri  # url-decoded back to the token URI
    assert aws.describe_certificate(certificateId=out["certificate_id"])["certificateDescription"]["status"] == "ACTIVE"


def test_register_refuses_a_key_uri_carrying_its_pin_before_issuing(aws, tmp_path):
    provision.setup_tenant(aws, "lab", region=REGION, account=ACCOUNT)
    key_uri = "pkcs11:token=lab;object=arm-9?module-path=/opt/hsm.so&pin-value=5678"
    with pytest.raises(ValueError, match="PKCS11_PIN"):
        provision.register(aws, {**RECORD, "id": "arm-9"}, tmp_path, csr_pem=provision.csr("arm-9")[1], key=key_uri)
    assert not (tmp_path / "arm-9").exists() and aws.list_certificates()["certificates"] == []


def test_register_needs_both_an_hsm_csr_and_its_key_uri(aws, tmp_path):
    with pytest.raises(ValueError, match="both"):
        provision.register(aws, RECORD, tmp_path, csr_pem="a-csr")
    with pytest.raises(ValueError, match="both"):
        provision.register(aws, RECORD, tmp_path, key="pkcs11:token=lab;object=arm-1")


def test_rotate_refuses_an_hsm_identity_without_a_new_csr(aws, tmp_path):
    provision.setup_tenant(aws, "lab", region=REGION, account=ACCOUNT)
    _, csr_pem = provision.csr("arm-9")
    key_uri = "pkcs11:token=lab;object=arm-9?module-path=/opt/hsm.so"
    out = provision.register(aws, {**RECORD, "id": "arm-9"}, tmp_path, csr_pem=csr_pem, key=key_uri)
    cert_before = (tmp_path / "arm-9" / "cert.pem").read_text()
    with pytest.raises(ValueError, match="csr_pem"):
        provision.rotate(aws, "arm-9", tmp_path)
    assert sorted(p.name for p in (tmp_path / "arm-9").iterdir()) == ["cert.pem"]  # no software key written
    assert (tmp_path / "arm-9" / "cert.pem").read_text() == cert_before  # the link still matches the token key
    assert [a.rsplit("/", 1)[-1] for a in aws.list_thing_principals(thingName="arm-9")["principals"]] == [
        out["certificate_id"]
    ]  # nothing issued


def test_rotate_an_hsm_identity_with_a_csr_signed_by_its_token_key(aws, tmp_path):
    provision.setup_tenant(aws, "lab", region=REGION, account=ACCOUNT)
    key_uri = "pkcs11:token=lab;object=arm-9?module-path=/opt/hsm.so"
    out = provision.register(aws, {**RECORD, "id": "arm-9"}, tmp_path, csr_pem=provision.csr("arm-9")[1], key=key_uri)
    rot = provision.rotate(aws, "arm-9", tmp_path, csr_pem=provision.csr("arm-9")[1], key=key_uri)
    assert sorted(p.name for p in (tmp_path / "arm-9").iterdir()) == ["cert.pem"]
    assert rot["key"] == key_uri and rot["retire"] == [out["certificate_id"]]
    with pytest.raises(ValueError, match="both"):
        provision.rotate(aws, "arm-9", tmp_path, csr_pem="a-csr")
