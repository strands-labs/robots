"""Claim provisioning as data (MAPPING 3.5) and the read-only registry (7.1)."""

from __future__ import annotations

import json

from strands_robots.mesh.iot import claim, provision, registry

ACCOUNT, REGION = "123456789012", "us-west-2"


def test_template_binds_thing_to_device_id_and_group_to_tenant_within_10_kb():
    body = json.loads(claim.template_body())
    assert len(claim.template_body()) <= 10 * 1024
    thing = body["Resources"]["thing"]["Properties"]
    assert thing["ThingName"] == {"Ref": "device_id"}
    assert thing["ThingGroups"] == [{"Fn::Join": ["", ["strands-", {"Ref": "tenant"}]]}]
    assert body["Resources"]["policy"]["Properties"]["PolicyName"] == provision.POLICY_NAME
    assert set(body["Parameters"]) == {"device_id", "device_type", "tenant", "AWS::IoT::Certificate::Id"}


def _cert_pem(cn: str) -> str:
    """A certificate with ``CN=<cn>`` in the PEM shape IoT hands the hook (self-signed here; IoT's CA there)."""
    import datetime as dt

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    now = dt.datetime.now(dt.timezone.utc)
    b = x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key()).serial_number(1)
    cert = b.not_valid_before(now).not_valid_after(now + dt.timedelta(days=1)).sign(key, hashes.SHA256())
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def test_hook_accepts_only_a_certificate_whose_cn_is_the_thing_being_created():
    compile(claim.HOOK_SOURCE, "hook.py", "exec")
    assert "boto3" not in claim.HOOK_SOURCE  # the hook needs no dependencies and no IAM beyond logging
    honest, stolen = _cert_pem("arm-7"), _cert_pem("arm-1")
    assert claim.hook_decision({"device_id": "arm-7"}, honest) is True
    assert claim.hook_decision({"device_id": "arm-7"}, stolen) is False, "CN must be the thing being created"
    assert claim.hook_decision({"device_id": "arm-1"}, honest) is False
    assert claim.hook_decision({}, honest) is False
    assert claim.hook_decision({"device_id": "arm-7"}, "") is False
    # the same CN parser accepts the CN a CSR from provision.csr carries (what IoT signs)
    from cryptography import x509

    _key, csr_pem = provision.csr("arm-7")
    assert x509.load_pem_x509_csr(csr_pem.encode()).subject.rfc4514_string() == "CN=arm-7"


def test_registry_lists_the_tenants_things_and_fails_with_a_word(aws, tmp_path, monkeypatch):
    provision.setup_tenant(aws, "lab", region=REGION, account=ACCOUNT)
    rec = {"id": "arm-1", "device_type": "robot_arm", "tenant": "lab", "labels": {}, "manifest": {}}
    provision.register(aws, rec, tmp_path)
    rows = registry.list_things("lab", region=REGION, iot=aws)
    assert [(r["id"], r["device_type"], r["status"]) for r in rows] == [("arm-1", "robot_arm", "ok")]
    assert rows[0]["connected"] in (True, None)  # moto's index says connected; no indexing → None
    rows = registry.list_things("nonsuch", region=REGION, iot=aws)
    assert rows[0]["status"] == "unavailable" and rows[0]["id"] == ""
    monkeypatch.setenv("STRANDS_IOT_REGISTRY", "off")
    assert registry.list_things("lab", region=REGION, iot=aws)[0]["status"] == "off"
