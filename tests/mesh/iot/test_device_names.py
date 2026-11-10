"""A robot's thing name is its MQTT client id, and IoT drops the older session when two clients share one: so no
device may be named like an operator (``strands-op-…``) or a robot mid-claim (``strands-claim-…``)."""

from __future__ import annotations

import ast
import warnings

import pytest
from conftest import ACCOUNT, REGION
from test_claim_registry import _approved, _ask, _cert_pem

from strands_robots.mesh.iot import claim, claim_hook, lifecycle, policy, provision

OPERATOR = policy.operator_client_prefix("lab") + "c0ffee12ab34"
TAKEN = [OPERATOR, "strands-op-x", "STRANDS-OP-lab-1", "strands-claim-arm-1"]


@pytest.mark.parametrize("name", TAKEN)
def test_a_reserved_client_id_is_no_device_id(name):
    with pytest.raises(ValueError, match="operator and claim client ids"):
        policy.check_device_id(name)


@pytest.mark.parametrize("name", ["arm-1", "strands-1a2b3c4d5e6f", "strands-operator", "op-arm"])
def test_ordinary_names_stay_device_ids(name):
    assert policy.check_device_id(name) == name


@pytest.mark.parametrize("name", TAKEN)
def test_register_refuses_the_name_before_it_creates_anything(aws, tmp_path, name):
    provision.setup_tenant(aws, "labb", region=REGION, account=ACCOUNT)
    with pytest.raises(ValueError, match="operator and claim client ids"):
        provision.register(aws, {"id": name, "device_type": "robot_arm", "tenant": "labb"}, tmp_path)
    assert aws.list_things()["things"] == []


def test_register_checks_the_name_with_an_hsm_csr_too(aws, tmp_path):
    _, csr_pem = provision.csr("arm-1")
    with pytest.raises(ValueError, match="device_id"):
        provision.register(aws, {"id": "arm/1", "device_type": "robot_arm", "tenant": "lab"}, tmp_path,
                           csr_pem=csr_pem, key="pkcs11:token=t;object=k")  # fmt: skip
    assert aws.list_things()["things"] == []


def test_an_approval_a_claim_and_the_policies_refuse_the_name(tmp_path):
    with pytest.raises(ValueError, match="operator and claim client ids"):
        claim_hook.allow(object(), OPERATOR, tenant="labb", device_type="robot_arm")
    with pytest.raises(ValueError, match="operator and claim client ids"):
        claim.claim("e", "c", "k", {"id": OPERATOR, "device_type": "robot_arm", "tenant": "labb"}, tmp_path, secret="s")
    with pytest.raises(ValueError, match="operator and claim client ids"):
        policy.operator_policy(REGION, ACCOUNT, "labb", things=["arm-1", OPERATOR])
    with pytest.raises(ValueError, match="operator and claim client ids"):
        lifecycle.clear_presence_rule("labb", "arn:aws:iam::123456789012:role/r", things=[OPERATOR])


@pytest.mark.parametrize("name", TAKEN)
def test_the_claim_hook_refuses_the_name_even_when_approved(name):
    assert claim.hook_decision(_ask(name), _cert_pem(name), approved=_approved(name)) is False
    assert claim.hook_decision(_ask("arm-7"), _cert_pem("arm-7"), approved=_approved("arm-7")) is True


def test_the_hook_and_the_library_reserve_the_same_prefixes():
    tree = ast.parse(claim_hook.HOOK_SOURCE)
    (value,) = [n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "RESERVED"]
    assert ast.literal_eval(value) == policy.RESERVED_CLIENT_PREFIXES


def test_all_things_warns_and_cannot_be_mixed_with_things():
    with pytest.warns(UserWarning, match="single-tenant"):
        policy.operator_policy(REGION, ACCOUNT, "lab", all_things=True)
    with pytest.raises(ValueError, match="not both"):
        policy.operator_policy(REGION, ACCOUNT, "lab", things=["arm-1"], all_things=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        policy.operator_policy(REGION, ACCOUNT, "lab", things=["arm-1"])
