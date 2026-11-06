"""A leaked claim certificate names no robot: only an approved, new name, with its one-time secret, in the tenant and
type the operator approved, once; and the template is never left without its hook."""

from __future__ import annotations

import hashlib
import json

import pytest
from test_claim_registry import SECRET, _approved, _ask, _cert_pem

from strands_robots.mesh.iot import claim, claim_hook


def test_an_unapproved_name_is_refused_even_with_a_matching_cn():
    assert claim.hook_decision(_ask("arm-9"), _cert_pem("arm-9"), approved=_approved("arm-7")) is False


@pytest.mark.parametrize(
    "over",
    [{"claim_secret": "guess"}, {"claim_secret": ""}, {"tenant": "other"}, {"device_type": "humanoid"}],
    ids=["wrong-secret", "no-secret", "another-tenant", "another-type"],
)
def test_the_request_must_carry_the_approvals_secret_tenant_and_type(over):
    assert claim.hook_decision(_ask("arm-7", **over), _cert_pem("arm-7"), approved=_approved("arm-7")) is False


def test_an_approval_is_used_once_and_the_bound_values_override_the_request():
    approved = _approved("arm-7")
    ns: dict = {}
    exec(claim.HOOK_SOURCE, ns)  # noqa: S102
    ns["thing_exists"] = lambda name: False
    ns["allowed"] = approved.get
    ns["consume"] = approved.pop
    out = ns["handler"]({"parameters": _ask("arm-7"), "certificatePem": _cert_pem("arm-7")}, None)
    bound = {"device_id": "arm-7", "tenant": "lab", "device_type": "robot_arm"}
    assert out == {"allowProvisioning": True, "parameterOverrides": bound}
    assert approved == {}
    again = ns["handler"]({"parameters": _ask("arm-7"), "certificatePem": _cert_pem("arm-7")}, None)
    assert again["allowProvisioning"] is False


class FakeSsm:
    def __init__(self) -> None:
        self.params: dict = {}

    def put_parameter(self, *, Name, Value, Type, Overwrite):  # noqa: N803 — boto3's names
        assert Type == "SecureString" and Overwrite is False
        if Name in self.params:
            raise RuntimeError("ParameterAlreadyExists")
        self.params[Name] = Value


def test_allow_writes_a_hashed_entry_and_returns_the_secret_once():
    ssm = FakeSsm()
    secret = claim_hook.allow(ssm, "arm-7", tenant="lab", device_type="robot_arm")
    ((name, value),) = ssm.params.items()
    assert name == claim_hook.allow_name("arm-7") and name.startswith("/strands-robots/claim/") and "arm-7" not in name
    entry = json.loads(value)
    assert entry["tenant"] == "lab" and entry["secret_sha256"] == hashlib.sha256(secret.encode()).hexdigest()
    assert secret not in value
    with pytest.raises(RuntimeError):
        claim_hook.allow(ssm, "arm-7", tenant="other", device_type="robot_arm")  # a pending approval is never replaced
    assert SECRET != secret


class FakeIot:
    def __init__(self, exists: bool) -> None:
        self.exists, self.calls = exists, []

    def create_provisioning_template(self, **kw):
        self.calls.append(("create", kw))
        if self.exists:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "ResourceAlreadyExistsException"}}, "CreateProvisioningTemplate")

    def create_provisioning_template_version(self, **kw):
        self.calls.append(("version", kw))

    def update_provisioning_template(self, **kw):
        self.calls.append(("update", kw))


def test_a_template_created_earlier_without_a_hook_gets_it_on_the_next_run():
    iot = FakeIot(exists=True)
    assert claim.create_template(iot, "arn:role", hook_arn="arn:hook") == "updated"
    ((_, update),) = [c for c in iot.calls if c[0] == "update"]
    assert update["preProvisioningHook"] == {"targetArn": "arn:hook", "payloadVersion": "2020-04-01"}
    assert update["enabled"] is True and update["provisioningRoleArn"] == "arn:role"


def test_a_template_without_a_hook_is_refused():
    with pytest.raises(ValueError, match="hook_arn"):
        claim.create_template(FakeIot(exists=False), "arn:role", hook_arn="")
    with pytest.raises(TypeError):
        claim.create_template(FakeIot(exists=False), "arn:role")  # type: ignore[call-arg]
