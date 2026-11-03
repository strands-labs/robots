"""The IoT/IAM policy documents (MAPPING 3.2): the fleet policy fits the cap and names no literal tenant or thing,
and the operator policy scopes the shadow to the tenant's own device ids rather than the whole account."""

from __future__ import annotations

import json

import pytest

from strands_robots.mesh.iot import policy

ACCOUNT, REGION = "123456789012", "us-west-2"


def _by_action(doc, action):
    return [s for s in doc["Statement"] if s["Action"] == action or action in s["Action"]]


def test_fleet_policy_fits_the_cap_and_names_no_tenant_or_thing():
    text = policy.policy_text(policy.device_policy(REGION, ACCOUNT))
    assert len(text) <= policy.POLICY_CAP, len(text)
    assert "${iot:Connection.Thing.ThingName}" in text and "${iot:Connection.Thing.Attributes[tenant]}" in text
    assert "lab" not in text and "arm" not in text
    doc = json.loads(text)
    actions = {a for s in doc["Statement"] for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])}
    assert actions == {"iot:Connect", "iot:Publish", "iot:RetainPublish", "iot:Subscribe", "iot:Receive"}
    pub = [s for s in doc["Statement"] if "iot:Publish" in s["Action"]][0]["Resource"]
    assert not any("@estop" in r for r in pub), "a device never publishes the tenant e-stop"
    assert any(r.endswith("@presence/${iot:Connection.Thing.ThingName}") for r in pub)


def test_operator_policy_has_no_publish_on_the_devices_word():
    text = policy.policy_text(policy.operator_policy(REGION, ACCOUNT, "lab", things=["arm-1", "arm-2"]))
    doc = json.loads(text)
    pub = _by_action(doc, "iot:Publish")[0]["Resource"]
    assert any(r.endswith("strands/lab/@estop") for r in pub)
    for word in ("@presence", "/manifest", "/lockout", "/state"):
        assert not any(r.endswith(word) for r in pub), word
    assert any(s["Action"] == "iot:SendDirectMessage" for s in doc["Statement"])


def test_operator_policy_scopes_the_shadow_to_the_tenants_things_not_the_account():
    # The shadow path has no author gate, so the IAM policy is the boundary: it must name this tenant's device
    # ids and never grant a wildcard thing, or an operator of one tenant could move another tenant's robot.
    text = policy.policy_text(policy.operator_policy(REGION, ACCOUNT, "lab", things=["arm-1", "arm-2"]))
    doc = json.loads(text)
    pub = _by_action(doc, "iot:Publish")[0]["Resource"]
    assert not any("$aws/things/*" in r for r in pub), "no wildcard-thing shadow publish"
    assert any(r.endswith("$aws/things/arm-1/shadow/name/control/update") for r in pub)
    assert any(r.endswith("$aws/things/arm-2/shadow/name/control/update") for r in pub)
    for stmt in doc["Statement"]:
        if "iot:UpdateThingShadow" in stmt["Action"]:
            assert stmt["Resource"] != "*", "UpdateThingShadow is never account-wide in the default document"
            assert all(r.endswith(("thing/arm-1/control", "thing/arm-2/control")) for r in stmt["Resource"])


def test_operator_policy_refuses_an_empty_thing_list():
    with pytest.raises(ValueError, match="device ids"):
        policy.operator_policy(REGION, ACCOUNT, "lab")
    with pytest.raises(ValueError, match="device ids"):
        policy.operator_policy(REGION, ACCOUNT, "lab", things=[])


def test_operator_policy_all_things_opts_into_the_account_wide_grant():
    # The single-tenant-account escape hatch is explicit and still carries the wildcard it is documented to.
    doc = json.loads(policy.policy_text(policy.operator_policy(REGION, ACCOUNT, "lab", all_things=True)))
    pub = _by_action(doc, "iot:Publish")[0]["Resource"]
    assert any("$aws/things/*/shadow/name/control/update" in r for r in pub)
    assert any("iot:UpdateThingShadow" in s["Action"] and s["Resource"] == "*" for s in doc["Statement"])


def test_operator_policy_names_the_control_shadow_as_iam_does():
    # AWS scopes a NAMED shadow as thing/<name>/<shadow> (iot-action-resources: thing/thingOne/shadowOne); a grant
    # on thing/<name> alone covers the classic shadow, so the docs' update_thing_shadow(shadowName="control")
    # over HTTPS would be refused, and a grant on thing/<name>/* would let an operator write `state` and
    # `manifest`, the device's word. Operators write `control` only and read every shadow.
    doc = json.loads(policy.policy_text(policy.operator_policy(REGION, ACCOUNT, "lab", things=["arm-1", "arm-2"])))
    upd = [s for s in doc["Statement"] if "iot:UpdateThingShadow" in s["Action"]]
    assert len(upd) == 1 and "iot:GetThingShadow" not in upd[0]["Action"], "update and read are scoped apart"
    assert upd[0]["Resource"] == [
        f"arn:aws:iot:{REGION}:{ACCOUNT}:thing/arm-1/control",
        f"arn:aws:iot:{REGION}:{ACCOUNT}:thing/arm-2/control",
    ]
    get = [s for s in doc["Statement"] if "iot:GetThingShadow" in s["Action"]][0]["Resource"]
    assert (
        f"arn:aws:iot:{REGION}:{ACCOUNT}:thing/arm-1/*" in get
        and f"arn:aws:iot:{REGION}:{ACCOUNT}:thing/arm-2/*" in get
    )
    assert len(policy.policy_text(doc)) <= policy.IAM_POLICY_CAP


def test_operator_policy_refuses_a_document_over_iams_cap_by_name():
    # IAM refuses a managed policy over 6,144 characters; each device adds six ARNs, so about nine fit. The
    # refusal names the cap and the fix (several documents), instead of IAM's MalformedPolicyDocument later.
    with pytest.raises(ValueError, match="6144"):
        policy.operator_policy(REGION, ACCOUNT, "lab", things=[f"arm-{i}" for i in range(40)])
    policy.operator_policy(REGION, ACCOUNT, "lab", things=[f"arm-{i}" for i in range(8)])  # fits
