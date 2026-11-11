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
    pub = [r for s in doc["Statement"] if "iot:Publish" in s["Action"] for r in s["Resource"]]
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
    with pytest.warns(UserWarning, match="single-tenant"):
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
    # IAM refuses a managed policy over 6,144 characters; each device adds nine ARNs, so about seven fit. The
    # refusal names the cap and the fix (several documents), instead of IAM's MalformedPolicyDocument later.
    with pytest.raises(ValueError, match="6144"):
        policy.operator_policy(REGION, ACCOUNT, "lab", things=[f"arm-{i}" for i in range(40)])
    policy.operator_policy(REGION, ACCOUNT, "lab", things=[f"arm-{i}" for i in range(7)])  # fits


def _resources(doc, action):
    out = []
    for s in _by_action(doc, action):
        out += [s["Resource"]] if isinstance(s["Resource"], str) else s["Resource"]
    return out


def test_a_device_cannot_publish_a_liveliness_token_or_retain_a_reply():
    # A device's certificate is not a commander: a forged @peer beat would keep a dead operator "alive" for the
    # watchdog of every robot in the tenant, and a forged goodbye would fire their on_disconnect.
    doc = policy.device_policy(REGION, ACCOUNT)
    assert not any("@peer" in r for r in _resources(doc, "iot:Publish"))
    retained = _resources(doc, "iot:RetainPublish")
    assert not any(w in r for r in retained for w in ("@reply", "@presence", "@peer", "$aws/rules")), retained
    assert any(r.endswith("topicfilter/strands/${iot:Connection.Thing.Attributes[tenant]}/@peer/*")
               for r in _resources(doc, "iot:Subscribe")), "the watchdog still hears its commanders"  # fmt: skip
    ctl = policy.controller_policy(REGION, ACCOUNT)
    assert any(r.endswith("/@peer/*") for r in _resources(ctl, "iot:Publish")), "a commander keeps its beat"


def test_a_device_reaches_only_its_own_telemetry_rule():
    doc = policy.device_policy(REGION, ACCOUNT)
    rules = [r for r in _resources(doc, "iot:Publish") if "$aws/rules/" in r]
    assert rules == [
        f"arn:aws:iot:{REGION}:{ACCOUNT}:topic/$aws/rules/${{iot:Connection.Thing.Attributes[telemetry_rule]}}/"
        "strands/${iot:Connection.Thing.Attributes[tenant]}/${iot:Connection.Thing.ThingName}/state"
    ]
    assert not any("$aws/rules/*" in r for r in _resources(doc, "iot:Publish"))


def test_a_thing_without_a_telemetry_rule_attribute_still_answers():
    # IoT skips a whole statement when one of its variables does not resolve; a thing registered before the
    # telemetry_rule attribute existed must keep its presence and reply grants until it is registered again.
    doc = policy.device_policy(REGION, ACCOUNT)
    for s in _by_action(doc, "iot:Publish"):
        res = [s["Resource"]] if isinstance(s["Resource"], str) else s["Resource"]
        if any("telemetry_rule" in r for r in res):
            assert len(res) == 1, res


def test_the_operator_hears_and_messages_only_its_tenants_devices():
    doc = policy.operator_policy(REGION, ACCOUNT, "lab", things=["arm-1", "arm-2"])
    life = [r for r in _resources(doc, "iot:Receive") if "$aws/events/presence" in r]
    assert sorted(r.rsplit("/", 1)[-1] for r in life) == ["arm-1", "arm-2"], life  # Receive is the isolation
    (dm,) = _by_action(doc, "iot:SendDirectMessage")
    assert dm["Resource"] == [
        f"arn:aws:iot:{REGION}:{ACCOUNT}:client/arm-1",
        f"arn:aws:iot:{REGION}:{ACCOUNT}:client/arm-2",
    ]
    assert dm["Condition"] == {"StringLike": {"iot:Topic": ["strands/lab/*/rpc/*", "strands/lab/*/manifest"]}}
    with pytest.warns(UserWarning, match="EVERY thing"):
        wide = policy.operator_policy(REGION, ACCOUNT, "lab", all_things=True)
    (dm,) = _by_action(wide, "iot:SendDirectMessage")
    tenant_bound = {"StringLike": {"iot:Topic": ["strands/lab/*/rpc/*", "strands/lab/*/manifest"]}}
    assert dm["Condition"] == tenant_bound, "the tenant bound stays"


def test_the_operator_connects_only_with_its_tenants_client_ids():
    (connect,) = _by_action(policy.operator_policy(REGION, ACCOUNT, "lab", things=["arm-1"]), "iot:Connect")
    assert connect["Resource"] == f"arn:aws:iot:{REGION}:{ACCOUNT}:client/strands-op-lab-*"
    assert policy.operator_client_prefix("lab") == "strands-op-lab-"
