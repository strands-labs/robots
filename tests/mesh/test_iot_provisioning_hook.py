"""Tests for the Fleet Provisioning PreProvisioningHook.

Without a PreProvisioningHook, any holder of the shared claim cert can
register an arbitrary Thing. These tests pin the hook's deny-by-default
behaviour and that the template wires it.
"""

from __future__ import annotations

import ast
import inspect
from unittest.mock import MagicMock

from strands_robots.mesh.iot import bootstrap as b


def test_hook_source_is_valid_python():
    ast.parse(b._PROVISIONING_HOOK_SOURCE)


def test_hook_zip_builds():
    assert len(b._build_provisioning_hook_zip()) > 0


def test_template_wires_pre_provisioning_hook():
    src = inspect.getsource(b._ensure_provisioning_template)
    assert "preProvisioningHook" in src
    assert "hook_lambda_arn" in src


def test_template_create_includes_hook_when_arn_supplied():
    """create_provisioning_template must receive preProvisioningHook."""
    iot = MagicMock()
    iot.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})
    iot.exceptions.InvalidRequestException = type("IRE", (Exception,), {})
    # describe -> not found so it proceeds to create
    iot.describe_provisioning_template.side_effect = iot.exceptions.ResourceNotFoundException()
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    # stub the role helper to avoid IAM calls
    import strands_robots.mesh.iot.bootstrap as mod

    orig = mod._ensure_provisioning_role
    mod._ensure_provisioning_role = lambda *a, **k: "arn:aws:iam::123456789012:role/x"
    try:
        b._ensure_provisioning_template(
            iot, acct, hook_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:hook"
        )
    finally:
        mod._ensure_provisioning_role = orig

    kwargs = iot.create_provisioning_template.call_args.kwargs
    assert "preProvisioningHook" in kwargs
    assert kwargs["preProvisioningHook"]["targetArn"].endswith(":function:hook")


def test_template_omits_hook_when_no_arn():
    iot = MagicMock()
    iot.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})
    iot.exceptions.InvalidRequestException = type("IRE", (Exception,), {})
    iot.describe_provisioning_template.side_effect = iot.exceptions.ResourceNotFoundException()
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    import strands_robots.mesh.iot.bootstrap as mod

    orig = mod._ensure_provisioning_role
    mod._ensure_provisioning_role = lambda *a, **k: "arn:aws:iam::123456789012:role/x"
    try:
        b._ensure_provisioning_template(iot, acct)  # no hook arn
    finally:
        mod._ensure_provisioning_role = orig

    kwargs = iot.create_provisioning_template.call_args.kwargs
    assert "preProvisioningHook" not in kwargs


# --- Behavioural tests of the hook handler itself ------------------------


def _run_handler(event, *, thing_exists=False, serial_allowed=True):
    """Exec the hook source with a controllable fake boto3 and invoke it."""
    fake_boto3 = MagicMock()

    iot_client = MagicMock()
    ssm_client = MagicMock()

    class _RNF(Exception):
        pass

    class _PNF(Exception):
        pass

    iot_client.exceptions.ResourceNotFoundException = _RNF
    ssm_client.exceptions.ParameterNotFound = _PNF

    if thing_exists:
        iot_client.describe_thing.return_value = {"thingName": "x"}
    else:
        iot_client.describe_thing.side_effect = _RNF()

    if not serial_allowed:
        ssm_client.get_parameter.side_effect = _PNF()

    def _client(name, *a, **k):
        return {"iot": iot_client, "ssm": ssm_client}[name]

    fake_boto3.client.side_effect = _client

    # The hook source does `import boto3` at module level, which shadows
    # any exec-global we inject. Patch sys.modules so the import resolves
    # to our fake instead of the real SDK (which would hit AWS).
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"boto3": fake_boto3}):
        g: dict = {}
        exec(compile(b._PROVISIONING_HOOK_SOURCE, "<hook>", "exec"), g)
        return g["lambda_handler"](event, MagicMock())


def test_hook_allows_valid_allowlisted_serial():
    res = _run_handler(
        {"parameters": {"SerialNumber": "robot-001", "ThingName": "g1-robot-001"}},
        thing_exists=False,
        serial_allowed=True,
    )
    assert res == {"allowProvisioning": True}


def test_hook_denies_bad_serial():
    res = _run_handler(
        {"parameters": {"SerialNumber": "../../etc", "ThingName": "x"}},
    )
    assert res == {"allowProvisioning": False}


def test_hook_denies_missing_serial():
    res = _run_handler({"parameters": {"ThingName": "x"}})
    assert res == {"allowProvisioning": False}


def test_hook_denies_existing_thing():
    res = _run_handler(
        {"parameters": {"SerialNumber": "robot-001", "ThingName": "g1-robot-001"}},
        thing_exists=True,
    )
    assert res == {"allowProvisioning": False}


def test_hook_denies_serial_not_in_allowlist():
    res = _run_handler(
        {"parameters": {"SerialNumber": "robot-999", "ThingName": "g1-robot-999"}},
        thing_exists=False,
        serial_allowed=False,
    )
    assert res == {"allowProvisioning": False}


def test_hook_role_grants_describe_thing_and_ssm_getparameter():
    """The hook role must permit the two reads the hook makes (F-19/B-13).

    Regression for the review blocker: the hook was originally created with
    the E-stop Lambda role, which grants neither iot:DescribeThing nor
    ssm:GetParameter. Those calls would then AccessDenied, get swallowed by
    the deny-on-error envelope, and refuse *every* registration.
    """
    iam = MagicMock()

    class _NoSuchEntity(Exception):
        pass

    iam.exceptions.NoSuchEntityException = _NoSuchEntity
    iam.get_role.side_effect = _NoSuchEntity()
    iam.create_role.return_value = {
        "Role": {"Arn": "arn:aws:iam::123456789012:role/strands-mesh-provisioning-hook-role"}
    }
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    import strands_robots.mesh.iot.bootstrap as mod

    orig_sleep = mod.time.sleep
    mod.time.sleep = lambda *_a, **_k: None
    try:
        arn = b._ensure_provisioning_hook_role(iam, acct)
    finally:
        mod.time.sleep = orig_sleep

    assert arn.endswith(":role/strands-mesh-provisioning-hook-role")

    # The inline policy must grant both actions the hook needs.
    inline = iam.put_role_policy.call_args.kwargs
    import json as _json

    doc = _json.loads(inline["PolicyDocument"])
    actions = {a for stmt in doc["Statement"] for a in stmt["Action"]}
    assert "iot:DescribeThing" in actions
    assert "ssm:GetParameter" in actions
    # SSM read must be scoped to the allowlist namespace, not "*".
    ssm_stmt = next(s for s in doc["Statement"] if "ssm:GetParameter" in s["Action"])
    assert all("provisioning/allow/" in r for r in ssm_stmt["Resource"])
    assert all(r != "*" for r in ssm_stmt["Resource"])


def test_bootstrap_uses_dedicated_hook_role():
    """bootstrap_account must wire the hook to its own role, not the E-stop role."""
    src = inspect.getsource(b.bootstrap_account)
    assert "_ensure_provisioning_hook_role" in src


def test_hook_lambda_stamps_version_description():
    """Create path stamps the version tag so drift can be detected later."""
    lam = MagicMock()

    class _RNF(Exception):
        pass

    lam.exceptions.ResourceNotFoundException = _RNF
    lam.exceptions.InvalidParameterValueException = type("IPV", (Exception,), {})
    lam.get_function.side_effect = _RNF()
    lam.create_function.return_value = {
        "FunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:strands-mesh-provisioning-hook"
    }
    acct = b.BootstrappedAccount(region="us-east-1", account_id="123456789012")

    b._ensure_provisioning_hook_lambda(lam, "arn:aws:iam::123456789012:role/hook", acct)

    desc = lam.create_function.call_args.kwargs["Description"]
    assert f"[v{b._PROVISIONING_HOOK_VERSION}]" in desc
