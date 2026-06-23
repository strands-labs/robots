"""Unit tests for the account-wide bootstrap.

Real AWS calls are mocked. The end-to-end "really creates resources" test
is in /tmp/test_bootstrap.py and was run manually against us-west-2 during
implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh.iot import bootstrap as boot_mod
from strands_robots.mesh.iot.bootstrap import (
    ESTOP_LAMBDA_NAME,
    LOG_GROUP_NAME,
    PROVISIONING_TEMPLATE,
    RULE_ESTOP_FANOUT,
    RULE_SAFETY_TO_DYNAMODB,
    SAFETY_TABLE_NAME,
    BootstrappedAccount,
    _build_lambda_zip,
)


class TestLambdaZipBuild:
    def test_zip_contains_handler(self):
        zb = _build_lambda_zip()
        # First two bytes are the zip magic (PK)
        assert zb[:2] == b"PK"
        import zipfile
        from io import BytesIO

        with zipfile.ZipFile(BytesIO(zb)) as zf:
            names = zf.namelist()
            assert "lambda_function.py" in names
            src = zf.read("lambda_function.py").decode()
            assert "def lambda_handler" in src
            assert "iot_data.publish" in src
            assert "strands-mesh-role" in src


class TestResourceNames:
    """Lock in the canonical names - changing them is a breaking change."""

    def test_safety_table_name(self):
        assert SAFETY_TABLE_NAME == "strands-mesh-safety-events"

    def test_estop_lambda_name(self):
        assert ESTOP_LAMBDA_NAME == "strands-mesh-estop-fanout"

    def test_rule_names(self):
        assert RULE_SAFETY_TO_DYNAMODB == "strands_safety_to_dynamodb"
        assert RULE_ESTOP_FANOUT == "strands_estop_fanout"

    def test_provisioning_template_name(self):
        assert PROVISIONING_TEMPLATE == "strands-mesh-fleet-provisioning"

    def test_log_group_under_aws_iot(self):
        # Must be under /aws/iot/ for IoT Logs Role to deliver logs natively.
        assert LOG_GROUP_NAME.startswith("/aws/iot/")


class TestBootstrappedAccountDataclass:
    def test_default_lists_empty(self):
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        assert a.created == []
        assert a.skipped == []


class TestBootstrapRequireBoto3:
    def test_helpful_import_error(self):
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                boot_mod._require_boto3()


class TestEnsureSafetyTable:
    def test_skips_when_present(self):
        ddb = MagicMock()
        ddb.describe_table.return_value = {"Table": {"TableArn": "arn:test"}}
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        result = boot_mod._ensure_safety_table(ddb, a)
        assert result == "arn:test"
        ddb.create_table.assert_not_called()
        assert "dynamodb:strands-mesh-safety-events" in a.skipped

    def test_creates_when_missing(self):
        class _NotFound(Exception):
            pass

        ddb = MagicMock()
        ddb.exceptions = MagicMock()
        ddb.exceptions.ResourceNotFoundException = _NotFound
        ddb.describe_table.side_effect = _NotFound("not found")
        ddb.create_table.return_value = {"TableDescription": {"TableArn": "arn:new"}}
        a = BootstrappedAccount(region="us-west-2", account_id="123")

        result = boot_mod._ensure_safety_table(ddb, a)

        assert result == "arn:new"
        ddb.create_table.assert_called_once()
        kwargs = ddb.create_table.call_args.kwargs
        # Must request KMS encryption - safety audit is a security-sensitive table.
        assert kwargs["SSESpecification"]["Enabled"] is True
        assert kwargs["SSESpecification"]["SSEType"] == "KMS"
        # Pay-per-request (so test accounts don't get autoscaling surprises).
        assert kwargs["BillingMode"] == "PAY_PER_REQUEST"
        # Tags must include strands-mesh=managed for IaC discoverability.
        tag_keys = {t["Key"] for t in kwargs["Tags"]}
        assert "strands-mesh" in tag_keys
        assert "dynamodb:strands-mesh-safety-events" in a.created


class TestEnsureRuleIdempotence:
    def test_skips_when_get_returns_existing(self):
        iot = MagicMock()
        iot.get_topic_rule.return_value = {"ruleArn": "arn:rule"}
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        with patch(
            "strands_robots.mesh.iot.bootstrap._ensure_iot_action_role",
            return_value="arn:iam:role",
        ):
            result = boot_mod._ensure_safety_to_dynamodb_rule(iot, "arn:t", a)
        assert result == "arn:rule"
        iot.create_topic_rule.assert_not_called()

    def test_skips_when_get_raises_unauthorised(self):
        """AWS IoT returns UnauthorizedException when a rule doesn't exist."""

        class _NotFound(Exception):
            pass

        class _Unauthorized(Exception):
            pass

        iot = MagicMock()
        iot.exceptions = MagicMock()
        iot.exceptions.ResourceNotFoundException = _NotFound
        iot.exceptions.UnauthorizedException = _Unauthorized
        iot.get_topic_rule.side_effect = _Unauthorized("denied")

        a = BootstrappedAccount(region="us-west-2", account_id="123")
        with patch(
            "strands_robots.mesh.iot.bootstrap._ensure_iot_action_role",
            return_value="arn:iam:role",
        ):
            boot_mod._ensure_safety_to_dynamodb_rule(iot, "arn:t", a)

        # Rule was created (UnauthorizedException treated as not-found).
        iot.create_topic_rule.assert_called_once()


class TestEnsureLogGroup:
    def test_creates_when_missing(self):
        from strands_robots.mesh.iot.bootstrap import LOG_GROUP_NAME, BootstrappedAccount, _ensure_log_group

        logs = MagicMock()
        # First describe returns no matching group
        logs.describe_log_groups.side_effect = [
            {"logGroups": []},  # initial check
            {
                "logGroups": [{"arn": "arn:aws:logs:us-west-2:123:log-group:test", "logGroupName": LOG_GROUP_NAME}]
            },  # post-create check
        ]
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_log_group(logs, a)
        assert arn.startswith("arn:aws:logs")
        logs.create_log_group.assert_called_once()
        logs.put_retention_policy.assert_called_once()
        assert f"logs:{LOG_GROUP_NAME}" in a.created

    def test_skips_when_present(self):
        from strands_robots.mesh.iot.bootstrap import LOG_GROUP_NAME, BootstrappedAccount, _ensure_log_group

        logs = MagicMock()
        logs.describe_log_groups.return_value = {"logGroups": [{"arn": "arn:logs", "logGroupName": LOG_GROUP_NAME}]}
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_log_group(logs, a)
        assert arn == "arn:logs"
        logs.create_log_group.assert_not_called()
        assert f"logs:{LOG_GROUP_NAME}" in a.skipped


class TestEnsureLambdaRole:
    def test_skips_when_present(self):
        from strands_robots.mesh.iot.bootstrap import ESTOP_LAMBDA_ROLE, BootstrappedAccount, _ensure_lambda_role

        iam = MagicMock()
        iam.get_role.return_value = {"Role": {"Arn": "arn:iam:role"}}
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_lambda_role(iam, a)
        assert arn == "arn:iam:role"
        iam.create_role.assert_not_called()
        assert f"iam:{ESTOP_LAMBDA_ROLE}" in a.skipped


class TestEnsureEstopLambda:
    def test_skips_when_present(self):
        from strands_robots.mesh.iot.bootstrap import BootstrappedAccount, _ensure_estop_lambda

        lam = MagicMock()
        lam.get_function.return_value = {"Configuration": {"FunctionArn": "arn:lambda"}}
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_estop_lambda(lam, "arn:role", a)
        assert arn == "arn:lambda"
        lam.create_function.assert_not_called()


class TestGrantIotInvokeLambda:
    def test_grants_permission_when_missing(self):
        from strands_robots.mesh.iot.bootstrap import BootstrappedAccount, _grant_iot_invoke_lambda

        lam = MagicMock()
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        _grant_iot_invoke_lambda(lam, "arn:lambda", a)
        lam.add_permission.assert_called_once()
        kwargs = lam.add_permission.call_args.kwargs
        assert kwargs["Principal"] == "iot.amazonaws.com"
        assert kwargs["Action"] == "lambda:InvokeFunction"

    def test_swallows_existing_permission(self):
        from strands_robots.mesh.iot.bootstrap import BootstrappedAccount, _grant_iot_invoke_lambda

        class _Conflict(Exception):
            pass

        lam = MagicMock()
        lam.exceptions = MagicMock()
        lam.exceptions.ResourceConflictException = _Conflict
        lam.add_permission.side_effect = _Conflict("already granted")
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        # Must not raise
        _grant_iot_invoke_lambda(lam, "arn:lambda", a)


class TestBootstrappedAccountEnv:
    """Sanity checks - names locked in."""

    def test_lambda_zip_size_reasonable(self):
        from strands_robots.mesh.iot.bootstrap import _build_lambda_zip

        zb = _build_lambda_zip()
        # Lambda source is ~2 KB; zipped should be well under 10 KB.
        assert 500 < len(zb) < 10_000


# === Coverage gaps: create-paths for _ensure_* helpers ===


@pytest.fixture(autouse=False)
def _no_sleep(monkeypatch):
    """IAM role propagation has a `time.sleep(8)`; mock it in tests."""
    monkeypatch.setattr("time.sleep", lambda *a, **kw: None)


class TestEnsureLambdaRoleCreate:
    """Cover the `_ensure_lambda_role` create-path (skipped covered above)."""

    def test_creates_with_correct_trust_and_policies(self, _no_sleep):
        from strands_robots.mesh.iot.bootstrap import (
            ESTOP_LAMBDA_ROLE,
            BootstrappedAccount,
            _ensure_lambda_role,
        )

        class _NotFound(Exception):
            pass

        iam = MagicMock()
        iam.exceptions = MagicMock()
        iam.exceptions.NoSuchEntityException = _NotFound
        iam.get_role.side_effect = _NotFound("no role")
        iam.create_role.return_value = {"Role": {"Arn": "arn:iam:role/created"}}

        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_lambda_role(iam, a)

        assert arn == "arn:iam:role/created"
        # Trust policy must allow lambda.amazonaws.com to AssumeRole
        create_kwargs = iam.create_role.call_args.kwargs
        import json as _json

        trust = _json.loads(create_kwargs["AssumeRolePolicyDocument"])
        assert trust["Statement"][0]["Principal"] == {"Service": "lambda.amazonaws.com"}
        # AWS basic execution + our publish policy attached
        iam.attach_role_policy.assert_called_once()
        iam.put_role_policy.assert_called_once()
        publish_kwargs = iam.put_role_policy.call_args.kwargs
        assert publish_kwargs["PolicyName"] == "strands-mesh-iot-publish"
        # Inline policy must scope iot:Publish to strands/* topics
        policy = _json.loads(publish_kwargs["PolicyDocument"])
        publish_stmt = next(s for s in policy["Statement"] if "iot:Publish" in s["Action"])
        assert any("strands/*" in r for r in publish_stmt["Resource"])
        assert f"iam:{ESTOP_LAMBDA_ROLE}" in a.created


class TestEnsureEstopLambdaCreate:
    def test_creates_when_missing(self, _no_sleep):
        from strands_robots.mesh.iot.bootstrap import (
            ESTOP_LAMBDA_NAME,
            BootstrappedAccount,
            _ensure_estop_lambda,
        )

        class _NotFound(Exception):
            pass

        lam = MagicMock()
        lam.exceptions = MagicMock()
        lam.exceptions.ResourceNotFoundException = _NotFound
        lam.get_function.side_effect = _NotFound()
        lam.create_function.return_value = {"FunctionArn": "arn:lambda:created"}

        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_estop_lambda(lam, "arn:role", a)

        assert arn == "arn:lambda:created"
        kw = lam.create_function.call_args.kwargs
        # Sanity: handler, runtime, version-tagged description
        assert kw["Handler"] == "lambda_function.lambda_handler"
        assert kw["Runtime"] == "python3.12"
        assert "[v" in kw["Description"], "description must carry version tag"
        # Source must be the zipped lambda we built
        assert kw["Code"]["ZipFile"][:2] == b"PK"
        assert f"lambda:{ESTOP_LAMBDA_NAME}" in a.created

    def test_force_update_replaces_stale_version(self, _no_sleep):
        from strands_robots.mesh.iot.bootstrap import (
            BootstrappedAccount,
            _ensure_estop_lambda,
        )

        lam = MagicMock()
        # Simulate an existing function with a stale version description
        lam.get_function.return_value = {
            "Configuration": {
                "Description": "strands-mesh: defence-in-depth E-stop fan-out [v0]",
                "FunctionArn": "arn:lambda:existing",
            }
        }
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_estop_lambda(lam, "arn:role", a, force_update=True)

        assert arn == "arn:lambda:existing"
        lam.update_function_code.assert_called_once()
        lam.update_function_configuration.assert_called_once()
        assert any("(updated)" in entry for entry in a.created)

    def test_warns_on_stale_version_without_force_update(self, _no_sleep, caplog):
        import logging

        from strands_robots.mesh.iot.bootstrap import (
            ESTOP_LAMBDA_NAME,
            BootstrappedAccount,
            _ensure_estop_lambda,
        )

        lam = MagicMock()
        lam.get_function.return_value = {
            "Configuration": {
                "Description": "strands-mesh: defence-in-depth E-stop fan-out [v0]",
                "FunctionArn": "arn:lambda:existing",
            }
        }
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.iot.bootstrap"):
            arn = _ensure_estop_lambda(lam, "arn:role", a, force_update=False)

        assert arn == "arn:lambda:existing"
        # No update call -- only WARNING
        lam.update_function_code.assert_not_called()
        assert any("stale version" in m for m in caplog.messages)
        assert f"lambda:{ESTOP_LAMBDA_NAME}" in a.skipped


class TestEnsureSafetyToDynamoDbRuleCreate:
    """The create-path: builds the IoT Rule action when the rule is missing."""

    def test_creates_rule_with_correct_sql_and_action(self):
        from strands_robots.mesh.iot.bootstrap import (
            BootstrappedAccount,
            _ensure_safety_to_dynamodb_rule,
        )

        class _NotFound(Exception):
            pass

        iot = MagicMock()
        iot.exceptions = MagicMock()
        iot.exceptions.ResourceNotFoundException = _NotFound
        iot.exceptions.UnauthorizedException = type("UE", (Exception,), {})
        iot.get_topic_rule.side_effect = _NotFound()
        iot.create_topic_rule.return_value = None
        # After creation, get_topic_rule is called again to retrieve the ARN
        iot.list_topic_rules.return_value = {
            "rules": [{"ruleName": "strands_safety_to_dynamodb", "ruleArn": "arn:rule:safety"}]
        }

        a = BootstrappedAccount(region="us-west-2", account_id="123")
        with patch(
            "strands_robots.mesh.iot.bootstrap._ensure_iot_action_role",
            return_value="arn:iam:action-role",
        ):
            arn = _ensure_safety_to_dynamodb_rule(iot, "arn:t:safety", a)

        assert arn  # non-empty
        iot.create_topic_rule.assert_called_once()
        kw = iot.create_topic_rule.call_args.kwargs
        # SQL select on safety/event topic
        sql = kw["topicRulePayload"]["sql"]
        assert "safety/event" in sql
        # DynamoDBv2 action wired
        actions = kw["topicRulePayload"]["actions"]
        assert any("dynamoDBv2" in a or "dynamoDBv2" in str(a) for a in actions)


class TestEnsureIotActionRoleCreate:
    """Tests the `_ensure_iot_action_role` create-path (require_boto3 wrapper)."""

    def test_creates_role_with_dynamodb_putitem_policy(self, _no_sleep, monkeypatch):
        from strands_robots.mesh.iot.bootstrap import (
            BootstrappedAccount,
            _ensure_iot_action_role,
        )

        class _NotFound(Exception):
            pass

        iam = MagicMock()
        iam.exceptions = MagicMock()
        iam.exceptions.NoSuchEntityException = _NotFound
        iam.get_role.side_effect = _NotFound()
        iam.create_role.return_value = {"Role": {"Arn": "arn:iam:action"}}

        boto3_mock = MagicMock()
        boto3_mock.client.return_value = iam
        monkeypatch.setattr("strands_robots.mesh.iot.bootstrap._require_boto3", lambda: boto3_mock)

        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_iot_action_role(a)

        assert arn == "arn:iam:action"
        # Trust must allow iot.amazonaws.com
        import json as _json

        trust = _json.loads(iam.create_role.call_args.kwargs["AssumeRolePolicyDocument"])
        assert trust["Statement"][0]["Principal"] == {"Service": "iot.amazonaws.com"}
        # Inline policy must scope DynamoDB:PutItem to the safety table
        inline = _json.loads(iam.put_role_policy.call_args.kwargs["PolicyDocument"])
        stmt = inline["Statement"][0]
        assert stmt["Action"] == ["dynamodb:PutItem"]
        assert "strands-mesh-safety-events" in stmt["Resource"]
        assert any("iam:strands-mesh-iot-action-role" in entry for entry in a.created)


class TestEnsureProvisioningTemplateCreate:
    """Cover the provisioning-template create-path -- one of the largest
    untested code blocks in this module."""

    def test_creates_template_with_thing_resource(self, _no_sleep):
        from strands_robots.mesh.iot.bootstrap import (
            PROVISIONING_TEMPLATE,
            BootstrappedAccount,
            _ensure_provisioning_template,
        )

        class _NotFound(Exception):
            pass

        iot = MagicMock()
        iot.exceptions = MagicMock()
        iot.exceptions.ResourceNotFoundException = _NotFound
        iot.describe_provisioning_template.side_effect = _NotFound()
        iot.create_provisioning_template.return_value = {"templateArn": "arn:iot:template:provisioning"}

        a = BootstrappedAccount(region="us-west-2", account_id="123")
        with patch(
            "strands_robots.mesh.iot.bootstrap._ensure_provisioning_role",
            return_value="arn:iam:provisioning",
        ):
            arn = _ensure_provisioning_template(iot, a)

        assert arn == "arn:aws:iot:us-west-2:123:provisioningtemplate/strands-mesh-fleet-provisioning"
        kw = iot.create_provisioning_template.call_args.kwargs
        assert kw["templateName"] == PROVISIONING_TEMPLATE
        assert kw["enabled"] is True
        # Body must reference AWS::IoT::Thing
        body_str = kw["templateBody"]
        assert "AWS::IoT::Thing" in body_str
        assert "AWS::IoT::Certificate" in body_str
        assert "AWS::IoT::Policy" in body_str
        assert f"iot-prov-template:{PROVISIONING_TEMPLATE}" in a.created

    def test_skips_when_template_present(self):
        from strands_robots.mesh.iot.bootstrap import (
            PROVISIONING_TEMPLATE,
            BootstrappedAccount,
            _ensure_provisioning_template,
        )

        iot = MagicMock()
        iot.describe_provisioning_template.return_value = {"templateArn": "arn:iot:template:existing"}
        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_provisioning_template(iot, a)
        assert arn == "arn:aws:iot:us-west-2:123:provisioningtemplate/strands-mesh-fleet-provisioning"
        iot.create_provisioning_template.assert_not_called()
        assert f"iot-prov-template:{PROVISIONING_TEMPLATE}" in a.skipped


class TestEnsureProvisioningRoleCreate:
    def test_creates_role_with_provisioning_policy(self, _no_sleep, monkeypatch):
        from strands_robots.mesh.iot.bootstrap import (
            BootstrappedAccount,
            _ensure_provisioning_role,
        )

        class _NotFound(Exception):
            pass

        iam = MagicMock()
        iam.exceptions = MagicMock()
        iam.exceptions.NoSuchEntityException = _NotFound
        iam.get_role.side_effect = _NotFound()
        iam.create_role.return_value = {"Role": {"Arn": "arn:iam:provisioning"}}

        boto3_mock = MagicMock()
        boto3_mock.client.return_value = iam
        monkeypatch.setattr("strands_robots.mesh.iot.bootstrap._require_boto3", lambda: boto3_mock)

        a = BootstrappedAccount(region="us-west-2", account_id="123")
        arn = _ensure_provisioning_role(a)
        assert arn == "arn:iam:provisioning"
        # Managed policy attachment for fleet provisioning
        iam.attach_role_policy.assert_called()


class TestBootstrapAccountGuards:
    """The public bootstrap_account() refuses to create resources without
    explicit confirmation, and aborts on a wrong-account mismatch."""

    def test_requires_confirm_when_not_dry_run(self):
        with pytest.raises(ValueError, match="confirm=True"):
            boot_mod.bootstrap_account(dry_run=False, confirm=False)

    def test_dry_run_previews_without_creating(self, monkeypatch, capsys):
        sts = MagicMock()
        sts.get_caller_identity.return_value = {"Account": "111122223333"}
        sts.meta.region_name = "us-west-2"
        boto3_mock = MagicMock()
        boto3_mock.client.return_value = sts
        monkeypatch.setattr(boot_mod, "_require_boto3", lambda: boto3_mock)

        result = boot_mod.bootstrap_account(region="us-west-2")

        assert isinstance(result, BootstrappedAccount)
        assert result.account_id == "111122223333"
        assert result.region == "us-west-2"
        # Dry run touches only STS - never any mutating client.
        assert {c.args[0] for c in boto3_mock.client.call_args_list} == {"sts"}
        assert result.created == []
        # The preview lists the resources it would create, on stderr.
        err = capsys.readouterr().err
        assert "strands-mesh-fleet-provisioning" in err
        assert "dry_run=False, confirm=True" in err

    def test_aborts_on_account_id_mismatch(self, monkeypatch):
        sts = MagicMock()
        sts.get_caller_identity.return_value = {"Account": "999988887777"}
        sts.meta.region_name = "us-west-2"
        boto3_mock = MagicMock()
        boto3_mock.client.return_value = sts
        monkeypatch.setattr(boot_mod, "_require_boto3", lambda: boto3_mock)

        with pytest.raises(ValueError, match="does not match"):
            boot_mod.bootstrap_account(confirm=True, dry_run=False, account_id_expected="111122223333")


class TestBootstrapAccountProvisioning:
    """Full bootstrap orchestration (confirm=True, dry_run=False). The
    per-resource _ensure_* helpers have their own tests, so here we stub them
    and assert bootstrap_account wires their ARNs into the returned record
    and invokes them in dependency order."""

    def test_provisions_and_collects_arns(self, monkeypatch):
        sts = MagicMock()
        sts.get_caller_identity.return_value = {"Account": "111122223333"}
        sts.meta.region_name = "eu-central-1"
        boto3_mock = MagicMock()
        boto3_mock.client.return_value = sts
        monkeypatch.setattr(boot_mod, "_require_boto3", lambda: boto3_mock)

        calls: list[str] = []

        def _stub(name, ret):
            def _fn(*args, **kwargs):
                calls.append(name)
                return ret

            return _fn

        monkeypatch.setattr(boot_mod, "_ensure_log_group", _stub("log", "arn:logs"))
        monkeypatch.setattr(boot_mod, "_ensure_safety_table", _stub("ddb", "arn:ddb"))
        monkeypatch.setattr(boot_mod, "_ensure_lambda_role", _stub("lam_role", "arn:lam-role"))
        monkeypatch.setattr(boot_mod, "_ensure_estop_lambda", _stub("estop", "arn:estop"))
        monkeypatch.setattr(boot_mod, "_ensure_safety_to_dynamodb_rule", _stub("rule_safety", "arn:rule-safety"))
        monkeypatch.setattr(boot_mod, "_ensure_estop_rule", _stub("rule_estop", "arn:rule-estop"))
        monkeypatch.setattr(boot_mod, "_grant_iot_invoke_lambda", _stub("grant_estop", None))
        monkeypatch.setattr(boot_mod, "_ensure_provisioning_hook_role", _stub("hook_role", "arn:hook-role"))
        monkeypatch.setattr(boot_mod, "_ensure_provisioning_hook_lambda", _stub("hook", "arn:hook"))
        monkeypatch.setattr(boot_mod, "_ensure_provisioning_template", _stub("template", "arn:template"))
        monkeypatch.setattr(boot_mod, "_grant_iot_invoke_provisioning_hook", _stub("grant_hook", None))

        out = boot_mod.bootstrap_account(confirm=True, dry_run=False)

        assert out.account_id == "111122223333"
        assert out.region == "eu-central-1"
        assert out.log_group_arn == "arn:logs"
        assert out.safety_table_arn == "arn:ddb"
        assert out.estop_lambda_arn == "arn:estop"
        assert out.rule_safety_arn == "arn:rule-safety"
        assert out.rule_estop_arn == "arn:rule-estop"
        assert out.provisioning_hook_lambda_arn == "arn:hook"
        assert out.provisioning_template_arn == "arn:template"
        # Logs/table come before the Lambda that depends on the table ARN;
        # the hook role precedes the hook Lambda it grants.
        assert calls.index("log") < calls.index("estop")
        assert calls.index("ddb") < calls.index("rule_safety")
        assert calls.index("hook_role") < calls.index("hook")
        assert calls.index("hook") < calls.index("template")


class TestTeardownAccount:
    """teardown_account() is best-effort: it deletes every managed resource
    in dependency order and swallows per-resource failures so a partially
    provisioned account can still be cleaned up."""

    def test_deletes_resources_in_order(self, monkeypatch):
        iot = MagicMock()
        iam = MagicMock()
        lam = MagicMock()
        ddb = MagicMock()
        logs = MagicMock()
        iam.list_role_policies.return_value = {"PolicyNames": []}
        iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}

        clients = {"iot": iot, "iam": iam, "lambda": lam, "dynamodb": ddb, "logs": logs}
        boto3_mock = MagicMock()
        boto3_mock.client.side_effect = lambda svc, **kw: clients[svc]
        monkeypatch.setattr(boot_mod, "_require_boto3", lambda: boto3_mock)

        boot_mod.teardown_account(region="us-west-2")

        # Both topic rules removed.
        deleted_rules = {c.kwargs["ruleName"] for c in iot.delete_topic_rule.call_args_list}
        assert deleted_rules == {RULE_SAFETY_TO_DYNAMODB, RULE_ESTOP_FANOUT}
        # E-stop Lambda and DynamoDB table and log group and template all removed.
        lam.delete_function.assert_called_once()
        ddb.delete_table.assert_called_once_with(TableName=SAFETY_TABLE_NAME)
        logs.delete_log_group.assert_called_once_with(logGroupName=LOG_GROUP_NAME)
        iot.delete_provisioning_template.assert_called_once_with(templateName=PROVISIONING_TEMPLATE)
        # All three managed roles cleaned up.
        torn_roles = {c.kwargs["RoleName"] for c in iam.delete_role.call_args_list}
        assert torn_roles == {
            boot_mod.ESTOP_LAMBDA_ROLE,
            "strands-mesh-iot-action-role",
            boot_mod.PROVISIONING_ROLE,
        }

    def test_swallows_per_resource_failures(self, monkeypatch):
        iot = MagicMock()
        iam = MagicMock()
        lam = MagicMock()
        ddb = MagicMock()
        logs = MagicMock()
        # Every deletion raises - teardown must not propagate.
        iot.delete_topic_rule.side_effect = RuntimeError("boom")
        iot.delete_provisioning_template.side_effect = RuntimeError("boom")
        lam.delete_function.side_effect = RuntimeError("boom")
        ddb.delete_table.side_effect = RuntimeError("boom")
        logs.delete_log_group.side_effect = RuntimeError("boom")
        iam.list_role_policies.side_effect = RuntimeError("boom")

        clients = {"iot": iot, "iam": iam, "lambda": lam, "dynamodb": ddb, "logs": logs}
        boto3_mock = MagicMock()
        boto3_mock.client.side_effect = lambda svc, **kw: clients[svc]
        monkeypatch.setattr(boot_mod, "_require_boto3", lambda: boto3_mock)

        # Must complete without raising despite every call failing.
        boot_mod.teardown_account()
        ddb.delete_table.assert_called_once()


class TestEnsureEstopRule:
    """The E-stop fan-out IoT Rule (`_ensure_estop_rule`).

    This rule wires the ``strands/safety/estop`` MQTT topic to the E-stop
    fan-out Lambda. It is the WAN-side path that broadcasts an emergency stop
    to every robot in the fleet, so both its create-path (correct SQL + Lambda
    action) and its idempotent skip-path (an existing rule is not recreated)
    are pinned here.
    """

    def _iot_with_missing_rule(self):
        """A mocked IoT client whose ``get_topic_rule`` reports 'not found'."""

        class _NotFound(Exception):
            pass

        iot = MagicMock()
        iot.exceptions = MagicMock()
        iot.exceptions.ResourceNotFoundException = _NotFound
        iot.exceptions.UnauthorizedException = type("UE", (Exception,), {})
        iot.get_topic_rule.side_effect = _NotFound()
        iot.create_topic_rule.return_value = None
        return iot

    def test_creates_rule_with_estop_topic_and_lambda_action(self):
        iot = self._iot_with_missing_rule()
        a = BootstrappedAccount(region="us-west-2", account_id="123456789012")

        arn = boot_mod._ensure_estop_rule(iot, "arn:aws:lambda:us-west-2:123456789012:function:estop", a)

        # ARN points at the well-known E-stop rule in the caller's account/region.
        assert arn == f"arn:aws:iot:us-west-2:123456789012:rule/{RULE_ESTOP_FANOUT}"
        iot.create_topic_rule.assert_called_once()
        kw = iot.create_topic_rule.call_args.kwargs
        assert kw["ruleName"] == RULE_ESTOP_FANOUT
        payload = kw["topicRulePayload"]
        # SQL must select from the safety/estop topic, and the rule stays enabled.
        assert "strands/safety/estop" in payload["sql"]
        assert payload["ruleDisabled"] is False
        # The only action is a Lambda invocation of the supplied function ARN.
        actions = payload["actions"]
        assert len(actions) == 1
        assert actions[0]["lambda"]["functionArn"] == ("arn:aws:lambda:us-west-2:123456789012:function:estop")
        # Creation is recorded for the bootstrap summary.
        assert f"iot-rule:{RULE_ESTOP_FANOUT}" in a.created

    def test_skips_when_rule_already_present(self):
        iot = MagicMock()
        iot.get_topic_rule.return_value = {"ruleArn": "arn:aws:iot:us-west-2:123456789012:rule/x"}
        a = BootstrappedAccount(region="us-west-2", account_id="123456789012")

        arn = boot_mod._ensure_estop_rule(iot, "arn:lambda:estop", a)

        # Idempotent: no creation, ARN derived from account/region, skip recorded.
        iot.create_topic_rule.assert_not_called()
        assert arn == f"arn:aws:iot:us-west-2:123456789012:rule/{RULE_ESTOP_FANOUT}"
        assert f"iot-rule:{RULE_ESTOP_FANOUT}" in a.skipped

    def test_creates_when_get_raises_unauthorised(self):
        """IoT returns UnauthorizedException (not NotFound) for an absent rule."""

        class _NotFound(Exception):
            pass

        class _Unauthorized(Exception):
            pass

        iot = MagicMock()
        iot.exceptions = MagicMock()
        iot.exceptions.ResourceNotFoundException = _NotFound
        iot.exceptions.UnauthorizedException = _Unauthorized
        iot.get_topic_rule.side_effect = _Unauthorized("denied")
        a = BootstrappedAccount(region="eu-west-1", account_id="210987654321")

        arn = boot_mod._ensure_estop_rule(iot, "arn:lambda:estop", a)

        # UnauthorizedException is treated as "missing" -> the rule is created.
        iot.create_topic_rule.assert_called_once()
        assert arn == f"arn:aws:iot:eu-west-1:210987654321:rule/{RULE_ESTOP_FANOUT}"
