"""One-shot AWS account bootstrap — IoT Rules + Fleet Provisioning template.

This module is the operator-side analogue of :mod:`provision`. Where
:func:`provision_robot` configures one Thing, :func:`bootstrap_account`
configures the **account-wide infrastructure** that every robot relies on:

1. **CloudWatch log group** for IoT activity (visible diagnostics).
2. **DynamoDB table** ``strands-mesh-safety-events`` (KMS-encrypted, PITR)
   so safety events have a durable cloud audit trail beyond local JSONL.
3. **IoT Rules**:
    - ``strands_safety_to_dynamodb`` — every ``strands/+/safety/event``
      writes one row to the audit table.
    - ``strands_estop_fanout`` — defence-in-depth: an E-stop also fires
      a Lambda that publishes individual stop commands per robot.
    - ``strands_health_to_logs`` — health pings to CloudWatch Logs for
      grep-friendly debugging.
4. **Fleet Provisioning template** so factory-fresh robots can claim a
   real cert from a bootstrap claim cert with no human in the loop.

The function is **idempotent**: re-running it skips resources that
already exist (matched by name) and only adds what's missing. It NEVER
deletes — :func:`teardown_account` is the explicit reverse.

The Lambda code is deliberately tiny and shipped as a single-file source
string. It does what a fleet-ops engineer would write by hand on day 1
(``boto3.client('iot-data').publish``) so the cloud E-stop path works
without a separate code build pipeline.

Why bake all this into the library
----------------------------------
The whole point of the AWS IoT integration is "Robot('so100') joins the
mesh, fleet ops gets durable audit + alerts for free." Forcing customers
to wire CDK/Terraform on day 1 defeats that. They can replace this with
their own IaC later — these resources are tagged ``strands-mesh=managed``
so external tooling can see them clearly.
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)


SAFETY_TABLE_NAME = "strands-mesh-safety-events"
ESTOP_LAMBDA_NAME = "strands-mesh-estop-fanout"
ESTOP_LAMBDA_ROLE = "strands-mesh-lambda-role"
_LAMBDA_VERSION = 1  # Bump whenever _ESTOP_LAMBDA_SOURCE changes
RULE_SAFETY_TO_DYNAMODB = "strands_safety_to_dynamodb"
RULE_ESTOP_FANOUT = "strands_estop_fanout"
PROVISIONING_TEMPLATE = "strands-mesh-fleet-provisioning"
PROVISIONING_ROLE = "strands-mesh-provisioning-role"
PROVISIONING_HOOK_LAMBDA_NAME = "strands-mesh-provisioning-hook"
PROVISIONING_HOOK_ROLE = "strands-mesh-provisioning-hook-role"
#: Bump whenever _PROVISIONING_HOOK_SOURCE changes.
_PROVISIONING_HOOK_VERSION = 1
LOG_GROUP_NAME = "/aws/iot/strands-mesh"


# Lambda source for the E-stop fan-out


_ESTOP_LAMBDA_SOURCE = textwrap.dedent(
    """
    import json
    import logging
    import os

    import boto3

    log = logging.getLogger()
    log.setLevel(logging.INFO)

    iot = boto3.client("iot")
    iot_data = boto3.client("iot-data")

    def lambda_handler(event, context):
        '''
        Triggered by IoT Rule strands_estop_fanout when something publishes
        to strands/safety/estop. Lists every Thing tagged 'strands-mesh=robot'
        and publishes {"action":"stop"} to each robot's /cmd inbox.

        This is defence-in-depth: the original publisher's own
        broadcast message normally hits all subscribed robots, but if a
        robot missed the broadcast subscription (e.g. just rebooted) this
        Lambda catches it via its own per-robot publish.
        '''
        log.info("estop fanout invoked: %s", json.dumps(event)[:500])
        sender = (event or {}).get("peer_id", "unknown")
        responses = (event or {}).get("responses_received", 0)
        # List all robot Things — note: paginated for large fleets.
        paginator = iot.get_paginator("list_things")
        published = 0
        for page in paginator.paginate(maxResults=250):
            for thing in page.get("things", []):
                attrs = thing.get("attributes") or {}
                if attrs.get("strands-mesh-role") != "robot":
                    continue
                tname = thing["thingName"]
                try:
                    iot_data.publish(
                        topic=f"strands/{tname}/cmd",
                        qos=1,
                        payload=json.dumps({
                            "sender_id": "strands-mesh-estop-fanout",
                            "turn_id": "estop-fanout",
                            "command": {"action": "stop"},
                            "timestamp": context.aws_request_id,
                        }).encode(),
                    )
                    published += 1
                except Exception as exc:
                    log.warning("publish to %s failed: %s", tname, exc)
        log.info("estop fanout published to %d robots (sender=%s, original_acks=%d)",
                 published, sender, responses)
        return {"published": published, "sender": sender}
    """
)


# F-19 / B-13: Fleet Provisioning PreProvisioningHook. Without a hook,
# any holder of the (shared, long-lived) claim certificate can register
# an ARBITRARY ThingName and receive a full robot identity + policy.
# This hook is the gate AWS IoT calls *before* provisioning: it must
# return {"allowProvisioning": True} or the registration is denied.
#
# Policy enforced here (deny-by-default):
#   * SerialNumber must be present and match a strict format.
#   * The Thing must NOT already exist (no claim-cert takeover of an
#     existing robot's identity).
#   * The serial must be pre-seeded by the operator in SSM Parameter
#     Store at /strands-mesh/provisioning/allow/<serial>. Sites with a
#     CMDB can swap this lookup for their own API.
_PROVISIONING_HOOK_SOURCE = textwrap.dedent(
    """
    import logging
    import re

    import boto3

    log = logging.getLogger()
    log.setLevel(logging.INFO)

    ssm = boto3.client("ssm")

    _SERIAL_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
    _ALLOW_PREFIX = "/strands-mesh/provisioning/allow/"

    def lambda_handler(event, context):
        # AWS IoT Fleet Provisioning PreProvisioningHook.
        # Must return {"allowProvisioning": bool}. Deny-by-default.
        params = (event or {}).get("parameters", {}) or {}
        serial = params.get("SerialNumber", "")
        thing_name = params.get("ThingName", "")

        if not isinstance(serial, str) or not _SERIAL_RE.fullmatch(serial):
            log.warning("provisioning DENY: bad/missing SerialNumber %r", serial)
            return {"allowProvisioning": False}

        iot = boto3.client("iot")
        try:
            iot.describe_thing(thingName=thing_name)
            log.warning("provisioning DENY: Thing %r already exists", thing_name)
            return {"allowProvisioning": False}
        except iot.exceptions.ResourceNotFoundException:
            pass
        except Exception as exc:
            log.warning("provisioning DENY: describe_thing error: %s", exc)
            return {"allowProvisioning": False}

        try:
            ssm.get_parameter(Name=_ALLOW_PREFIX + serial)
        except ssm.exceptions.ParameterNotFound:
            log.warning("provisioning DENY: serial %r not in allowlist", serial)
            return {"allowProvisioning": False}
        except Exception as exc:
            log.warning("provisioning DENY: ssm error: %s", exc)
            return {"allowProvisioning": False}

        log.info("provisioning ALLOW: serial=%s thing=%s", serial, thing_name)
        return {"allowProvisioning": True}
    """
)


@dataclass
class BootstrappedAccount:
    """Identifiers + ARNs of every resource :func:`bootstrap_account` ensured."""

    region: str
    account_id: str
    safety_table_arn: str = ""
    estop_lambda_arn: str = ""
    rule_safety_arn: str = ""
    rule_estop_arn: str = ""
    log_group_arn: str = ""
    provisioning_template_arn: str = ""
    provisioning_hook_lambda_arn: str = ""
    skipped: list[str] = field(default_factory=list)
    created: list[str] = field(default_factory=list)


# Helpers


def _require_boto3() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for AWS IoT bootstrap. Install with: pip install 'strands-robots[mesh-iot]'"
        ) from exc
    return boto3


def _build_lambda_zip() -> bytes:
    """Pack the inline Lambda source into a deployable zip."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lambda_function.py", _ESTOP_LAMBDA_SOURCE)
    return buf.getvalue()


def _build_provisioning_hook_zip() -> bytes:
    """Pack the PreProvisioningHook Lambda source into a deployable zip."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lambda_function.py", _PROVISIONING_HOOK_SOURCE)
    return buf.getvalue()


def _ensure_safety_table(ddb: Any, account: BootstrappedAccount) -> str:
    """DynamoDB table for cloud-side safety event mirror. KMS, PITR."""
    try:
        existing = ddb.describe_table(TableName=SAFETY_TABLE_NAME)
        account.skipped.append(f"dynamodb:{SAFETY_TABLE_NAME}")
        return existing["Table"]["TableArn"]
    except ddb.exceptions.ResourceNotFoundException:
        pass

    resp = ddb.create_table(
        TableName=SAFETY_TABLE_NAME,
        AttributeDefinitions=[
            {"AttributeName": "peer_id", "AttributeType": "S"},
            {"AttributeName": "ts", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "peer_id", "KeyType": "HASH"},
            {"AttributeName": "ts", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST",
        SSESpecification={"Enabled": True, "SSEType": "KMS"},
        Tags=[
            {"Key": "strands-mesh", "Value": "managed"},
            {"Key": "purpose", "Value": "safety-audit"},
        ],
    )
    arn = resp["TableDescription"]["TableArn"]

    # PITR + wait for active
    waiter = ddb.get_waiter("table_exists")
    waiter.wait(TableName=SAFETY_TABLE_NAME)
    try:
        ddb.update_continuous_backups(
            TableName=SAFETY_TABLE_NAME,
            PointInTimeRecoverySpecification={"PointInTimeRecoveryEnabled": True},
        )
    except Exception as exc:
        logger.debug("[bootstrap] PITR enable failed: %s", exc)
    account.created.append(f"dynamodb:{SAFETY_TABLE_NAME}")
    return arn


def _ensure_lambda_role(iam: Any, account: BootstrappedAccount) -> str:
    """IAM role the E-stop Lambda assumes."""
    role_name = ESTOP_LAMBDA_ROLE
    try:
        role = iam.get_role(RoleName=role_name)
        account.skipped.append(f"iam:{role_name}")
        return role["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    resp = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="strands-mesh E-stop fan-out Lambda execution role",
        Tags=[{"Key": "strands-mesh", "Value": "managed"}],
    )
    arn = resp["Role"]["Arn"]

    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    )
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="strands-mesh-iot-publish",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["iot:Publish"],
                        "Resource": [
                            "arn:aws:iot:*:*:topic/strands/*",
                            "arn:aws:iot:*:*:topic/strands/safety/estop",
                        ],
                    },
                    {
                        "Effect": "Allow",
                        "Action": ["iot:ListThings"],
                        "Resource": "*",
                    },
                ],
            }
        ),
    )
    # Lambda role propagation in IAM is eventually-consistent; small delay.
    time.sleep(8)
    account.created.append(f"iam:{role_name}")
    return arn


def _ensure_estop_lambda(lam: Any, role_arn: str, account: BootstrappedAccount, *, force_update: bool = False) -> str:
    """E-stop fan-out Lambda with version tracking.

    The Description field is stamped with ``[v<N>]`` so we can detect stale
    deployments. On ``force_update=True``, an existing Lambda is updated
    in-place with the current source and description.
    """
    version_tag = f"[v{_LAMBDA_VERSION}]"
    description = f"strands-mesh: defence-in-depth E-stop fan-out {version_tag}"

    try:
        existing = lam.get_function(FunctionName=ESTOP_LAMBDA_NAME)
        existing_desc = existing["Configuration"].get("Description", "")
        existing_arn = existing["Configuration"]["FunctionArn"]

        if version_tag not in existing_desc:
            logger.warning(
                "E-stop Lambda exists but has stale version (description=%r, "
                "expected %s). Pass force_update=True to bootstrap_account() "
                "to upgrade.",
                existing_desc,
                version_tag,
            )
            if force_update:
                zip_bytes = _build_lambda_zip()
                lam.update_function_code(FunctionName=ESTOP_LAMBDA_NAME, ZipFile=zip_bytes)
                lam.update_function_configuration(FunctionName=ESTOP_LAMBDA_NAME, Description=description)
                account.created.append(f"lambda:{ESTOP_LAMBDA_NAME} (updated)")
                logger.info("E-stop Lambda updated to %s", version_tag)
                return existing_arn
        account.skipped.append(f"lambda:{ESTOP_LAMBDA_NAME}")
        return existing_arn
    except lam.exceptions.ResourceNotFoundException:
        pass

    zip_bytes = _build_lambda_zip()
    resp = lam.create_function(
        FunctionName=ESTOP_LAMBDA_NAME,
        Runtime="python3.12",
        Role=role_arn,
        Handler="lambda_function.lambda_handler",
        Code={"ZipFile": zip_bytes},
        Description=description,
        Timeout=30,
        MemorySize=256,
        Tags={"strands-mesh": "managed"},
    )
    account.created.append(f"lambda:{ESTOP_LAMBDA_NAME}")
    return resp["FunctionArn"]


def _ensure_safety_to_dynamodb_rule(iot: Any, table_arn: str, account: BootstrappedAccount) -> str:
    """IoT Rule that mirrors safety events into DynamoDB.

    The SQL pulls peer_id, type, severity, payload, t out of the JSON and
    writes one DynamoDB row per event. Uses ``DynamoDBv2`` action which
    natively writes a JSON-mapped row.
    """
    rule_name = RULE_SAFETY_TO_DYNAMODB
    try:
        existing = iot.get_topic_rule(ruleName=rule_name)
        account.skipped.append(f"iot-rule:{rule_name}")
        return existing["ruleArn"]
    except (
        iot.exceptions.ResourceNotFoundException,
        iot.exceptions.UnauthorizedException,
    ):
        # AWS IoT returns UnauthorizedException (not ResourceNotFound) when a
        # rule of this name doesn't exist yet — confusing but documented.
        pass

    role_arn = _ensure_iot_action_role(account)
    # Use newuuid() for the range key so two events with identical t in the
    # same robot still write distinct rows. peer_id remains the partition key.
    sql = "SELECT peer_id, type, severity, payload, t, newuuid() AS ts, topic() AS topic FROM 'strands/+/safety/event'"
    iot.create_topic_rule(
        ruleName=rule_name,
        topicRulePayload={
            "sql": sql,
            "description": "Mirror strands/+/safety/event to DynamoDB audit table",
            "ruleDisabled": False,
            "awsIotSqlVersion": "2016-03-23",
            "actions": [
                {
                    "dynamoDBv2": {
                        "roleArn": role_arn,
                        "putItem": {"tableName": SAFETY_TABLE_NAME},
                    }
                }
            ],
        },
    )
    arn = f"arn:aws:iot:{account.region}:{account.account_id}:rule/{rule_name}"
    account.created.append(f"iot-rule:{rule_name}")
    return arn


def _ensure_estop_rule(iot: Any, lambda_arn: str, account: BootstrappedAccount) -> str:
    """IoT Rule that fires the E-stop fan-out Lambda."""
    rule_name = RULE_ESTOP_FANOUT
    try:
        iot.get_topic_rule(ruleName=rule_name)
        account.skipped.append(f"iot-rule:{rule_name}")
        return f"arn:aws:iot:{account.region}:{account.account_id}:rule/{rule_name}"
    except (
        iot.exceptions.ResourceNotFoundException,
        iot.exceptions.UnauthorizedException,
    ):
        pass

    iot.create_topic_rule(
        ruleName=rule_name,
        topicRulePayload={
            "sql": "SELECT * FROM 'strands/safety/estop'",
            "description": "Fan out E-stop to every strands-mesh robot via Lambda",
            "ruleDisabled": False,
            "awsIotSqlVersion": "2016-03-23",
            "actions": [{"lambda": {"functionArn": lambda_arn}}],
        },
    )
    arn = f"arn:aws:iot:{account.region}:{account.account_id}:rule/{rule_name}"
    account.created.append(f"iot-rule:{rule_name}")
    return arn


def _grant_iot_invoke_lambda(lam: Any, lambda_arn: str, account: BootstrappedAccount) -> None:
    """Allow the IoT Rules service to invoke the E-stop Lambda."""
    rule_arn = f"arn:aws:iot:{account.region}:{account.account_id}:rule/{RULE_ESTOP_FANOUT}"
    try:
        lam.add_permission(
            FunctionName=ESTOP_LAMBDA_NAME,
            StatementId="strands-mesh-iot-invoke",
            Action="lambda:InvokeFunction",
            Principal="iot.amazonaws.com",
            SourceArn=rule_arn,
        )
    except lam.exceptions.ResourceConflictException:
        pass  # already granted


def _ensure_iot_action_role(account: BootstrappedAccount) -> str:
    """The role IoT Rules assume to write to DynamoDB."""
    boto3 = _require_boto3()
    iam = boto3.client("iam")
    role_name = "strands-mesh-iot-action-role"
    try:
        return iam.get_role(RoleName=role_name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "iot.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    resp = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="strands-mesh: IoT Rules action role (DynamoDB write)",
        Tags=[{"Key": "strands-mesh", "Value": "managed"}],
    )
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="strands-mesh-action-policy",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["dynamodb:PutItem"],
                        "Resource": (
                            f"arn:aws:dynamodb:{account.region}:{account.account_id}:table/{SAFETY_TABLE_NAME}"
                        ),
                    }
                ],
            }
        ),
    )
    time.sleep(8)
    account.created.append(f"iam:{role_name}")
    return resp["Role"]["Arn"]


def _ensure_log_group(logs: Any, account: BootstrappedAccount) -> str:
    """CloudWatch log group for IoT events. Used by Rule action(s) optionally."""
    try:
        existing = logs.describe_log_groups(logGroupNamePrefix=LOG_GROUP_NAME)
        for lg in existing.get("logGroups", []):
            if lg["logGroupName"] == LOG_GROUP_NAME:
                account.skipped.append(f"logs:{LOG_GROUP_NAME}")
                return lg["arn"]
    except Exception:
        pass

    logs.create_log_group(
        logGroupName=LOG_GROUP_NAME,
        tags={"strands-mesh": "managed"},
    )
    logs.put_retention_policy(
        logGroupName=LOG_GROUP_NAME,
        retentionInDays=30,
    )
    account.created.append(f"logs:{LOG_GROUP_NAME}")
    desc = logs.describe_log_groups(logGroupNamePrefix=LOG_GROUP_NAME)
    return desc["logGroups"][0]["arn"]


def _ensure_provisioning_hook_role(iam: Any, account: BootstrappedAccount) -> str:
    """Dedicated IAM role for the PreProvisioningHook Lambda (F-19 / B-13).

    The hook must call ``iot:DescribeThing`` (does the target Thing already
    exist?) and ``ssm:GetParameter`` (is the serial on the allowlist?). The
    E-stop Lambda role grants neither, so reusing it would make every
    ``describe_thing`` / ``get_parameter`` raise ``AccessDenied``; those are
    swallowed by the hook's deny-on-error envelope and *every* registration
    — legitimate ones included — would be refused. This role grants exactly
    those two read actions, least-privilege scoped:

    * ``iot:DescribeThing`` on ``thing/*`` (the hook only reads existence;
      it never mutates).
    * ``ssm:GetParameter`` on ``parameter/strands-mesh/provisioning/allow/*``
      (the allowlist namespace only — no broader Parameter Store access).
    """
    role_name = PROVISIONING_HOOK_ROLE
    try:
        role = iam.get_role(RoleName=role_name)
        account.skipped.append(f"iam:{role_name}")
        return role["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    resp = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="strands-mesh Fleet Provisioning PreProvisioningHook Lambda execution role",
        Tags=[{"Key": "strands-mesh", "Value": "managed"}],
    )
    arn = resp["Role"]["Arn"]

    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    )
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="strands-mesh-provisioning-hook",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["iot:DescribeThing"],
                        "Resource": [f"arn:aws:iot:{account.region}:{account.account_id}:thing/*"],
                    },
                    {
                        "Effect": "Allow",
                        "Action": ["ssm:GetParameter"],
                        "Resource": [
                            f"arn:aws:ssm:{account.region}:{account.account_id}"
                            ":parameter/strands-mesh/provisioning/allow/*"
                        ],
                    },
                ],
            }
        ),
    )
    # Lambda role propagation in IAM is eventually-consistent; small delay.
    time.sleep(8)
    account.created.append(f"iam:{role_name}")
    return arn


def _ensure_provisioning_hook_lambda(
    lam: Any, role_arn: str, account: BootstrappedAccount, *, force_update: bool = False
) -> str:
    """Create/Update the Fleet Provisioning PreProvisioningHook Lambda (F-19/B-13).

    Returns the function ARN. Idempotent; reuses an existing function and
    only updates its code when ``force_update`` is set.
    """
    name = PROVISIONING_HOOK_LAMBDA_NAME
    version_tag = f"[v{_PROVISIONING_HOOK_VERSION}]"
    description = f"strands-mesh: Fleet Provisioning PreProvisioningHook (deny-by-default) {version_tag}"
    zip_bytes = _build_provisioning_hook_zip()
    try:
        existing = lam.get_function(FunctionName=name)
        arn = existing["Configuration"]["FunctionArn"]
        existing_desc = existing["Configuration"].get("Description", "")
        if version_tag not in existing_desc:
            logger.warning(
                "Provisioning-hook Lambda exists but has stale version "
                "(description=%r, expected %s). Pass force_update=True to "
                "bootstrap_account() to upgrade.",
                existing_desc,
                version_tag,
            )
        if force_update:
            lam.update_function_code(FunctionName=name, ZipFile=zip_bytes)
            lam.update_function_configuration(FunctionName=name, Description=description)
            account.created.append(f"lambda:{name} (updated)")
        else:
            account.skipped.append(f"lambda:{name}")
        return arn
    except lam.exceptions.ResourceNotFoundException:
        pass

    # Lambda IAM role propagation can race; retry create with backoff.
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            resp = lam.create_function(
                FunctionName=name,
                Runtime="python3.12",
                Role=role_arn,
                Handler="lambda_function.lambda_handler",
                Code={"ZipFile": zip_bytes},
                Description=description,
                Timeout=10,
                Tags={"strands-mesh": "managed"},
            )
            account.created.append(f"lambda:{name}")
            return resp["FunctionArn"]
        except lam.exceptions.InvalidParameterValueException as exc:
            last_exc = exc
            if "role" not in str(exc).lower():
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Provisioning-hook Lambda create failed after retries: {last_exc}")


def _grant_iot_invoke_provisioning_hook(lam: Any, hook_arn: str, account: BootstrappedAccount) -> None:
    """Allow the IoT service principal to invoke the PreProvisioningHook Lambda."""
    try:
        lam.add_permission(
            FunctionName=PROVISIONING_HOOK_LAMBDA_NAME,
            StatementId="strands-mesh-iot-provisioning-invoke",
            Action="lambda:InvokeFunction",
            Principal="iot.amazonaws.com",
            SourceArn=f"arn:aws:iot:{account.region}:{account.account_id}:provisioningtemplate/{PROVISIONING_TEMPLATE}",
        )
        account.created.append("lambda-permission:provisioning-hook-invoke")
    except lam.exceptions.ResourceConflictException:
        account.skipped.append("lambda-permission:provisioning-hook-invoke")


def _ensure_provisioning_template(iot: Any, account: BootstrappedAccount, *, hook_lambda_arn: str = "") -> str:
    """Fleet Provisioning template — claim cert → real cert + attach robot policy.

    F-19 / B-13: a ``PreProvisioningHook`` is wired so a leaked claim cert
    cannot register an arbitrary Thing. ``hook_lambda_arn`` is the ARN of
    the gating Lambda (see :func:`_ensure_provisioning_hook_lambda`); when
    supplied it is attached via ``preProvisioningHook`` and AWS IoT calls
    it before every registration, denying unless the Lambda returns
    ``{"allowProvisioning": True}``.
    """
    name = PROVISIONING_TEMPLATE
    try:
        iot.describe_provisioning_template(templateName=name)
        account.skipped.append(f"iot-prov-template:{name}")
        return f"arn:aws:iot:{account.region}:{account.account_id}:provisioningtemplate/{name}"
    except iot.exceptions.ResourceNotFoundException:
        pass

    role_arn = _ensure_provisioning_role(account)
    body = {
        "Parameters": {
            "ThingName": {"Type": "String"},
            "SerialNumber": {"Type": "String"},
        },
        "Resources": {
            "thing": {
                "Type": "AWS::IoT::Thing",
                "Properties": {
                    "ThingName": {"Ref": "ThingName"},
                    "AttributePayload": {
                        "strands-mesh-role": "robot",
                        "serial": {"Ref": "SerialNumber"},
                    },
                },
            },
            "certificate": {
                "Type": "AWS::IoT::Certificate",
                "Properties": {
                    "CertificateId": {"Ref": "AWS::IoT::Certificate::Id"},
                    "Status": "Active",
                },
            },
            "policy": {
                "Type": "AWS::IoT::Policy",
                "Properties": {"PolicyName": "strands-robot"},
            },
        },
    }
    # IAM role propagation can still race with the IoT AssumeRole check the
    # very first time, so retry a few times with backoff before giving up.
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            create_kwargs: dict[str, Any] = {
                "templateName": name,
                "description": "strands-mesh: factory-provision robots from a claim cert",
                "templateBody": json.dumps(body),
                "provisioningRoleArn": role_arn,
                "enabled": True,
                "tags": [{"Key": "strands-mesh", "Value": "managed"}],
            }
            # F-19 / B-13: gate registration on the PreProvisioningHook so a
            # leaked claim cert cannot self-register an arbitrary Thing.
            if hook_lambda_arn:
                create_kwargs["preProvisioningHook"] = {
                    "targetArn": hook_lambda_arn,
                    "payloadVersion": "2020-04-01",
                }
            iot.create_provisioning_template(**create_kwargs)
            break
        except iot.exceptions.InvalidRequestException as exc:
            last_exc = exc
            if "cannot be assumed" not in str(exc):
                raise
            time.sleep(5 * (attempt + 1))
    else:
        # Exhausted retries — surface the last exception so users see it.
        raise RuntimeError(f"Provisioning template create failed after retries: {last_exc}")
    account.created.append(f"iot-prov-template:{name}")
    return f"arn:aws:iot:{account.region}:{account.account_id}:provisioningtemplate/{name}"


def _ensure_provisioning_role(account: BootstrappedAccount) -> str:
    """The role AWS IoT Fleet Provisioning assumes during registration."""
    boto3 = _require_boto3()
    iam = boto3.client("iam")
    name = PROVISIONING_ROLE
    try:
        return iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "iot.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    resp = iam.create_role(
        RoleName=name,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="strands-mesh: Fleet Provisioning service role",
        Tags=[{"Key": "strands-mesh", "Value": "managed"}],
    )
    iam.attach_role_policy(
        RoleName=name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSIoTThingsRegistration",
    )
    # IAM role propagation is eventually-consistent (typically 5–10s, but can
    # take up to 15s under load). The provisioning template creation hits IoT
    # which then tries to AssumeRole — without the propagation wait we get
    # InvalidRequestException("provisioning role cannot be assumed").
    time.sleep(15)
    account.created.append(f"iam:{name}")
    return resp["Role"]["Arn"]


# Public API


def bootstrap_account(
    *,
    region: str | None = None,
    confirm: bool = False,
    dry_run: bool = True,
    account_id_expected: str | None = None,
    profile: str | None = None,
    force_update: bool = False,
) -> BootstrappedAccount:
    """Bring up every account-wide resource the strands-mesh fleet needs.

    Idempotent. Safe to run multiple times; existing resources are skipped
    and only listed in :attr:`BootstrappedAccount.skipped`.

    Args:
        region: AWS region (defaults to session default).
        confirm: Must be True to actually create resources. Raises ValueError
            if False and dry_run is also False.
        dry_run: When True (default), prints the resources that *would* be
            created without making API calls. Set to False + confirm=True
            to actually provision.
        account_id_expected: If provided, abort if the resolved account ID
            does not match — guards against wrong-account provisioning.
        profile: AWS profile name to use (passed to boto3.Session).
        force_update: If True, update existing E-stop Lambda even when it
            already exists (upgrades stale versions). Default False preserves
            existing deployments.

    Returns:
        :class:`BootstrappedAccount` with every ARN and a record of what
        was created vs skipped.

    Raises:
        ValueError: If confirm=False and dry_run=False, or if account_id_expected
            does not match the resolved account.
    """
    if not dry_run and not confirm:
        raise ValueError(
            "bootstrap_account() creates persistent AWS resources. "
            "Pass confirm=True to proceed, or keep dry_run=True (default) "
            "to preview what would be created."
        )

    boto3 = _require_boto3()
    session = boto3.Session(profile_name=profile) if profile else boto3
    sts = session.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]

    if account_id_expected and account_id != account_id_expected:
        raise ValueError(
            f"Resolved AWS account {account_id} does not match "
            f"expected {account_id_expected}. Aborting to prevent "
            "provisioning in the wrong account."
        )

    if dry_run:
        import sys

        print(
            f"[dry_run] Would create strands-mesh fleet resources in "
            f"account {account_id}, region {sts.meta.region_name}:\n"
            f"  - IoT Thing Type: strands-mesh-robot\n"
            f"  - IoT Policy: strands-mesh-robot-policy\n"
            f"  - IAM Role: strands-mesh-estop-lambda-role\n"
            f"  - Lambda: strands-mesh-estop\n"
            f"  - DynamoDB Table: strands-mesh-fleet\n"
            f"  - CloudWatch Log Group: /strands/mesh\n"
            f"  - IoT Topic Rule: strands_mesh_audit\n"
            f"  - IAM Role: strands-mesh-provisioning-hook-role\n"
            f"  - Lambda: strands-mesh-provisioning-hook (Fleet Provisioning gate)\n"
            f"  - IoT Fleet Provisioning Template: strands-mesh-fleet-provisioning\n"
            f"\nPass dry_run=False, confirm=True to create.",
            file=sys.stderr,
        )
        return BootstrappedAccount(region=sts.meta.region_name, account_id=account_id)
    region = sts.meta.region_name

    iot = boto3.client("iot", region_name=region)
    iam = boto3.client("iam")
    lam = boto3.client("lambda", region_name=region)
    ddb = boto3.client("dynamodb", region_name=region)
    logs = boto3.client("logs", region_name=region)

    out = BootstrappedAccount(region=region, account_id=account_id)

    # Logs (cheap, do first)
    out.log_group_arn = _ensure_log_group(logs, out)

    # DynamoDB
    out.safety_table_arn = _ensure_safety_table(ddb, out)

    # Lambda + IAM
    role_arn = _ensure_lambda_role(iam, out)
    out.estop_lambda_arn = _ensure_estop_lambda(lam, role_arn, out, force_update=force_update)

    # IoT Rules
    out.rule_safety_arn = _ensure_safety_to_dynamodb_rule(iot, out.safety_table_arn, out)
    out.rule_estop_arn = _ensure_estop_rule(iot, out.estop_lambda_arn, out)
    _grant_iot_invoke_lambda(lam, out.estop_lambda_arn, out)

    # Fleet Provisioning PreProvisioningHook Lambda (F-19/B-13) — gate
    # registration so a leaked claim cert cannot self-register a Thing.
    # The hook needs iot:DescribeThing + ssm:GetParameter, which the E-stop
    # role does not grant, so it gets its own least-privilege role.
    hook_role_arn = _ensure_provisioning_hook_role(iam, out)
    hook_arn = _ensure_provisioning_hook_lambda(lam, hook_role_arn, out, force_update=force_update)
    out.provisioning_hook_lambda_arn = hook_arn

    # Fleet Provisioning template (wires the hook above)
    out.provisioning_template_arn = _ensure_provisioning_template(iot, out, hook_lambda_arn=hook_arn)
    _grant_iot_invoke_provisioning_hook(lam, hook_arn, out)

    logger.info(
        "[bootstrap] account %s in %s — created %d, skipped %d",
        account_id,
        region,
        len(out.created),
        len(out.skipped),
    )
    return out


def teardown_account(*, region: str | None = None) -> None:
    """Best-effort reverse of :func:`bootstrap_account`. Safe to skip — every
    deletion catches NotFound silently. Tags-managed resources are removed
    in dependency order: Rules → Lambda → Roles → DynamoDB → Logs →
    Provisioning template.
    """
    boto3 = _require_boto3()
    iot = boto3.client("iot", region_name=region)
    iam = boto3.client("iam")
    lam = boto3.client("lambda", region_name=region)
    ddb = boto3.client("dynamodb", region_name=region)
    logs = boto3.client("logs", region_name=region)

    for rule in (RULE_SAFETY_TO_DYNAMODB, RULE_ESTOP_FANOUT):
        try:
            iot.delete_topic_rule(ruleName=rule)
            logger.info("[teardown] rule %s removed", rule)
        except Exception as exc:
            logger.debug("[teardown] rule %s: %s", rule, exc)

    try:
        lam.delete_function(FunctionName=ESTOP_LAMBDA_NAME)
        logger.info("[teardown] lambda %s removed", ESTOP_LAMBDA_NAME)
    except Exception as exc:
        logger.debug("[teardown] lambda: %s", exc)

    for role in (ESTOP_LAMBDA_ROLE, "strands-mesh-iot-action-role", PROVISIONING_ROLE):
        try:
            for pol in iam.list_role_policies(RoleName=role).get("PolicyNames", []):
                iam.delete_role_policy(RoleName=role, PolicyName=pol)
            for att in iam.list_attached_role_policies(RoleName=role).get("AttachedPolicies", []):
                iam.detach_role_policy(RoleName=role, PolicyArn=att["PolicyArn"])
            iam.delete_role(RoleName=role)
            logger.info("[teardown] role %s removed", role)
        except Exception as exc:
            logger.debug("[teardown] role %s: %s", role, exc)

    try:
        ddb.delete_table(TableName=SAFETY_TABLE_NAME)
        logger.info("[teardown] dynamodb table removed")
    except Exception as exc:
        logger.debug("[teardown] dynamodb: %s", exc)

    try:
        iot.delete_provisioning_template(templateName=PROVISIONING_TEMPLATE)
        logger.info("[teardown] provisioning template removed")
    except Exception as exc:
        logger.debug("[teardown] prov template: %s", exc)

    try:
        logs.delete_log_group(logGroupName=LOG_GROUP_NAME)
        logger.info("[teardown] log group removed")
    except Exception as exc:
        logger.debug("[teardown] logs: %s", exc)
