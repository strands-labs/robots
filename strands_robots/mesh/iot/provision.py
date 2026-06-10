"""One-command AWS IoT provisioning for strands-robots.

This module is the magical out-of-box experience: a developer with AWS
credentials runs::

    from strands_robots.mesh.iot import provision_robot
    provision_robot("so100-arm-01")

...and the function:

1. Creates an AWS IoT Thing named ``so100-arm-01``.
2. Generates an X.509 keypair + cert (AWS-issued, ``CreateKeysAndCertificate``).
3. Creates the canonical strands-robot IoT Policy if it doesn't exist (idempotent).
4. Attaches policy → cert → Thing.
5. Writes ``cert.pem`` / ``private.key`` / ``AmazonRootCA1.pem`` to
   ``~/.strands_robots/iot/`` with mode 0o600.
6. Discovers the IoT data endpoint and writes it to ``~/.strands_robots/iot/endpoint``.

After provisioning, the next ``Robot("so100", peer_id="so100-arm-01")`` call
with ``STRANDS_MESH_BACKEND=iot`` joins the AWS IoT mesh transparently.

All operations are idempotent: re-running ``provision_robot("so100-arm-01")``
re-uses the existing Thing and policy if they're there. A new cert is created
each time (you can't list private keys after the fact, so re-running
generates fresh credentials and keeps the file naming stable).

Operator provisioning
---------------------
:func:`provision_operator` is the analogue for fleet operators (Bedrock
agents, ops consoles). The two policies differ — robots can publish to
their own topic prefix and respond to any operator; operators can publish
``cmd`` / ``broadcast`` and observe the whole fleet.

CLI
---
The same logic is exposed as a CLI entry point (registered in
``pyproject.toml``)::

    strands-robots iot provision so100-arm-01
    strands-robots iot provision-operator bedrock-agent-01
    strands-robots iot teardown so100-arm-01  # cleanup
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_AMAZON_ROOT_CA1_URL = "https://www.amazontrust.com/repository/AmazonRootCA1.pem"

# Pinned SHA-256 fingerprints of the canonical Amazon Root CA1 PEM bytes.
# Pinning prevents a network-level attacker (DNS hijack, captive portal,
# BGP, malicious local proxy) from substituting a rogue CA at the URL.
#
# Recompute when AWS rotates the root::
#
#  python -c "import hashlib, urllib.request as u; \
#  print(hashlib.sha256(u.urlopen( \
#  'https://www.amazontrust.com/repository/AmazonRootCA1.pem' \
#  ).read()).hexdigest())"
#
# Last verified 2026-05.
#
# this is now a TUPLE so a CA rotation can ship as a code change
# that adds the new pin in advance and removes the old one in a follow-
# up after rollout. Operators can also extend the accepted pins via
# ``STRANDS_MESH_CA_PINS`` (comma-separated 64-char lowercase hex). The
# env var augments the built-in tuple; it does not replace it.
_AMAZON_ROOT_CA1_PINS: tuple[str, ...] = ("2c43952ee9e000ff2acc4e2ed0897c0a72ad5fa72c3d934e81741cbd54f05bd1",)
# The legacy ``_AMAZON_ROOT_CA1_SHA256`` alias was deleted.
# CodeQL #229 flagged it as unused after every reader was wired
# through ``_resolve_ca_pins`` / ``_AMAZON_ROOT_CA1_PINS``. Internal
# code references the tuple directly; error messages now format the
# full pin set via ``_resolve_ca_pins`` so operators see every
# accepted pin (not just the canonical first one).

# Regex: 64 hex chars, lowercase. Matches what hashlib.sha256(...).hexdigest()
# emits and rejects anything else (operator typos surface immediately).
_PIN_RE = re.compile(r"^[0-9a-f]{64}$")

# Cap the CA download response to a generous multiple of the real ~1.4 KiB
# certificate. Defeats body-size DoS attacks (a captive portal returning a
# multi-megabyte HTML "login page" instead of the expected PEM).
_CA_FETCH_MAX_BYTES = 64 * 1024

DEFAULT_CERT_DIR = Path.home() / ".strands_robots" / "iot"
ROBOT_POLICY_NAME = "strands-robot"
OPERATOR_POLICY_NAME = "strands-operator"


# Provisioning result


@dataclass
class ProvisionedThing:
    """The artefacts of a single :func:`provision_robot` /
    :func:`provision_operator` call.

    Attributes:
        thing_name: The AWS IoT Thing name (== Mesh peer_id).
        thing_arn: ARN of the Thing.
        cert_arn: ARN of the active certificate.
        cert_id: AWS IoT certificate id (last segment of the ARN).
        cert_path: Local path to the cert PEM (mode 0o600).
        key_path: Local path to the private key (mode 0o600).
        ca_path: Local path to the Amazon Root CA1.
        endpoint: The IoT Data ATS endpoint to connect to.
        policy_name: The policy attached (``strands-robot`` or ``strands-operator``).
        region: The AWS region these resources live in.
    """

    thing_name: str
    thing_arn: str
    cert_arn: str
    cert_id: str
    cert_path: Path
    key_path: Path
    ca_path: Path
    endpoint: str
    policy_name: str
    region: str

    def env_vars(self) -> dict[str, str]:
        """Return env vars a process can export to use these artefacts."""
        return {
            "STRANDS_IOT_THING_NAME": self.thing_name,
            "STRANDS_IOT_ENDPOINT": self.endpoint,
            "STRANDS_IOT_CERT_DIR": str(self.cert_path.parent),
            "STRANDS_MESH_BACKEND": "iot",
        }

    def export_lines(self) -> list[str]:
        """Shell-export lines suitable for ``eval $(...)``."""
        return [f"export {k}={v}" for k, v in self.env_vars().items()]


# Policy documents — verified working in the spike

_ROBOT_POLICY_DOC: dict[str, Any] = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowConnect",
            "Effect": "Allow",
            "Action": "iot:Connect",
            "Resource": "arn:aws:iot:*:*:client/${iot:Connection.Thing.ThingName}",
        },
        {
            "Sid": "AllowOwnTopics",
            "Effect": "Allow",
            "Action": ["iot:Publish", "iot:RetainPublish"],
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/${iot:Connection.Thing.ThingName}/*",
            ],
        },
        {
            # F-15 / B-09: the response topic is
            # ``strands/{operator}/response/{robot_thingname}/{turn}``.
            # The first wildcard is the OPERATOR'S thing-name (the
            # recipient inbox the robot must reach to complete the RPC).
            # The ``${iot:Connection.Thing.ThingName}`` segment pins the
            # RESPONDER to its own identity -- so robot-A can no longer
            # forge a response that claims to come from robot-B. The
            # trailing ``/*`` is the per-turn UUID.
            #
            # Defence-in-depth: the operator-side ACL
            # (``OperatorReceiveResponses`` / ``AllowOwnSubscriptions``)
            # already restricts each operator to its own response prefix;
            # this statement closes the responder-identity surface on the
            # publish side that the pentest (F-15) proved exploitable.
            "Sid": "AllowResponseToAnyOperator",
            "Effect": "Allow",
            "Action": "iot:Publish",
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/*/response/${iot:Connection.Thing.ThingName}/*",
            ],
        },
        {
            "Sid": "AllowSafetyEstop",
            "Effect": "Allow",
            "Action": ["iot:Publish", "iot:RetainPublish"],
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/safety/estop",
            ],
        },
        {
            # Design note: Subscribe is intentionally broader than Receive.
            # ``AllowOwnSubscriptions`` permits a robot to Subscribe to any of
            # its own ``${ThingName}/*`` topics (e.g. health, state,
            # safety/event), but ``AllowReceiveScoped`` below does NOT grant
            # Receive on those. The broker therefore silently drops inbound
            # messages on them. This is deliberate: health/state/safety-event
            # are publish-only at the robot (the operator consumes them), so
            # the robot never needs to Receive its own copy. Do NOT widen
            # ``AllowReceiveScoped`` back to ``${ThingName}/*`` to "fix" this
            # asymmetry -- that re-opens the fleet-eavesdrop surface the narrow
            # Receive list closes. See issue #253 / PR #228 R5.
            "Sid": "AllowOwnSubscriptions",
            "Effect": "Allow",
            "Action": "iot:Subscribe",
            "Resource": [
                "arn:aws:iot:*:*:topicfilter/strands/${iot:Connection.Thing.ThingName}/*",
                "arn:aws:iot:*:*:topicfilter/strands/broadcast",
                "arn:aws:iot:*:*:topicfilter/strands/safety/estop",
                "arn:aws:iot:*:*:topicfilter/strands/+/presence",
            ],
        },
        {
            # Tightly scoped Receive: a robot only sees the messages
            # delivered to topics it actually subscribes to (own /cmd, own
            # /response/*, broadcast, safety/estop, +/presence). Previously
            # this was a wildcard ``iot:Receive`` on ``strands/*``, which
            # would have let any robot eavesdrop on the entire fleet's
            # traffic -- including other robots' commands and responses.
            "Sid": "AllowReceiveScoped",
            "Effect": "Allow",
            "Action": "iot:Receive",
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/${iot:Connection.Thing.ThingName}/cmd",
                "arn:aws:iot:*:*:topic/strands/${iot:Connection.Thing.ThingName}/response/*",
                "arn:aws:iot:*:*:topic/strands/broadcast",
                "arn:aws:iot:*:*:topic/strands/safety/estop",
                "arn:aws:iot:*:*:topic/strands/+/presence",
            ],
        },
        {
            "Sid": "AllowShadow",
            "Effect": "Allow",
            "Action": ["iot:Publish", "iot:Subscribe", "iot:Receive"],
            "Resource": [
                "arn:aws:iot:*:*:topic/$aws/things/${iot:Connection.Thing.ThingName}/shadow/*",
                "arn:aws:iot:*:*:topicfilter/$aws/things/${iot:Connection.Thing.ThingName}/shadow/*",
            ],
        },
    ],
}


_OPERATOR_POLICY_DOC: dict[str, Any] = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "OperatorConnect",
            "Effect": "Allow",
            "Action": "iot:Connect",
            "Resource": "arn:aws:iot:*:*:client/${iot:Connection.Thing.ThingName}",
        },
        {
            # Deliberate wildcard: any operator credential can publish
            # ``cmd`` to any robot (``strands/*/cmd``). The system has
            # no per-operator-to-per-robot binding by design -- the
            # threat model of a compromised operator is equivalent to a
            # compromised fleet command authority. Mitigations: short-
            # lived certs (rotation via ``provision_operator`` re-run),
            # the OperatorShadow attribute condition that gates shadow
            # reads, and operational audit (``mesh_audit.jsonl`` logs
            # every command dispatch). A per-robot operator scope would
            # require a per-robot policy document, which explodes the
            # policy count linearly with fleet size. Pinned as
            # intentional by test_iot_policy_scope.py::TestOperatorPolicy
            # ::test_publish_to_fleet_wildcard_is_deliberate.
            "Sid": "OperatorPublishToFleet",
            "Effect": "Allow",
            "Action": ["iot:Publish", "iot:RetainPublish"],
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/*/cmd",
                "arn:aws:iot:*:*:topic/strands/broadcast",
                "arn:aws:iot:*:*:topic/strands/safety/estop",
            ],
        },
        {
            "Sid": "OperatorReceiveResponses",
            "Effect": "Allow",
            "Action": ["iot:Subscribe", "iot:Receive"],
            "Resource": [
                # F-15: response topic gained a ``{robot_thingname}``
                # segment (strands/{op}/response/{robot}/{turn}); the
                # multi-level ``#`` filter covers both the old and new
                # depth so the operator still receives every response
                # addressed to it.
                "arn:aws:iot:*:*:topic/strands/${iot:Connection.Thing.ThingName}/response/*",
                "arn:aws:iot:*:*:topicfilter/strands/${iot:Connection.Thing.ThingName}/response/#",
            ],
        },
        {
            # Operator monitoring scope. Operators can subscribe to fleet
            # state (presence/state/health) and safety events but NOT to
            # other operators' command/response streams. The policy used to
            # include a wildcard ``strands/*`` in Receive which exposed all
            # fleet traffic to every operator credential.
            "Sid": "OperatorObserveFleet",
            "Effect": "Allow",
            "Action": ["iot:Subscribe", "iot:Receive"],
            "Resource": [
                "arn:aws:iot:*:*:topic/strands/+/presence",
                "arn:aws:iot:*:*:topicfilter/strands/+/presence",
                "arn:aws:iot:*:*:topic/strands/+/state",
                "arn:aws:iot:*:*:topicfilter/strands/+/state",
                "arn:aws:iot:*:*:topic/strands/+/health",
                "arn:aws:iot:*:*:topicfilter/strands/+/health",
                "arn:aws:iot:*:*:topic/strands/+/safety/event",
                "arn:aws:iot:*:*:topicfilter/strands/+/safety/event",
                "arn:aws:iot:*:*:topic/strands/safety/estop",
                "arn:aws:iot:*:*:topicfilter/strands/safety/estop",
            ],
        },
        {
            # F-20 / B-14: scope shadow access to strands-managed Things
            # only. The bare ``thing/*`` resource let an operator cert
            # GetThingShadow / UpdateThingShadow on EVERY Thing in the
            # account (the attribute Condition does not restrict shadow
            # APIs by resource name -- the pentest wrote shadow.reported
            # on non-strands Things). AWS does not apply the attribute
            # condition to the shadow data-plane resource, so the practical
            # fix is a resource-name prefix: strands robots are provisioned
            # with ``strands-`` ThingName prefixes (see PROVISIONING
            # template + _validate_thing_name).
            "Sid": "OperatorShadow",
            "Effect": "Allow",
            "Action": ["iot:GetThingShadow", "iot:UpdateThingShadow"],
            "Resource": ["arn:aws:iot:*:*:thing/strands-*"],
            "Condition": {
                "StringEquals": {
                    "iot:Connection.Thing.Attributes.strands-mesh-role": "robot",
                },
            },
        },
    ],
}


# Public API


# Thing names flow into S3 keys, IoT ARNs, and on-disk cert filenames.
# Our regex is a STRICT SUBSET of AWS IoT's accepted charset (AWS allows
# ``:`` per the AWS IoT docs; we deliberately reject it because ``:`` is
# a stream separator on NTFS and a directory separator on classic Mac,
# and our cert files are written to ``cert_dir / f"{thing_name}.pem"``).
# Operators who use ``:`` in pre-existing AWS IoT Thing names need to
# rename the Thing or maintain a mapping; we choose the safer subset
# here over compatibility with every legal AWS Thing name.
_THING_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _validate_thing_name(thing_name: str) -> None:
    """Raise :class:`ValueError` when *thing_name* is unsafe for use as a
    filesystem component AND as an AWS IoT Thing name.

    The accepted pattern is ``^[a-zA-Z0-9_-]{1,128}$``: alphanumerics,
    dash and underscore, length 1-128.  This is a **strict subset** of
    AWS IoT's accepted Thing-name charset (AWS allows ``:`` server-side;
    we reject it because of NTFS / classic Mac filesystem semantics).
    Anything else (slashes, colons, dots, spaces, NUL, non-ASCII,...)
    is rejected -- a slip in upstream validation can never produce a
    path such as ``../../../etc/foo`` reaching
    ``cert_dir / f"{thing_name}.pem"``.

    Operators with pre-existing AWS IoT Things whose names contain
    ``:`` will hit a ``ValueError`` here. Rename the Thing or maintain
    an external mapping; the strict charset is intentional.
    """
    if not isinstance(thing_name, str) or not thing_name:
        raise ValueError(f"thing_name must be a non-empty string, got {thing_name!r}")
    if not _THING_NAME_RE.fullmatch(thing_name):
        raise ValueError(
            f"thing_name={thing_name!r} contains invalid characters; "
            "allowed: ASCII letters, digits, '-', '_'; max 128 chars."
        )


def provision_robot(
    thing_name: str,
    *,
    region: str | None = None,
    cert_dir: Path | str | None = None,
    attributes: dict[str, str] | None = None,
) -> ProvisionedThing:
    """Provision a robot Thing and write its credentials to disk.

    Validates *thing_name* against ``^[a-zA-Z0-9_-]{1,128}$`` before any
    AWS call. The pattern is a **strict subset** of AWS IoT's accepted
    Thing-name charset (AWS server-side accepts ``:`` as well; we reject
    it for filesystem-path safety on NTFS / classic Mac where ``:`` is a
    stream / directory separator). Operators with pre-existing AWS IoT
    Things containing ``:`` must rename or maintain a mapping; the
    error message will direct them here.

    Args:
        thing_name: The Thing name. MUST equal the intended Mesh peer_id —
            the IoT Policy uses ``${iot:Connection.Thing.ThingName}`` for
            topic ACL substitution. Should be DNS-safe (alphanumeric + ``-_``).
        region: AWS region. Defaults to the default boto3 session region.
        cert_dir: Where to write certs. Defaults to ``~/.strands_robots/iot``.
        attributes: Optional thing-attribute dict (<=3 keys, <=800 chars total).

    Returns:
        :class:`ProvisionedThing` describing the artefacts.

    Raises:
        ImportError: If ``boto3`` is not installed.
        botocore.exceptions.ClientError: For AWS-side failures (auth, throttling).

    Idempotence:
        - Thing creation: ``CreateThing`` is idempotent if the attributes match.
        - Policy creation: skipped if the policy name already exists.
        - Cert creation: a new cert is always issued (private keys aren't
          recoverable). Old certs from prior runs remain on the Thing —
          call :func:`teardown_thing` to clean them up.
    """

    _validate_thing_name(thing_name)
    boto3 = _require_boto3()
    iot = boto3.client("iot", region_name=region)
    region = iot.meta.region_name

    # Inject strands-mesh-role attribute for ACL — the OperatorShadow policy
    # uses an attribute condition to scope shadow access to robot Things only.
    attributes = dict(attributes) if attributes else {}
    attributes.setdefault("strands-mesh-role", "robot")

    cert_dir = Path(cert_dir) if cert_dir else DEFAULT_CERT_DIR
    cert_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(cert_dir, 0o700)
    except OSError:
        pass

    # 1. Thing
    thing_arn = _ensure_thing(iot, thing_name, attributes)

    # 2. Policy
    policy_arn = _ensure_policy(iot, ROBOT_POLICY_NAME, _ROBOT_POLICY_DOC)
    logger.info("[provision] %s: using policy %s", thing_name, policy_arn)

    # 3. Cert + key
    # Clean up stale certs from prior provision_robot runs on the same Thing.
    # Each call to AWS IoT CreateKeysAndCertificate yields a brand-new cert
    # (private keys cannot be recovered after issuance), so without cleanup
    # the Thing would accumulate certs across re-runs — every leftover is
    # an active credential that could impersonate the robot.
    _cleanup_stale_certs(iot, thing_name)

    cert_path = cert_dir / f"{thing_name}.cert.pem"
    key_path = cert_dir / f"{thing_name}.private.key"
    cert_arn, cert_id = _create_cert(iot, cert_path, key_path)

    # 4. Attach policy → cert → thing
    iot.attach_policy(policyName=ROBOT_POLICY_NAME, target=cert_arn)
    iot.attach_thing_principal(thingName=thing_name, principal=cert_arn)
    logger.info("[provision] %s: cert %s attached", thing_name, cert_id)

    # 5. CA + endpoint
    ca_path = cert_dir / "AmazonRootCA1.pem"
    _ensure_ca(ca_path)
    endpoint = _discover_endpoint(iot)
    (cert_dir / "endpoint").write_text(endpoint)

    return ProvisionedThing(
        thing_name=thing_name,
        thing_arn=thing_arn,
        cert_arn=cert_arn,
        cert_id=cert_id,
        cert_path=cert_path,
        key_path=key_path,
        ca_path=ca_path,
        endpoint=endpoint,
        policy_name=ROBOT_POLICY_NAME,
        region=region,
    )


def provision_operator(
    thing_name: str,
    *,
    region: str | None = None,
    cert_dir: Path | str | None = None,
    attributes: dict[str, str] | None = None,
) -> ProvisionedThing:
    """Provision an operator Thing (Bedrock agent / fleet ops console).

    Same as :func:`provision_robot` but with the operator policy
    (``strands-operator``) which can publish ``cmd`` / ``broadcast`` and
    observe the whole fleet.
    """

    _validate_thing_name(thing_name)
    boto3 = _require_boto3()
    iot = boto3.client("iot", region_name=region)
    region = iot.meta.region_name

    # Inject strands-mesh-role attribute — operators get role=operator so the
    # OperatorShadow attribute condition (role=robot) excludes their shadows.
    attributes = dict(attributes) if attributes else {}
    attributes.setdefault("strands-mesh-role", "operator")

    cert_dir = Path(cert_dir) if cert_dir else DEFAULT_CERT_DIR
    cert_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(cert_dir, 0o700)
    except OSError:
        pass

    thing_arn = _ensure_thing(iot, thing_name, attributes)
    policy_arn = _ensure_policy(iot, OPERATOR_POLICY_NAME, _OPERATOR_POLICY_DOC)
    logger.info("[provision] %s: using policy %s", thing_name, policy_arn)

    # Clean up stale certs from prior provision_operator runs.
    _cleanup_stale_certs(iot, thing_name)

    cert_path = cert_dir / f"{thing_name}.cert.pem"
    key_path = cert_dir / f"{thing_name}.private.key"
    cert_arn, cert_id = _create_cert(iot, cert_path, key_path)

    iot.attach_policy(policyName=OPERATOR_POLICY_NAME, target=cert_arn)
    iot.attach_thing_principal(thingName=thing_name, principal=cert_arn)

    ca_path = cert_dir / "AmazonRootCA1.pem"
    _ensure_ca(ca_path)
    endpoint = _discover_endpoint(iot)
    (cert_dir / "endpoint").write_text(endpoint)

    return ProvisionedThing(
        thing_name=thing_name,
        thing_arn=thing_arn,
        cert_arn=cert_arn,
        cert_id=cert_id,
        cert_path=cert_path,
        key_path=key_path,
        ca_path=ca_path,
        endpoint=endpoint,
        policy_name=OPERATOR_POLICY_NAME,
        region=region,
    )


def teardown_thing(
    thing_name: str,
    *,
    region: str | None = None,
    cert_dir: Path | str | None = None,
) -> None:
    """Detach + delete every cert attached to *thing_name*, then delete the Thing.

    Cleans up the cert files under *cert_dir* (defaults to
    :data:`DEFAULT_CERT_DIR`) if they're named after this Thing.  Pass the
    same ``cert_dir`` you used at provision time so the on-disk cert and key
    are removed instead of orphaned.  Does NOT delete the policies — those
    are shared across all robots and removing them would break siblings.

    Idempotent: missing Thing or no certs is a silent success.

    Note:
        ``cert_dir`` is treated as trusted operator input -- it is not
        validated beyond ``Path()`` coercion.  Do not pass LLM-generated
        or otherwise untrusted values; this is a privileged provisioning API.
    """
    _validate_thing_name(thing_name)
    boto3 = _require_boto3()
    iot = boto3.client("iot", region_name=region)

    # Paginate principals: ``list_thing_principals`` returns up to 8
    # principals per call (AWS IoT default page size); a Thing with more
    # than 8 attached certs (rare but possible after multiple
    # provision_robot calls) would otherwise leave certs orphaned.
    # Fallback to single-call when the client doesn't expose
    # ``get_paginator`` (test mocks, custom shims).
    principals: list[str] = []
    try:
        if hasattr(iot, "get_paginator"):
            paginator = iot.get_paginator("list_thing_principals")
            for page in paginator.paginate(thingName=thing_name):
                principals.extend(page.get("principals", []))
        else:
            principals = list(iot.list_thing_principals(thingName=thing_name).get("principals", []))
    except iot.exceptions.ResourceNotFoundException:
        logger.info("[teardown] thing %s not found, skipping", thing_name)
        principals = []

    for cert_arn in principals:
        cert_id = cert_arn.rsplit("/", 1)[-1]
        try:
            iot.detach_thing_principal(thingName=thing_name, principal=cert_arn)
        except Exception as exc:  # noqa: BLE001 -- iot.exceptions.ClientError / BotoCoreError; teardown is idempotent best-effort
            logger.debug("[teardown] detach %s from %s: %s", cert_id, thing_name, exc)
        # Detach all attached policies first
        try:
            for pol in iot.list_attached_policies(target=cert_arn).get("policies", []):
                iot.detach_policy(policyName=pol["policyName"], target=cert_arn)
        except Exception as exc:  # noqa: BLE001 -- iot.exceptions.ClientError / BotoCoreError; teardown is idempotent best-effort
            logger.debug("[teardown] detach policies from %s: %s", cert_id, exc)
        try:
            iot.update_certificate(certificateId=cert_id, newStatus="INACTIVE")
            iot.delete_certificate(certificateId=cert_id, forceDelete=True)
        except Exception as exc:  # noqa: BLE001 -- iot.exceptions.ClientError / BotoCoreError; teardown is idempotent best-effort
            logger.warning("[teardown] could not delete cert %s: %s", cert_id, exc)

    # Delete the Thing
    try:
        iot.delete_thing(thingName=thing_name)
        logger.info("[teardown] deleted thing %s", thing_name)
    except iot.exceptions.ResourceNotFoundException:
        pass

    # Remove local cert files.  Honour a custom ``cert_dir`` so we don't
    # orphan certs provisioned with ``provision_robot(..., cert_dir=...)``.
    # ``_create_cert`` only writes ``.cert.pem`` and ``.private.key`` -- a
    # ``.public.key`` suffix was dead code and is intentionally dropped.
    target_cert_dir = Path(cert_dir) if cert_dir else DEFAULT_CERT_DIR
    for suffix in (".cert.pem", ".private.key"):
        p = target_cert_dir / f"{thing_name}{suffix}"
        if p.exists():
            try:
                p.unlink()
            except OSError as exc:
                logger.debug("[teardown] could not unlink %s: %s", p, exc)


# Internals


def _require_boto3() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for AWS IoT provisioning. Install with: pip install 'strands-robots[mesh-iot]'"
        ) from exc
    return boto3


def _ensure_thing(iot: Any, thing_name: str, attributes: dict[str, str] | None) -> str:
    """Create the Thing if absent, otherwise return its ARN unchanged."""
    try:
        existing = iot.describe_thing(thingName=thing_name)
        logger.info("[provision] thing %s already exists", thing_name)
        return existing["thingArn"]
    except iot.exceptions.ResourceNotFoundException:
        pass

    payload: dict[str, Any] = {"thingName": thing_name}
    if attributes:
        payload["attributePayload"] = {"attributes": attributes}
    resp = iot.create_thing(**payload)
    logger.info("[provision] thing %s created", thing_name)
    return resp["thingArn"]


def _ensure_policy(iot: Any, name: str, document: dict[str, Any]) -> str:
    """Create the policy if absent. Idempotent — does not update an existing
    policy; users who want to update should bump the policy version manually."""
    try:
        existing = iot.get_policy(policyName=name)
        logger.info("[provision] policy %s already exists (v%s)", name, existing.get("defaultVersionId", "?"))
        return existing["policyArn"]
    except iot.exceptions.ResourceNotFoundException:
        pass

    resp = iot.create_policy(
        policyName=name,
        policyDocument=json.dumps(document),
    )
    logger.info("[provision] policy %s created", name)
    return resp["policyArn"]


def _cleanup_stale_certs(iot: Any, thing_name: str) -> int:
    """Detach + delete any certificates already attached to *thing_name*.

    Re-running :func:`provision_robot` on the same Thing has historically
    caused certs to accumulate (each run issues a fresh cert because
    AWS doesn't expose previously-generated private keys). That left
    Things with 5–10 ACTIVE certs after a few dev iterations, which is
    a footgun: every old cert is a credential that *could* be used to
    impersonate the robot.

    This helper detaches every existing principal, removes its policy
    attachments, marks the cert INACTIVE, and force-deletes it. Failures
    are logged at DEBUG and swallowed so a partial cleanup never blocks
    the new cert issuance — the new cert is what users actually want.

    Returns the number of certs cleaned up (for logging in the caller).
    """
    cleaned = 0
    try:
        existing = iot.list_thing_principals(thingName=thing_name).get("principals", [])
    except iot.exceptions.ResourceNotFoundException:
        return 0

    for cert_arn in existing:
        cert_id = cert_arn.rsplit("/", 1)[-1]
        try:
            iot.detach_thing_principal(thingName=thing_name, principal=cert_arn)
        except Exception as exc:
            logger.debug("[provision] detach %s from %s: %s", cert_id, thing_name, exc)
        try:
            for pol in iot.list_attached_policies(target=cert_arn).get("policies", []):
                iot.detach_policy(policyName=pol["policyName"], target=cert_arn)
        except Exception as exc:
            logger.debug("[provision] detach policies from %s: %s", cert_id, exc)
        try:
            iot.update_certificate(certificateId=cert_id, newStatus="INACTIVE")
            iot.delete_certificate(certificateId=cert_id, forceDelete=True)
            cleaned += 1
        except Exception as exc:
            logger.warning("[provision] could not delete stale cert %s: %s", cert_id, exc)
    if cleaned:
        logger.info(
            "[provision] cleaned up %d stale cert(s) on %s before issuing new one",
            cleaned,
            thing_name,
        )
    return cleaned


def _create_cert(iot: Any, cert_path: Path, key_path: Path) -> tuple[str, str]:
    """Issue a fresh cert+key and write them to disk with mode 0o600."""
    resp = iot.create_keys_and_certificate(setAsActive=True)
    cert_arn = resp["certificateArn"]
    cert_id = resp["certificateId"]

    cert_path.write_text(resp["certificatePem"])
    key_path.write_text(resp["keyPair"]["PrivateKey"])
    try:
        os.chmod(cert_path, 0o600)
        os.chmod(key_path, 0o600)
    except OSError as exc:
        logger.warning("[provision] could not chmod certs: %s", exc)
    return cert_arn, cert_id


# Issue #261: one-WARN-per-process gate for unverified-origin CA re-use.
_UNVERIFIED_CA_WARNED: set[Path] = set()


def _ensure_ca(ca_path: Path) -> None:
    """Ensure a verified copy of Amazon Root CA1 lives at *ca_path*.

    Behaviour:

    * If *ca_path* already exists, re-check its bytes against the pinned
      SHA-256. A mismatch raises :class:`RuntimeError` and leaves the file
      untouched -- the caller decides whether to delete and retry.
    * Otherwise download the CA over HTTPS, cap the body at
      :data:`_CA_FETCH_MAX_BYTES`, verify the pin, and write the result
      with mode ``0o644``.

    Pinning defeats a network-level adversary (DNS hijack, captive portal,
    BGP route attacks, malicious corporate proxy) that could substitute a
    rogue CA at the canonical URL.

    Break glass: setting ``STRANDS_MESH_DISABLE_CA_PIN=true`` skips the pin
    check. A WARNING is logged on every disabled run. This exists for
    proxy environments that legitimately re-encode the certificate; it
    should never be set in production.
    """
    if ca_path.exists() and ca_path.stat().st_size > 0:
        # Existing-file branch: ALWAYS perform the raw hash compare,
        # regardless of STRANDS_MESH_DISABLE_CA_PIN. The break-glass
        # exists to allow re-encoding proxies on the *download* path
        # (lines below) -- it must NOT silently re-use a rogue CA from
        # a prior compromised provisioning run. Operators who need
        # to refresh a re-encoded cert can delete the file and let
        # the download path run with the override set.
        # O_NOFOLLOW to prevent TOCTOU symlink-swap
        # Issue #251: chunked-read loop (mirrors verify_ca_pin). Single
        # ``os.read(fd, 10MB)`` returns *up to* the requested byte count
        # so on interrupted syscalls / unusual filesystems the read can
        # return short, in which case ``_hash_matches_pin(existing)``
        # hashes a partial file and rejects (fail-closed, OK) -- but
        # the surrounding error message says "failed pin check" rather
        # than the truthful "short read", which is hostile to the
        # operator triaging the issue. The chunked loop drains the file
        # or hits the 10 MiB cap, matching ``verify_ca_pin`` posture.
        try:
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(ca_path, os.O_RDONLY | nofollow)
            try:
                chunks: list[bytes] = []
                remaining = 10 * 1024 * 1024  # 10 MiB bound
                while remaining > 0:
                    buf = os.read(fd, min(65536, remaining))
                    if not buf:
                        break
                    chunks.append(buf)
                    remaining -= len(buf)
                existing = b"".join(chunks)
            finally:
                os.close(fd)
        except OSError as exc:
            raise RuntimeError(f"AmazonRootCA1 at {ca_path} unreadable or symlink: {exc}") from exc
        if not _hash_matches_pin(existing):
            logger.warning(
                "[provision] existing CA at %s does NOT match pinned SHA-256. "
                "Refusing to use it (STRANDS_MESH_DISABLE_CA_PIN does not "
                "apply to the on-disk re-use path). Delete the file to "
                "force re-download.",
                ca_path,
            )
            accepted = ", ".join(sorted(_resolve_ca_pins()))
            raise RuntimeError(f"AmazonRootCA1 at {ca_path} failed pin check; accepted pins: {accepted}")
        # Issue #261: WARN if this CA was originally downloaded under
        # the STRANDS_MESH_DISABLE_CA_PIN break-glass. The pin check above
        # passed (so the bytes match a known good pin), but the operator
        # should be aware that an unverified-origin CA is being re-used
        # in case they want to refresh it via the canonical (pinned) path.
        # Emit one WARNING per process (gated on a module-level set).
        marker = ca_path.with_suffix(ca_path.suffix + ".unverified")
        if marker.exists() and ca_path not in _UNVERIFIED_CA_WARNED:
            _UNVERIFIED_CA_WARNED.add(ca_path)
            logger.warning(
                "[provision] re-using CA at %s that was originally downloaded "
                "with STRANDS_MESH_DISABLE_CA_PIN=true (sidecar marker %s "
                "exists). The pin check on the on-disk bytes passed, but the "
                "ORIGIN of those bytes was not pin-verified. Delete both files "
                "and re-run without the break-glass to refresh via the canonical path.",
                ca_path,
                marker,
            )
        return

    logger.info("[provision] downloading Amazon Root CA1 -> %s (pinned)", ca_path)
    # per-socket timeout via a custom HTTPSHandler.
    #
    # The previous implementation called ``socket.setdefaulttimeout(15.0)``
    # for the duration of the urlopen and restored it in ``finally``.
    # That is process-global -- every other thread doing socket I/O
    # during the CA download window observes the foreign 15s default
    # (boto3, Zenoh keepalives, requests pools all assume None). The
    # ``urllib.request.build_opener`` path here installs a one-shot
    # ``HTTPSHandler`` whose ``https_open`` builds connections via
    # ``socket.create_connection(timeout=...)`` so the per-recv deadline
    # sticks to that one socket only. No process-global mutation.
    body = _download_with_per_socket_timeout(_AMAZON_ROOT_CA1_URL, 15.0, _CA_FETCH_MAX_BYTES + 1)
    if len(body) > _CA_FETCH_MAX_BYTES:
        raise RuntimeError(f"AmazonRootCA1 download exceeded {_CA_FETCH_MAX_BYTES} bytes -- refusing")

    if not _verify_ca_bytes(body):
        accepted = ", ".join(sorted(_resolve_ca_pins()))
        raise RuntimeError(
            "AmazonRootCA1 SHA-256 mismatch -- refusing to write rogue CA. "
            f"Got {hashlib.sha256(body).hexdigest()}; accepted pins: {accepted}"
        )

    ca_path.write_bytes(body)
    try:
        os.chmod(ca_path, 0o644)
    except OSError:
        pass

    # Issue #261: when the break-glass STRANDS_MESH_DISABLE_CA_PIN was
    # active during this download, write a sidecar marker so future
    # _ensure_ca invocations can WARN about re-using an unverified CA
    # even when the env var is no longer set.
    if os.getenv("STRANDS_MESH_DISABLE_CA_PIN", "").strip().lower() == "true":
        marker = ca_path.with_suffix(ca_path.suffix + ".unverified")
        try:
            marker.write_text(
                "# This CA was downloaded with STRANDS_MESH_DISABLE_CA_PIN=true.\n"
                "# Future _ensure_ca calls on this host will WARN until this\n"
                "# marker is removed (e.g. by deleting the CA + re-running with\n"
                "# the pin enforced).\n"
            )
            # Owner-only: marker is a local sentinel read only by this
            # process via _ensure_ca; no other user needs read access.
            # Tightens CodeQL py/overly-permissive-file-permission alert
            # vs the prior 0o644 default.
            os.chmod(marker, 0o600)
        except OSError:
            # Best-effort marker write: an unwritable cert_dir already
            # surfaced via the preceding write_bytes/write_text path.
            # Failing to chmod the marker should not abort provisioning;
            # the WARN-on-reuse contract is degraded-but-honest.
            logger.debug("[provision] CA-unverified marker chmod failed -- continuing", exc_info=True)


def _resolve_ca_pins() -> frozenset[str]:
    """Return the full set of accepted Amazon Root CA1 SHA-256 pins.

    Combines the built-in :data:`_AMAZON_ROOT_CA1_PINS` tuple with any
    operator-supplied pins from ``STRANDS_MESH_CA_PINS``
    (comma-separated, 64-char lowercase hex; invalid entries are
    rejected with a WARNING and skipped). The env-var path lets a
    fleet operator stage a new pin ahead of a code-level rotation without a flag-day deploy. The built-in tuple is always
    included; the env var is additive only.
    """
    pins = set(_AMAZON_ROOT_CA1_PINS)
    raw = os.getenv("STRANDS_MESH_CA_PINS", "").strip()
    if raw:
        for entry in raw.split(","):
            normalised = entry.strip().lower()
            if not normalised:
                continue
            if not _PIN_RE.fullmatch(normalised):
                logger.warning(
                    "[provision] STRANDS_MESH_CA_PINS entry %r is not a valid 64-char lowercase hex SHA-256; skipping.",
                    entry,
                )
                continue
            pins.add(normalised)
    return frozenset(pins)


def _download_with_per_socket_timeout(url: str, timeout_s: float, max_bytes: int) -> bytes:
    """Download *url* with a per-socket recv timeout -- no process-global mutation.

    ``socket.setdefaulttimeout`` is a process-global. While its
    try/finally restore is correct, every other thread doing socket I/O
    during the urlopen observes the foreign default. We install a one-
    shot ``HTTPSHandler`` whose ``https_open`` constructs HTTPSConnection
    instances with the timeout baked in, so the deadline is per-socket
    and never visible to other code paths.

    Raises ``RuntimeError`` on socket timeout (slow-loris responder /
    hostile proxy) with a message pointing at the break-glass env var.
    """
    import http.client
    import urllib.error

    class _TimedHTTPSHandler(urllib.request.HTTPSHandler):
        """HTTPSHandler whose connection factory bakes in *timeout_s*.

        urllib.request's default HTTPSHandler builds an HTTPSConnection
        without an explicit timeout -- only the urlopen(timeout=) value
        is forwarded, and that argument only covers connect + TLS
        handshake. A per-connection timeout on the HTTPSConnection
        itself propagates to recv() / sendall() and bounds wall-clock
        for the whole transaction.
        """

        def https_open(self, req: urllib.request.Request) -> Any:
            return self.do_open(self._connection_factory, req)

        @staticmethod
        def _connection_factory(host: str, **kwargs: Any) -> http.client.HTTPSConnection:
            kwargs["timeout"] = timeout_s
            return http.client.HTTPSConnection(host, **kwargs)

    opener = urllib.request.build_opener(_TimedHTTPSHandler())
    try:
        with opener.open(url, timeout=timeout_s) as resp:  # noqa: S310 -- HTTPS + pinned
            return resp.read(max_bytes)
    except (TimeoutError, urllib.error.URLError) as exc:
        # urllib wraps socket.timeout in URLError on some Python versions;
        # both surface as a RuntimeError pointing at the break-glass.
        if isinstance(exc, urllib.error.URLError) and not isinstance(exc.reason, TimeoutError):
            raise
        raise RuntimeError(
            "AmazonRootCA1 download timed out -- possible slow-loris "
            "responder or hostile proxy. Set "
            "STRANDS_MESH_DISABLE_CA_PIN=true and retry only after "
            "confirming the network path is trustworthy."
        ) from exc


def _hash_matches_pin(body: bytes) -> bool:
    """Return True iff *body*'s SHA-256 matches any accepted pin.

    Consults the full pin set returned by :func:`_resolve_ca_pins`
    (built-in :data:`_AMAZON_ROOT_CA1_PINS` plus any
    ``STRANDS_MESH_CA_PINS`` entries). Pure check -- does not honour
    ``STRANDS_MESH_DISABLE_CA_PIN`` (that's the contract that makes
    :func:`verify_ca_pin` safe for ops scripts).
    """
    digest = hashlib.sha256(body).hexdigest()
    return digest in _resolve_ca_pins()


def _verify_ca_bytes(body: bytes) -> bool:
    """Return True if *body* may be used as the Amazon Root CA1.

    This is the **provisioning-side** check used by :func:`_ensure_ca`.
    It honours the ``STRANDS_MESH_DISABLE_CA_PIN`` break-glass env var:
    when set, the function returns True for any input and logs a WARNING
    so the override surfaces in routine audits. Operators set the
    override only for proxy environments that legitimately re-encode the
    cert; production deployments must leave it unset.

    Forensic / ops scripts that want ground truth should call
    :func:`verify_ca_pin` instead -- that function never honours the
    break-glass.
    """
    if os.getenv("STRANDS_MESH_DISABLE_CA_PIN", "").strip().lower() == "true":
        logger.warning("[provision] STRANDS_MESH_DISABLE_CA_PIN=true -- CA pin check skipped")
        return True
    return _hash_matches_pin(body)


def verify_ca_pin(ca_path: Path) -> bool:
    """Public helper: does the CA file at *ca_path* match the pinned hash?

    This function NEVER honours the ``STRANDS_MESH_DISABLE_CA_PIN``
    break-glass -- its job is to tell the caller the truth about whether
    the file on disk is the canonical Amazon Root CA1. If it returned
    True under the break-glass an attacker who set the env var on a
    compromised host would defeat exactly the forensic check operators
    rely on.

    mirrors the ``O_NOFOLLOW`` discipline that
    ``_ensure_ca`` uses on the on-disk re-use path. Without it,
    an attacker who can race a symlink between the operator-supplied
    ``ca_path`` and this read can redirect ``read_bytes()`` to a hash-
    matching decoy, defeating exactly this verifier. The asymmetric
    posture (``_ensure_ca`` defends, ``verify_ca_pin`` does not) was
    the actual gap; this closes it.

    Returns False on any I/O error (missing file, permission denied,
    symlinked path, etc.). The caller should treat False as "do not
    trust this CA".
    """
    import os

    try:
        if ca_path.is_symlink():
            logger.warning(
                "[provision] verify_ca_pin: refusing %s -- it is a SYMLINK "
                "(target=%r). CA files must be regular files at the canonical path.",
                ca_path,
                os.readlink(ca_path),
            )
            return False
        flags = os.O_RDONLY
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(ca_path), flags | nofollow)
        try:
            # Bound the read at 1 MiB -- the AWS Root CA1 PEM is < 2 KiB;
            # anything larger is a suspicious file we should not pin against.
            chunks: list[bytes] = []
            remaining = 1 * 1024 * 1024
            while remaining > 0:
                buf = os.read(fd, min(65536, remaining))
                if not buf:
                    break
                chunks.append(buf)
                remaining -= len(buf)
            content = b"".join(chunks)
        finally:
            os.close(fd)
        return _hash_matches_pin(content)
    except OSError:
        return False


def _discover_endpoint(iot: Any) -> str:
    """Return the iot:Data-ATS endpoint for this region+account."""
    return iot.describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"]
