"""The pre-provisioning hook a claim template runs, and the allowlist it reads.

A claim certificate is shared by a whole fleet, so whoever holds it must not choose a robot's name, tenant or type.
The operator approves each robot ahead of its first boot with :func:`allow`, which writes one SSM SecureString
``/strands-robots/claim/<sha256 of the device id>`` = ``{"device_id", "tenant", "device_type", "secret_sha256"}`` and
returns a one-time secret to install on that robot. The hook (``HOOK_SOURCE``, a Lambda the user deploys) accepts a
RegisterThing only when the thing does not exist yet, the certificate's one subject CN is its name, an allowlist entry
exists for it, the request carries that entry's secret, tenant and device type, and then deletes the entry. So a
leaked claim certificate names no robot, joins no other tenant, and cannot take a name before its robot does. The
hook's role needs ``iot:DescribeThing`` and ``ssm:GetParameter`` / ``ssm:DeleteParameter`` on
``/strands-robots/claim/*`` (and ``kms:Decrypt`` for the key that encrypts them); the library creates no IAM.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Collection, Mapping
from typing import Any

from .. import wire

ALLOW_PREFIX = "/strands-robots/claim/"


def allow_name(device_id: str) -> str:
    """The SSM parameter of ``device_id``'s allowlist entry (hashed: SSM names cannot hold every device id)."""
    return ALLOW_PREFIX + hashlib.sha256(wire.check_id(device_id, "device_id").encode()).hexdigest()


def allow(ssm: Any, device_id: str, *, tenant: str, device_type: str) -> str:
    """Approve ``device_id`` to claim its first certificate in ``tenant`` as ``device_type``; returns the one-time
    secret to give that robot (``claim(..., secret=)``). Refuses to overwrite a pending approval."""
    secret = secrets.token_urlsafe(24)
    entry = {"device_id": wire.check_id(device_id, "device_id"), "tenant": wire.check_id(tenant, "tenant"),
             "device_type": wire.check_id(device_type, "device_type"),
             "secret_sha256": hashlib.sha256(secret.encode()).hexdigest()}  # fmt: skip
    ssm.put_parameter(Name=allow_name(device_id), Value=json.dumps(entry), Type="SecureString", Overwrite=False)
    return secret


HOOK_SOURCE = '''\
"""Pre-provisioning hook (python3.12 Lambda, boto3 from the runtime). Allow a RegisterThing only for a thing that does
not exist yet, whose certificate subject has exactly one CN equal to the thing name, that the operator approved
(an SSM entry under /strands-robots/claim/), with that entry's secret, tenant and device type; the entry is then
deleted. The CN is read by walking the DER to the TBS subject Name (never the issuer, never bytes inside another
attribute's value). Any error denies."""
import base64
import hashlib
import hmac
import json

PREFIX = "/strands-robots/claim/"


def _tlv(der, i):
    tag, n = der[i], der[i + 1]
    i += 2
    if n & 0x80:
        k = n & 0x7F
        n, i = int.from_bytes(der[i : i + k], "big"), i + k
    return tag, i, i + n  # tag, start of value, end of value


def _items(der, start, end):
    while start < end:
        tag, v, start = _tlv(der, start)
        yield tag, v, start


def subject_cns(pem):
    der = base64.b64decode("".join(line for line in pem.splitlines() if "-----" not in line))
    _, c, _ = _tlv(der, 0)  # Certificate
    _, t, t_end = _tlv(der, c)  # tbsCertificate
    fields = [f for f in _items(der, t, t_end) if f[0] != 0xA0]  # drop [0] version
    _, s, s_end = fields[4]  # serial, signature, issuer, validity, SUBJECT
    cns = []
    for _, r, r_end in _items(der, s, s_end):  # RDN (SET)
        for _, a, a_end in _items(der, r, r_end):  # AttributeTypeAndValue
            (_, o, o_end), (tag, v, v_end) = list(_items(der, a, a_end))[:2]
            if der[o:o_end] == bytes.fromhex("550403"):  # 2.5.4.3 commonName
                cns.append(der[v:v_end].decode("utf-8", "replace") if tag in (0x0C, 0x13) else None)
    return cns


def thing_exists(name):
    import boto3

    iot = boto3.client("iot")
    try:
        iot.describe_thing(thingName=name)
        return True
    except iot.exceptions.ResourceNotFoundException:
        return False


def _param(name):
    return PREFIX + hashlib.sha256(name.encode()).hexdigest()


def allowed(name):
    import boto3

    ssm = boto3.client("ssm")
    try:
        return json.loads(ssm.get_parameter(Name=_param(name), WithDecryption=True)["Parameter"]["Value"])
    except ssm.exceptions.ParameterNotFound:
        return None


def consume(name):
    import boto3

    boto3.client("ssm").delete_parameter(Name=_param(name))


def decide(params, pem):
    thing = str(params.get("device_id") or "")
    if not thing or subject_cns(pem) != [thing] or thing_exists(thing):
        return None
    entry = allowed(thing)
    if not isinstance(entry, dict) or entry.get("device_id") != thing:
        return None
    secret = str(params.get("claim_secret") or "").encode()
    if not hmac.compare_digest(hashlib.sha256(secret).hexdigest(), str(entry.get("secret_sha256") or "")):
        return None
    if params.get("tenant") != entry.get("tenant") or params.get("device_type") != entry.get("device_type"):
        return None
    consume(thing)  # one robot per approval: a second claim for this name finds no entry
    return {"device_id": thing, "tenant": entry["tenant"], "device_type": entry["device_type"]}


def handler(event, _context):
    try:
        bound = decide(event.get("parameters") or {}, event.get("certificatePem", ""))
    except Exception:  # a malformed certificate or any AWS error denies
        bound = None
    return {"allowProvisioning": bound is not None, "parameterOverrides": bound or {}}
'''
"""Source of the pre-provisioning Lambda: a new, approved thing, named by its certificate's one subject CN."""


def hook_decision(
    parameters: dict[str, Any],
    certificate_pem: str,
    *,
    existing: Collection[str] = (),
    approved: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """The hook's rule, callable without AWS: ``existing`` are the things IoT has, ``approved`` the allowlist
    entries by device id (an accepted claim removes its entry from a dict passed here)."""
    ns: dict[str, Any] = {}
    exec(HOOK_SOURCE, ns)  # noqa: S102 — the shipped source is the rule under test
    entries: Any = approved if approved is not None else {}
    ns["thing_exists"] = lambda name: name in existing
    ns["allowed"] = lambda name: entries.get(name)
    ns["consume"] = lambda name: entries.pop(name, None) if isinstance(entries, dict) else None
    return bool(ns["handler"]({"parameters": parameters, "certificatePem": certificate_pem}, None)["allowProvisioning"])
