"""Experiment 6: fleet provisioning by claim — template + pre-provisioning hook + claim cert → a device certificate from a LOCAL CSR.

    python scripts/iot/exp6_claim_provisioning.py [--role <existing iam role>]

Flow (what `Mesh.mount(…, iot=…)` would do on a fresh robot): connect with the CLAIM cert (MQTT5 mTLS, client id = device id),
`$aws/certificates/create-from-csr/json` with a CSR whose CN = the thing name the template will create, then
`$aws/provisioning-templates/<t>/provision/json` with the ownership token + parameters {DeviceId, DeviceType, Tenant}. The pre-provisioning
hook (Lambda) DENIES unless the certificate's CN equals `strands-v1-iot-exp-e6-<run>-<DeviceId>` (prior-art fact b: a claim device must not
inherit another robot's rights by naming it in its CSR). Then the device connects with its new cert as the thing and the claim cert is retired.
Measured: each step in ms, the hook's deny on a forged CN, the whole claim→connected time. Everything is deleted; the IAM role is reused
(iam:DeleteRole is denied for this session) and its inline policies are removed.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from _common import REGION, TAG_MAP, TAGS, Mqtt5, account_and_endpoint, device_policy, leftovers, name, run_id, sweep_deprecated_thing_types

HOOK = r'''
import re, base64
def _cn(pem):
    der = base64.b64decode("".join(l for l in pem.splitlines() if "-----" not in l))
    i = der.find(bytes.fromhex("0603550403"))  # OID 2.5.4.3 commonName
    if i < 0:
        return None
    j = i + 5
    tag, ln = der[j], der[j + 1]
    return der[j + 2:j + 2 + ln].decode("utf-8", "replace") if tag in (0x0C, 0x13) else None
def handler(event, _ctx):
    params = event.get("parameters", {})
    want = "%s-" % PREFIX + params.get("DeviceId", "")
    cn = _cn(event.get("certificatePem", ""))
    ok = cn == want and bool(params.get("DeviceId"))
    return {"allowProvisioning": ok, "parameterOverrides": {"Tenant": params.get("Tenant", "lab")}}
'''


def csr(workdir: Path, cn: str) -> tuple[Path, str]:
    key = workdir / f"{cn}.key.pem"
    req = workdir / f"{cn}.csr.pem"
    subprocess.run(["openssl", "req", "-new", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(req), "-subj", f"/CN={cn}"],
                   check=True, capture_output=True)
    key.chmod(0o600)
    return key, req.read_text()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="strands-v1-iot-exp-e3-aee426-rule-role")
    a = ap.parse_args()
    import boto3

    iot = boto3.client("iot", region_name=REGION)
    iam, lam = boto3.client("iam"), boto3.client("lambda", region_name=REGION)
    run, tenant, device_id = run_id(), "lab", "arm"
    account, endpoint = account_and_endpoint(iot)
    work = Path(tempfile.mkdtemp(prefix="strands-iot-exp6-"))
    prefix = name("e6", run)  # things are <prefix>-<DeviceId>
    template = f"strands-v1-iot-exp-e6-{run}-tpl"  # template names are capped at 36 chars
    hook_fn, claim_policy, dev_policy = name("e6", run, "hook"), name("e6", run, "claim-policy"), name("e6", run, "device-policy")
    out: dict = {"experiment": 6, "run": run, "timings_ms": {}}
    created: dict = {}
    T = out["timings_ms"]

    def timed(k, fn, **kw):
        t = time.perf_counter()
        r = fn(**kw)
        T[k] = round((time.perf_counter() - t) * 1e3, 1)
        return r

    try:
        role_arn = iam.get_role(RoleName=a.role)["Role"]["Arn"]
        # the provisioning role needs the registration rights; the hook's execution role needs nothing (no logs written)
        iam.put_role_policy(RoleName=a.role, PolicyName="provisioning", PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow", "Action": ["iot:AddThingToThingGroup", "iot:AttachPolicy", "iot:AttachPrincipalPolicy", "iot:AttachThingPrincipal",
                                          "iot:CreateCertificateFromCsr", "iot:CreatePolicy", "iot:CreateThing", "iot:DescribeCertificate",
                                          "iot:DescribeThing", "iot:DescribeThingGroup", "iot:DescribeThingType", "iot:DetachThingPrincipal",
                                          "iot:GetPolicy", "iot:ListPolicyPrincipals", "iot:ListPrincipalPolicies", "iot:ListPrincipalThings",
                                          "iot:ListThingGroupsForThing", "iot:ListThingPrincipals", "iot:RegisterCertificate", "iot:RegisterThing",
                                          "iot:RemoveThingFromThingGroup", "iot:UpdateCertificate", "iot:UpdateThing", "iot:UpdateThingGroupsForThing",
                                          "iot:AddThingToBillingGroup", "iot:DescribeBillingGroup", "iot:RemoveThingFromBillingGroup"], "Resource": "*"}]}))
        created["role_policy"] = True
        # pre-provisioning hook
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("hook.py", HOOK.replace("%s-\" % PREFIX", f'{prefix}-"'))
        fn: dict[str, Any] | None = None
        for attempt in range(10):
            try:
                fn = timed("lambda_create", lam.create_function, FunctionName=hook_fn, Runtime="python3.12", Role=role_arn, Handler="hook.handler",
                           Code={"ZipFile": buf.getvalue()}, Timeout=5, Tags=TAG_MAP)
                break
            except lam.exceptions.InvalidParameterValueException:
                time.sleep(2)
        if fn is None:
            raise RuntimeError(f"lambda {hook_fn}: the role never became assumable in 10 tries")
        created["lambda"] = True
        lam.get_waiter("function_active_v2").wait(FunctionName=hook_fn)
        lam.add_permission(FunctionName=hook_fn, StatementId="iot", Action="lambda:InvokeFunction", Principal="iot.amazonaws.com", SourceAccount=account)
        # the device policy the template attaches: the exp-1 least-privilege document (policy variables, so ONE document per tenant)
        timed("create_device_policy", iot.create_policy, policyName=dev_policy, tags=TAGS,
              policyDocument=json.dumps(device_policy(REGION, account, tenant, [device_id])))
        created["dev_policy"] = True
        body = {"Parameters": {"DeviceId": {"Type": "String"}, "DeviceType": {"Type": "String"}, "Tenant": {"Type": "String"},
                               "AWS::IoT::Certificate::Id": {"Type": "String"}},
                "Resources": {
                    "thing": {"Type": "AWS::IoT::Thing", "OverrideSettings": {"AttributePayload": "MERGE", "ThingGroups": "DO_NOTHING"},
                              "Properties": {"ThingName": {"Fn::Join": ["", [f"{prefix}-", {"Ref": "DeviceId"}]]},
                                             "AttributePayload": {"tenant": {"Ref": "Tenant"}, "device_type": {"Ref": "DeviceType"}, "provisioned_by": "claim"}}},
                    "certificate": {"Type": "AWS::IoT::Certificate", "Properties": {"CertificateId": {"Ref": "AWS::IoT::Certificate::Id"}, "Status": "Active"}},
                    "policy": {"Type": "AWS::IoT::Policy", "Properties": {"PolicyName": dev_policy}}}}
        timed("create_provisioning_template", iot.create_provisioning_template, templateName=template, enabled=True, provisioningRoleArn=role_arn,
              templateBody=json.dumps(body), tags=TAGS, preProvisioningHook={"targetArn": fn["FunctionArn"], "payloadVersion": "2020-04-01"})
        created["template"] = True
        # claim certificate + its narrow policy
        claim_doc = {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": "iot:Connect", "Resource": f"arn:aws:iot:{REGION}:{account}:client/{prefix}-*"},
            {"Effect": "Allow", "Action": "iot:Publish", "Resource": [f"arn:aws:iot:{REGION}:{account}:topic/$aws/certificates/create-from-csr/*",
                                                                     f"arn:aws:iot:{REGION}:{account}:topic/$aws/provisioning-templates/{template}/provision/*"]},
            {"Effect": "Allow", "Action": "iot:Subscribe", "Resource": [f"arn:aws:iot:{REGION}:{account}:topicfilter/$aws/certificates/create-from-csr/*",
                                                                       f"arn:aws:iot:{REGION}:{account}:topicfilter/$aws/provisioning-templates/{template}/provision/*"]},
            {"Effect": "Allow", "Action": "iot:Receive", "Resource": [f"arn:aws:iot:{REGION}:{account}:topic/$aws/certificates/create-from-csr/*",
                                                                     f"arn:aws:iot:{REGION}:{account}:topic/$aws/provisioning-templates/{template}/provision/*"]}]}
        timed("create_claim_policy", iot.create_policy, policyName=claim_policy, policyDocument=json.dumps(claim_doc), tags=TAGS)
        created["claim_policy"] = True
        claim = timed("create_claim_cert", iot.create_keys_and_certificate, setAsActive=True)
        created["claim_cert"] = (claim["certificateId"], claim["certificateArn"])
        iot.attach_policy(policyName=claim_policy, target=claim["certificateArn"])
        (work / "claim.cert.pem").write_text(claim["certificatePem"]); (work / "claim.key.pem").write_text(claim["keyPair"]["PrivateKey"])
        time.sleep(2.0)

        # --- the device side: claim connect → CSR → register thing → reconnect as the thing
        t_all = time.perf_counter()
        link = Mqtt5(endpoint, f"{prefix}-{device_id}-claim", cert=work / "claim.cert.pem", key=work / "claim.key.pem", timeout=15.0)
        T["claim_connect"] = link.connect_ms
        inbox: dict[str, list] = {}
        ev = threading.Event()

        def on(pkt):
            inbox.setdefault(str(pkt.topic), []).append(json.loads(bytes(pkt.payload or b"{}")))
            ev.set()

        link.on(on)
        for f in ("$aws/certificates/create-from-csr/json/accepted", "$aws/certificates/create-from-csr/json/rejected",
                  f"$aws/provisioning-templates/{template}/provision/json/accepted", f"$aws/provisioning-templates/{template}/provision/json/rejected"):
            codes, _ = link.subscribe(f, qos=1)
            assert codes[0] < 0x80, (f, codes)

        def ask(topic: str, payload: dict, accepted: str, rejected: str, key: str) -> dict | None:
            ev.clear()
            t = time.perf_counter()
            link.publish(topic, json.dumps(payload).encode(), qos=1)
            while not (inbox.get(accepted) or inbox.get(rejected)) and time.perf_counter() - t < 15:
                ev.wait(0.2)
            T[key] = round((time.perf_counter() - t) * 1e3, 1)
            if inbox.get(rejected):
                return {"rejected": inbox[rejected].pop(0)}
            return inbox[accepted].pop(0) if inbox.get(accepted) else None

        # 1) a FORGED CSR: CN names another robot → the hook must deny
        _, forged = csr(work, f"{prefix}-someone-else")
        r = ask("$aws/certificates/create-from-csr/json", {"certificateSigningRequest": forged},
                "$aws/certificates/create-from-csr/json/accepted", "$aws/certificates/create-from-csr/json/rejected", "csr_forged")
        forged_cert_id = r.get("certificateId") if r and "rejected" not in r else None
        r2 = ask(f"$aws/provisioning-templates/{template}/provision/json",
                 {"certificateOwnershipToken": r["certificateOwnershipToken"], "parameters": {"DeviceId": device_id, "DeviceType": "robot_arm", "Tenant": tenant}},
                 f"$aws/provisioning-templates/{template}/provision/json/accepted", f"$aws/provisioning-templates/{template}/provision/json/rejected", "register_forged")
        out["forged_cn_register"] = r2.get("rejected", r2) if r2 else None
        # 2) the honest CSR
        key_path, honest = csr(work, f"{prefix}-{device_id}")
        r = ask("$aws/certificates/create-from-csr/json", {"certificateSigningRequest": honest},
                "$aws/certificates/create-from-csr/json/accepted", "$aws/certificates/create-from-csr/json/rejected", "csr_to_cert")
        assert r and "rejected" not in r, r
        cert_id = r["certificateId"]
        (work / "device.cert.pem").write_text(r["certificatePem"])
        r2 = ask(f"$aws/provisioning-templates/{template}/provision/json",
                 {"certificateOwnershipToken": r["certificateOwnershipToken"], "parameters": {"DeviceId": device_id, "DeviceType": "robot_arm", "Tenant": tenant}},
                 f"$aws/provisioning-templates/{template}/provision/json/accepted", f"$aws/provisioning-templates/{template}/provision/json/rejected", "register_thing")
        assert r2 and "rejected" not in r2, r2
        out["registered_thing"] = r2.get("thingName")
        out["device_configuration_from_hook"] = r2.get("deviceConfiguration")
        link.close()
        # 3) connect as the thing with the new cert, prove the policy is the exp-1 one
        dev: Mqtt5 | None = None
        for attempt in range(20):
            try:
                dev = Mqtt5(endpoint, r2["thingName"], cert=work / "device.cert.pem", key=key_path, timeout=6.0)
                break
            except TimeoutError:
                time.sleep(0.5)
        if dev is None:
            raise RuntimeError(f"{r2['thingName']}: the new certificate never connected in 20 tries")
        T["new_cert_first_connect_attempts"] = attempt + 1
        T["new_cert_connect"] = dev.connect_ms
        T["claim_to_connected_total"] = round((time.perf_counter() - t_all) * 1e3, 1)
        codes, _ = dev.subscribe(f"strands/{tenant}/{device_id}/+")
        out["new_cert_sub_own"] = codes
        codes, _ = dev.subscribe("strands/other-tenant/#")
        out["new_cert_sub_other_tenant"] = codes
        th = iot.describe_thing(thingName=r2["thingName"])
        out["thing_attributes"] = th.get("attributes")
        cert_desc = iot.describe_certificate(certificateId=cert_id)["certificateDescription"]
        out["device_cert_cn_is_thing"] = bool(re.search(rf"CN={re.escape(r2['thingName'])}", subprocess.run(["openssl", "x509", "-noout", "-subject", "-in", str(work / "device.cert.pem")], capture_output=True, text=True).stdout))
        out["device_cert_status"] = cert_desc["status"]
        dev.close()
        # 4) retire the claim cert (one-shot claims) — part of the design, measured
        timed("retire_claim_cert", iot.update_certificate, certificateId=claim["certificateId"], newStatus="INACTIVE")
        created["device"] = (r2["thingName"], cert_id, r["certificateId"])
        created["forged_cert"] = forged_cert_id
    finally:
        done = []
        if "device" in created:
            thing, cid, _ = created["device"]
            arn = iot.describe_certificate(certificateId=cid)["certificateDescription"]["certificateArn"]
            for fn, kw in ((iot.detach_thing_principal, {"thingName": thing, "principal": arn}), (iot.detach_policy, {"policyName": dev_policy, "target": arn}),
                           (iot.update_certificate, {"certificateId": cid, "newStatus": "INACTIVE"}), (iot.delete_certificate, {"certificateId": cid, "forceDelete": True}),
                           (iot.delete_thing, {"thingName": thing})):
                try:
                    fn(**kw)
                except Exception as exc:
                    done.append(f"FAILED {fn.__name__}: {type(exc).__name__}")
            done.append("device thing + cert")
        if created.get("forged_cert"):
            try:
                iot.update_certificate(certificateId=created["forged_cert"], newStatus="INACTIVE")
                iot.delete_certificate(certificateId=created["forged_cert"], forceDelete=True)
                done.append("forged (denied) cert")
            except Exception as exc:
                done.append(f"FAILED forged cert: {type(exc).__name__}")
        if "claim_cert" in created:
            cid, arn = created["claim_cert"]
            try:
                iot.detach_policy(policyName=claim_policy, target=arn)
                iot.update_certificate(certificateId=cid, newStatus="INACTIVE"); iot.delete_certificate(certificateId=cid, forceDelete=True)
                done.append("claim cert")
            except Exception as exc:
                done.append(f"FAILED claim cert: {type(exc).__name__}")
        for flag, fn, kw, what in (("claim_policy", iot.delete_policy, {"policyName": claim_policy}, "claim policy"),
                                   ("template", iot.delete_provisioning_template, {"templateName": template}, "provisioning template"),
                                   ("dev_policy", iot.delete_policy, {"policyName": dev_policy}, "device policy"),
                                   ("lambda", lam.delete_function, {"FunctionName": hook_fn}, "hook lambda"),
                                   ("role_policy", iam.delete_role_policy, {"RoleName": a.role, "PolicyName": "provisioning"}, "role inline policy")):
            if created.get(flag):
                try:
                    fn(**kw); done.append(what)
                except Exception as exc:
                    done.append(f"FAILED {what}: {type(exc).__name__}")
        for f in work.glob("*.pem"):
            f.unlink()
        out["deleted"] = done
        out["swept_thing_types"] = sweep_deprecated_thing_types(iot)
        lo = leftovers(iot)
        lo["lambdas"] = [f["FunctionName"] for f in lam.list_functions()["Functions"] if f["FunctionName"].startswith("strands-v1-iot-exp")]
        lo["role_inline_policies"] = iam.list_role_policies(RoleName=a.role)["PolicyNames"]
        out["leftovers"] = lo
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
