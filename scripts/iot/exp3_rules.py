"""Experiment 3: the rules engine on the tenant e-stop topic.

    python scripts/iot/exp3_rules.py [--n 50]

(a) rule `SELECT *, timestamp() AS at FROM 'strands/+/@estop'` → republish to `strands/${tenant}/@events/estop` (tenant from the payload): device publish → operator receives, p50/p95
(b) the same rule → SNS topic → SQS queue: device publish → SQS receive, p50/p95 (the hook path)
(c) Basic Ingest: publish straight to `$aws/rules/<rule>` → republish latency; the cost difference is stated in EXPERIMENTS.md
(d) retained estop state: the publisher sets RETAIN on `strands/<tenant>/@estop`; a NEW operator connecting later receives it on
    subscribe (latency from SUBACK), and `iot-data list_retained_messages` sees it. (Republish actions cannot set RETAIN — measured, not assumed.)
Creates: 1 IAM role (+ inline policy), 1 topic rule, 1 SNS topic, 1 SQS queue, + the exp1 device set. Deletes all of it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

from _common import REGION, TAG_MAP, TAGS, Mqtt5, Provisioned, account_and_endpoint, leftovers, name, pct, run_id, so101_manifest, sweep_deprecated_thing_types


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--role", default="strands-v1-iot-exp-e3-aee426-rule-role",
                    help="an existing IAM role for the rule actions (bring your own: a role created seconds before the rule "
                         "drops messages until IAM propagates)")
    a = ap.parse_args()
    import boto3

    iot = boto3.client("iot", region_name=REGION)
    iam, sns, sqs = boto3.client("iam"), boto3.client("sns", region_name=REGION), boto3.client("sqs", region_name=REGION)
    run, dev, tenant = run_id(), so101_manifest(), "lab"
    account, endpoint = account_and_endpoint(iot)
    work = Path(tempfile.mkdtemp(prefix="strands-iot-exp3-"))
    p = Provisioned(iot, exp="e3", run=run, tenant=tenant, device_type=dev["device_type"], device_id=dev["id"], workdir=work)
    role_name, rule_name = name("e3", run, "rule-role"), name("e3", run, "estop").replace("-", "_")
    topic_name, queue_name = name("e3", run, "estop"), name("e3", run, "estop")
    out: dict = {"experiment": 3, "run": run}
    created: dict = {}
    try:
        p.create(account, [dev["id"]], dev)
        t = time.perf_counter()
        q = sqs.create_queue(QueueName=queue_name, tags=TAG_MAP)["QueueUrl"]
        created["queue"] = q
        q_arn = sqs.get_queue_attributes(QueueUrl=q, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]
        topic_arn = sns.create_topic(Name=topic_name, Tags=TAGS)["TopicArn"]
        created["topic"] = topic_arn
        sqs.set_queue_attributes(QueueUrl=q, Attributes={"Policy": json.dumps({"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow", "Principal": {"Service": "sns.amazonaws.com"}, "Action": "sqs:SendMessage", "Resource": q_arn,
            "Condition": {"ArnEquals": {"aws:SourceArn": topic_arn}}}]})})
        sns.subscribe(TopicArn=topic_arn, Protocol="sqs", Endpoint=q_arn, Attributes={"RawMessageDelivery": "true"})["SubscriptionArn"]
        role_name = a.role
        try:
            role = iam.get_role(RoleName=role_name)["Role"]["Arn"]
            out["rule_role"] = "reused (pre-existing)"
        except iam.exceptions.NoSuchEntityException:
            role = iam.create_role(RoleName=role_name, Tags=TAGS, AssumeRolePolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Principal": {"Service": "iot.amazonaws.com"}, "Action": "sts:AssumeRole",
                "Condition": {"StringEquals": {"aws:SourceAccount": account}}}]}))["Role"]["Arn"]
            created["role"] = role_name
            out["rule_role"] = "created (cannot be deleted by this session: iam:DeleteRole is explicitly denied)"
        iam.put_role_policy(RoleName=role_name, PolicyName="rule", PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": "iot:Publish", "Resource": f"arn:aws:iot:{REGION}:{account}:topic/strands/*/@events/*"},
            {"Effect": "Allow", "Action": "sns:Publish", "Resource": topic_arn}]}))
        out["hooks_setup_ms"] = round((time.perf_counter() - t) * 1e3)
        # the tenant rides IN the payload: with Basic Ingest there is no MQTT topic, so topic(2) would be "rules"
        sql = "SELECT *, timestamp() AS at FROM 'strands/+/@estop'"
        for attempt in range(20):  # IAM propagation
            try:
                iot.create_topic_rule(ruleName=rule_name, tags=f"Project={TAG_MAP['Project']}", topicRulePayload={
                    "sql": sql, "awsIotSqlVersion": "2016-03-23", "ruleDisabled": False, "actions": [
                        {"republish": {"roleArn": role, "topic": "strands/${tenant}/@events/estop", "qos": 1}},
                        {"sns": {"targetArn": topic_arn, "roleArn": role, "messageFormat": "RAW"}}]})
                created["rule"] = rule_name
                break
            except Exception as exc:
                if attempt == 19:
                    raise
                time.sleep(1.0)
        out["rule_create_attempts_iam_propagation"] = attempt + 1
        time.sleep(1.0)
        link = Mqtt5(endpoint, p.thing, cert=p.cert_path, key=p.key_path, timeout=10.0)
        op = Mqtt5(endpoint, name("e3", run, "operator"))
        got: list[tuple[float, dict]] = []
        op.on(lambda pkt: got.append((time.perf_counter(), json.loads(bytes(pkt.payload or b"{}")))))
        op.subscribe(f"strands/{tenant}/@events/estop")
        # warm the rule (first trigger is slow), then measure
        def fire(topic: str, i: int, retain: bool = False, via=None) -> float:
            got.clear()
            t = time.perf_counter()
            (via or link).publish(topic, json.dumps({"estop": True, "seq": i, "by": "operator", "tenant": tenant, "sent": time.time()}).encode(), qos=1, retain=retain)
            while not got and time.perf_counter() - t < 3:
                time.sleep(0.0005)
            return (got[0][0] - t) * 1e3 if got else float("nan")

        t_warm, lost = time.perf_counter(), 0
        while True:  # a rule whose role policy is seconds old drops messages silently until IAM propagates
            warm = fire(f"strands/{tenant}/@estop", -1)
            if not math.isnan(warm):
                break
            lost += 1
            if time.perf_counter() - t_warm > 120:
                raise SystemExit("rule never republished within 120 s")
        out["rule_warmup"] = {"seconds_until_first_republish": round(time.perf_counter() - t_warm, 1), "messages_lost_silently": lost, "first_ms": round(warm, 1)}
        time.sleep(0.5)
        rep = [fire(f"strands/{tenant}/@estop", i) for i in range(a.n)]
        out["rule_republish_ms"] = {**pct([x for x in rep if not math.isnan(x)]), "lost": sum(1 for x in rep if math.isnan(x))}
        out["republish_payload_example"] = got[0][1] if got else None
        # Basic Ingest: skip the pub/sub layer, hit the rule directly
        # (the device policy has no $aws/rules/* publish right, so a device publish there is refused: PUBACK 135; measured from the operator)
        r = link.publish(f"$aws/rules/{rule_name}", json.dumps({"estop": True, "seq": -2, "tenant": tenant}).encode(), qos=1)
        out["basic_ingest_from_device_without_policy_puback"] = int(r.puback.reason_code) if r and r.puback else None
        bi = [fire(f"$aws/rules/{rule_name}", i, via=op) for i in range(a.n)]
        rep_op = [fire(f"strands/{tenant}/@estop", i, via=op) for i in range(a.n)]
        out["rule_republish_from_operator_ms"] = {**pct([x for x in rep_op if not math.isnan(x)]), "lost": sum(1 for x in rep_op if math.isnan(x))}
        out["basic_ingest_republish_ms"] = {**pct([x for x in bi if not math.isnan(x)]), "lost": sum(1 for x in bi if math.isnan(x))}
        # SNS → SQS hook latency: drain, fire n, receive with timestamps
        while sqs.receive_message(QueueUrl=q, MaxNumberOfMessages=10, WaitTimeSeconds=1).get("Messages"):
            pass
        sent_at: dict[int, float] = {}
        for i in range(a.n):
            sent_at[i] = time.perf_counter()
            link.publish(f"strands/{tenant}/@estop", json.dumps({"estop": True, "seq": i, "hook": True}).encode(), qos=1)
            time.sleep(0.05)
        lat, deadline = [], time.perf_counter() + 20
        while len(lat) < a.n and time.perf_counter() < deadline:
            r = sqs.receive_message(QueueUrl=q, MaxNumberOfMessages=10, WaitTimeSeconds=2, AttributeNames=["SentTimestamp"])
            for m in r.get("Messages", []):
                body = json.loads(m["Body"])
                if body.get("hook") and body["seq"] in sent_at:
                    # SentTimestamp = when SQS got it (ms epoch); compare with our wall clock at publish
                    lat.append(int(m["Attributes"]["SentTimestamp"]) / 1e3 - (time.time() - (time.perf_counter() - sent_at[body["seq"]])))
                sqs.delete_message(QueueUrl=q, ReceiptHandle=m["ReceiptHandle"])
        out["rule_sns_sqs_ms"] = pct([x * 1e3 for x in lat])
        # retained e-stop state: publish with RETAIN, then a NEW operator subscribes and receives it
        r = link.publish(f"strands/{tenant}/@estop", json.dumps({"estop": True, "retained": True, "seq": 999}).encode(), qos=1, retain=True)
        out["retain_publish_puback"] = int(r.puback.reason_code) if r and r.puback else None  # 135 without iot:RetainPublish
        time.sleep(0.5)
        late = Mqtt5(endpoint, name("e3", run, "late-operator"))
        got2: list[tuple[float, dict, bool]] = []
        late.on(lambda pkt: got2.append((time.perf_counter(), json.loads(bytes(pkt.payload or b"{}")), bool(pkt.retain))))
        t = time.perf_counter()
        codes, sub_ms = late.subscribe(f"strands/{tenant}/@estop")
        while not got2 and time.perf_counter() - t < 5:
            time.sleep(0.0005)
        out["retained_on_subscribe"] = {"received": bool(got2), "retain_flag": got2[0][2] if got2 else None,
                                        "ms_after_subscribe_call": round((got2[0][0] - t) * 1e3, 1) if got2 else None, "suback_ms": sub_ms}
        data = boto3.client("iot-data", region_name=REGION, endpoint_url=f"https://{endpoint}")
        lr = data.list_retained_messages()
        out["list_retained_messages_sees_it"] = any(r["topic"] == f"strands/{tenant}/@estop" for r in lr.get("retainedTopics", []))
        # did the republished copy carry RETAIN? (it cannot; check the flag on the @events copy)
        got.clear()
        link.publish(f"strands/{tenant}/@estop", json.dumps({"estop": True, "seq": 1000}).encode(), qos=1, retain=True)
        time.sleep(1.0)
        late2 = Mqtt5(endpoint, name("e3", run, "late-operator2"))
        got3: list = []
        late2.on(lambda pkt: got3.append(bool(pkt.retain)))
        late2.subscribe(f"strands/{tenant}/@events/estop")
        time.sleep(1.5)
        out["republished_copy_is_retained"] = bool(got3)
        # clear the retained message (empty retained payload) and close links
        link.publish(f"strands/{tenant}/@estop", b"", qos=1, retain=True)
        time.sleep(0.5)
        out["retained_cleared"] = not any(r["topic"] == f"strands/{tenant}/@estop" for r in data.list_retained_messages().get("retainedTopics", []))
        for l in (late, late2, op, link):
            l.close()
    finally:
        done = []
        if "rule" in created:
            iot.delete_topic_rule(ruleName=rule_name); done.append("topic rule")
        try:
            iam.delete_role_policy(RoleName=role_name, PolicyName="rule"); done.append("iam role inline policy")
        except Exception as exc:
            done.append(f"FAILED role policy: {type(exc).__name__}")
        if "role" in created:
            try:
                iam.delete_role(RoleName=role_name); done.append("iam role")
            except Exception as exc:
                done.append(f"FAILED iam role (owner must delete): {type(exc).__name__}")
        if "topic" in created:
            for sub in sns.list_subscriptions_by_topic(TopicArn=created["topic"])["Subscriptions"]:
                if sub["SubscriptionArn"].startswith("arn:"):
                    sns.unsubscribe(SubscriptionArn=sub["SubscriptionArn"])
            sns.delete_topic(TopicArn=created["topic"]); done.append("sns topic")
        if "queue" in created:
            sqs.delete_queue(QueueUrl=created["queue"]); done.append("sqs queue")
        done += p.delete()
        out["deleted"] = done
        out["swept_thing_types"] = sweep_deprecated_thing_types(iot)
        lo = leftovers(iot)
        lo["sns_topics"] = [t["TopicArn"].rsplit(":", 1)[-1] for t in sns.list_topics().get("Topics", []) if "strands-v1-iot" in t["TopicArn"]]
        lo["sqs_queues"] = [u.rsplit("/", 1)[-1] for u in sqs.list_queues(QueueNamePrefix="strands-v1-iot").get("QueueUrls", [])]
        lo["iam_roles"] = [r["RoleName"] for pg in iam.get_paginator("list_roles").paginate() for r in pg["Roles"] if r["RoleName"].startswith("strands-v1-iot")]
        out["leftovers"] = lo
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
