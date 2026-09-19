#!/usr/bin/env bash
# Terminate the example instance and remove the security group. The IAM
# role/instance-profile (strands-isaac-ssm) is left in place - it is inert,
# free, and may be shared by another instance; delete it by hand if unwanted.
set -euo pipefail
STATE_FILE="$(dirname "$0")/.instance.json"
[ -f "$STATE_FILE" ] || { echo "no .instance.json - nothing to tear down"; exit 0; }
IID=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['instance_id'])")
REGION=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['region'])")
SG=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['security_group'])")
echo "terminating $IID in $REGION"
aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" >/dev/null
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$IID"
aws ec2 delete-security-group --region "$REGION" --group-id "$SG" >/dev/null 2>&1 || \
  echo "security group $SG left (still referenced); re-run teardown later to remove it"
rm -f "$STATE_FILE"
echo "done"
