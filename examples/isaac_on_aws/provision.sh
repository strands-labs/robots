#!/usr/bin/env bash
# Provision an AWS GPU instance that can run the Isaac Sim backend end to end.
#
# What this creates (and what teardown.sh removes):
#   - one EC2 instance (default g5.2xlarge: NVIDIA A10G, the GPU this backend
#     is verified against), Ubuntu 22.04, with the NVIDIA driver, docker and
#     the nvidia-container-toolkit installed by cloud-init, and the pinned
#     nvcr.io/nvidia/isaac-sim:6.0.1 container pulled;
#   - one security group with NO ingress rules - every interaction goes
#     through AWS Systems Manager (SSM), so nothing listens on the network;
#   - one IAM role/instance-profile (strands-isaac-ssm) carrying only
#     AmazonSSMManagedInstanceCore, created once and reused.
#
# Requirements on the machine running this script: aws CLI v2 with credentials
# that can create the above. Nothing else.
#
# Isaac Sim needs an RT-core GPU: g5 (A10G) and g6 (L4) work; p3/p4/p5
# (V100/A100/H100) do NOT - they have no RT cores and Isaac Sim refuses them.
set -euo pipefail

REGION="${REGION:-us-west-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.2xlarge}"
VOLUME_GB="${VOLUME_GB:-200}"          # the container alone is ~32 GB
NAME_TAG="${NAME_TAG:-strands-isaac-example}"
STATE_FILE="$(dirname "$0")/.instance.json"

echo "== strands-robots Isaac-on-AWS provision =="
echo "   region=$REGION type=$INSTANCE_TYPE disk=${VOLUME_GB}GB"

# --- IAM role + instance profile (idempotent) --------------------------------
ROLE=strands-isaac-ssm
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]
  }' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
fi
if ! aws iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  sleep 10   # IAM propagation before RunInstances references the profile
fi

# --- security group with no ingress ------------------------------------------
VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=is-default,Values=true \
      --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --region "$REGION" \
     --filters Name=group-name,Values="$NAME_TAG" Name=vpc-id,Values="$VPC" \
     --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG" ] || [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --region "$REGION" --vpc-id "$VPC" \
       --group-name "$NAME_TAG" --description "strands-robots isaac example - SSM only, no ingress" \
       --query 'GroupId' --output text)
fi

# --- Ubuntu 22.04 AMI via the canonical public SSM parameter -----------------
AMI=$(aws ssm get-parameter --region "$REGION" \
  --name /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
  --query 'Parameter.Value' --output text)

# --- cloud-init: driver + docker + nvidia-container-toolkit ------------------
USER_DATA=$(cat <<'CLOUDINIT'
#!/bin/bash
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl gnupg
# NVIDIA driver, named explicitly. 'ubuntu-drivers install --gpgpu' returns 0
# on this AMI while installing NOTHING (measured on the first end-to-end run
# of this script: no nvidia package, no module, no nvidia-smi - and the ||
# fallback never fired because the exit code lied). The server flavour loads
# without a reboot; the smoke run saw the A10G immediately after modprobe.
apt-get install -y nvidia-driver-535-server nvidia-utils-535-server
modprobe nvidia || true
# docker
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable" > /etc/apt/sources.list.d/docker.list
apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io
# nvidia container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
touch /var/local/strands-bootstrap-done
CLOUDINIT
)

echo "-- launching instance"
IID=$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
  --iam-instance-profile Name="$ROLE" \
  --security-group-ids "$SG" \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$VOLUME_GB,VolumeType=gp3}" \
  --metadata-options HttpTokens=required \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME_TAG}]" \
  --user-data "$USER_DATA" \
  --query 'Instances[0].InstanceId' --output text)
echo "   instance: $IID"
printf '{"instance_id": "%s", "region": "%s", "security_group": "%s"}\n' "$IID" "$REGION" "$SG" > "$STATE_FILE"

echo "-- waiting for the instance to be running"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"

echo "-- waiting for SSM to come online (this includes the OS boot)"
for i in $(seq 1 60); do
  STATE=$(aws ssm describe-instance-information --region "$REGION" \
    --filters Key=InstanceIds,Values="$IID" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
  [ "$STATE" = "Online" ] && break
  sleep 10
done
[ "$STATE" = "Online" ] || { echo "SSM never came online"; exit 1; }

run_ssm() {  # run_ssm "<command>" <timeout-polls>
  local CID
  CID=$(aws ssm send-command --region "$REGION" --instance-ids "$IID" \
    --document-name AWS-RunShellScript --parameters "commands=[\"$1\"]" \
    --query 'Command.CommandId' --output text)
  for _ in $(seq 1 "${2:-30}"); do
    sleep 10
    local S
    S=$(aws ssm get-command-invocation --region "$REGION" --command-id "$CID" \
        --instance-id "$IID" --query 'Status' --output text 2>/dev/null || true)
    case "$S" in Success) return 0;; Failed|Cancelled|TimedOut) return 1;; esac
  done
  return 1
}

echo "-- waiting for cloud-init (driver + docker; typically 4-8 minutes)"
run_ssm "test -f /var/local/strands-bootstrap-done" 90 || \
run_ssm "cloud-init status --wait && test -f /var/local/strands-bootstrap-done" 90

echo "-- verifying the GPU is visible"
run_ssm "nvidia-smi --query-gpu=name --format=csv,noheader" 6 || { echo "nvidia-smi failed - driver install did not land"; exit 1; }

echo "-- pulling nvcr.io/nvidia/isaac-sim:6.0.1 (~32 GB; typically 5-15 minutes)"
# Full major.minor.patch tag: NVIDIA publishes no :6.0 or :latest for this
# image (docker manifest inspect: 'no such manifest').
run_ssm "docker pull nvcr.io/nvidia/isaac-sim:6.0.1 >/tmp/pull.log 2>&1 && docker images -q nvcr.io/nvidia/isaac-sim:6.0.1 | grep -q ." 120 \
  || { echo "image pull failed - see /tmp/pull.log on the instance"; exit 1; }

echo "== ready. Instance $IID in $REGION. Next: ./run_smoke.sh  When done: ./teardown.sh =="
