#!/usr/bin/env bash
# Mesh mTLS for the cloud hub (plan-cloud.md step 3). Run on the desktop from the repo root:
#   bash deploy/cloud/mesh-certs.sh
# 1. Makes (once) a private CA in ~/.strands-mesh/ca and one cert per mesh member, all issued directly by it:
#      zenoh-router     the cloud Zenoh router; SAN = the hub's public name + localhost (peers verify it)
#      dashboard-cloud  the cloud dashboard (CN must equal its mesh peer id)
#      woodside         the Pi
#      leader-arms      the desktop leader arms
# 2. Stores the cloud's files as GitHub secret MESH_CLOUD_BUNDLE (the Cloud workflow unpacks it on the droplet).
# 3. Copies woodside's files to the Pi (~/.strands-mesh) and points ~/strands-control.env at the cloud hub.
# 4. Copies leader-arms' files to deploy/strands/mesh/ (gitignored, mounted by the desktop compose).
# Re-running reuses the CA and existing certs. Env: HUB_HOST (default 45-55-122-126.sslip.io), PI_SSH (xlerobot).
set -euo pipefail
HUB_HOST="${HUB_HOST:-45-55-122-126.sslip.io}"
PI_SSH="${PI_SSH:-xlerobot}"
REPO="${REPO:-Jacob-Brokloff/robots}"
D="$HOME/.strands-mesh"
umask 077
mkdir -p "$D/ca" "$D/certs"
chmod 700 "$D" "$D/ca" "$D/certs"

if [ ! -f "$D/ca/ca.key" ]; then
  openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$D/ca/ca.key" 2>/dev/null
  openssl req -x509 -new -key "$D/ca/ca.key" -sha256 -days 3650 -subj "/CN=strands-mesh-ca" \
    -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$D/ca/ca.pem"
  echo "created CA $D/ca"
fi

issue() {  # issue <cn> <subjectAltName>
  local cn="$1" san="$2" dir="$D/certs/$1"
  [ -f "$dir/$cn.pem" ] && return 0
  mkdir -p "$dir" && chmod 700 "$dir"
  openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$dir/$cn.key" 2>/dev/null
  chmod 600 "$dir/$cn.key"
  openssl req -new -key "$dir/$cn.key" -subj "/CN=$cn" -out "$dir/$cn.csr"
  printf 'basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth,clientAuth\nsubjectAltName=%s\n' "$san" > "$dir/ext.cnf"
  openssl x509 -req -in "$dir/$cn.csr" -CA "$D/ca/ca.pem" -CAkey "$D/ca/ca.key" -CAcreateserial \
    -days 825 -sha256 -extfile "$dir/ext.cnf" -out "$dir/$cn.pem" 2>/dev/null
  rm -f "$dir/$cn.csr" "$dir/ext.cnf"
  cp "$D/ca/ca.pem" "$dir/ca.pem"
  chmod 644 "$dir/$cn.pem" "$dir/ca.pem"
  echo "issued $cn"
}

issue zenoh-router "DNS:$HUB_HOST,DNS:localhost,IP:127.0.0.1"
issue dashboard-cloud "DNS:dashboard-cloud,DNS:localhost"
issue woodside "DNS:woodside"
issue leader-arms "DNS:leader-arms"

# Cloud: router + dashboard + CA, one secret.
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
cp "$D/ca/ca.pem" "$D/certs/zenoh-router/zenoh-router."{pem,key} "$D/certs/dashboard-cloud/dashboard-cloud."{pem,key} "$tmp/"
tar -C "$tmp" -czf - . | base64 -w0 | gh secret set MESH_CLOUD_BUNDLE -R "$REPO"
echo "secret MESH_CLOUD_BUNDLE set on $REPO"

# Desktop leader arms.
here="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$here/deploy/strands/mesh"
rm -rf "$here/deploy/strands/mesh/leader-arms"
cp -r "$D/certs/leader-arms" "$here/deploy/strands/mesh/leader-arms"
chmod 700 "$here/deploy/strands/mesh/leader-arms"
echo "leader-arms certs in deploy/strands/mesh/leader-arms"

# Pi: certs + env pointing at the cloud hub.
ssh "$PI_SSH" 'umask 077; mkdir -p ~/.strands-mesh && rm -rf ~/.strands-mesh/woodside'
scp -qr "$D/certs/woodside" "$PI_SSH:.strands-mesh/woodside"
ssh "$PI_SSH" HUB_HOST="$HUB_HOST" bash -s <<'PI'
set -euo pipefail
chmod 700 ~/.strands-mesh ~/.strands-mesh/woodside; chmod 600 ~/.strands-mesh/woodside/woodside.key
f=~/strands-control.env
cp "$f" "$f.bak-$(date +%s)"
sed -i -e '/^STRANDS_MESH_LOCAL_DEV=/d' -e '/^STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=/d' \
  -e '/^ZENOH_CONNECT=/d' -e '/^STRANDS_MESH_AUTH_MODE=/d' -e '/^STRANDS_MESH_TLS_/d' \
  -e '/^STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=/d' -e '/^STRANDS_ROBOT_COMMAND_ALLOW=/d' "$f"
cat >> "$f" <<EOF2
# Mesh (cloud hub, mTLS) - written by deploy/cloud/mesh-certs.sh
ZENOH_CONNECT=tls/$HUB_HOST:7447
STRANDS_MESH_AUTH_MODE=mtls
STRANDS_MESH_TLS_CA=$HOME/.strands-mesh/woodside/ca.pem
STRANDS_MESH_TLS_CERT=$HOME/.strands-mesh/woodside/woodside.pem
STRANDS_MESH_TLS_KEY=$HOME/.strands-mesh/woodside/woodside.key
STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1
STRANDS_ROBOT_COMMAND_ALLOW=teleop_receive@dashboard-cloud
EOF2
echo "Pi: ~/strands-control.env now dials tls/$HUB_HOST:7447 (backup kept)"
PI
echo "done. Next: merge the PR, then restart the Pi peer and the desktop leader."
