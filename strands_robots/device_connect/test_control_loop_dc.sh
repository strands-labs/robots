#!/usr/bin/env bash
# test_control_loop_dc.sh — End-to-end test: control loop + Zenoh event listener
#
# Verifies that Robot("so100") publishes Device Connect events over Zenoh
# while a mock-policy control loop is running.
#
# Usage:
#   bash strands_robots/device_connect/test_control_loop_dc.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export DEVICE_CONNECT_ALLOW_INSECURE=true

EVENTS_LOG=$(mktemp /tmp/zenoh_events_XXXX.log)
LOOP_LOG=$(mktemp /tmp/control_loop_XXXX.log)
LISTENER_PID=""

cleanup() {
    [ -n "$LISTENER_PID" ] && kill "$LISTENER_PID" 2>/dev/null || true
    echo ""
    echo "Logs:"
    echo "  Events:       $EVENTS_LOG"
    echo "  Control loop: $LOOP_LOG"
}
trap cleanup EXIT

# ── 1. Install dependencies ────────────────────────────────────────────
echo "==> Installing device-connect-edge..."
pip install -e "$WORKSPACE_ROOT/device-connect/packages/device-connect-edge" -q

echo "==> Installing device-connect-agent-tools..."
pip install -e "$WORKSPACE_ROOT/device-connect/packages/device-connect-agent-tools[strands]" -q

echo "==> Installing strands-robots[sim]..."
pip install -e "$REPO_ROOT[sim]" -q

echo "==> All dependencies installed."
echo ""

# ── 2. Start Zenoh listener ────────────────────────────────────────────
echo "==> Starting Zenoh event listener..."
python3 -c "
import json, time, zenoh

def on_sample(sample):
    try:
        data = json.loads(sample.payload.to_bytes().decode())
    except Exception:
        data = str(sample.payload.to_bytes().decode()[:200])
    print(f'[{time.strftime(\"%H:%M:%S\")}] {sample.key_expr}: {json.dumps(data, default=str)}', flush=True)

session = zenoh.open(zenoh.Config())
sub = session.declare_subscriber('device-connect/**', on_sample)
print('LISTENER_READY', flush=True)
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    sub.undeclare()
    session.close()
" > "$EVENTS_LOG" 2>&1 &
LISTENER_PID=$!

# Wait for listener to be ready
for i in $(seq 1 30); do
    grep -q "LISTENER_READY" "$EVENTS_LOG" 2>/dev/null && break
    sleep 0.2
done
echo "    Listener PID: $LISTENER_PID"
echo ""

# ── 3. Run the control loop ────────────────────────────────────────────
echo "==> Running control loop (200 steps @ 50Hz)..."
python3 -c "
import os, sys, time
os.environ.setdefault('MUJOCO_GL', 'egl')

from strands_robots.factory import Robot
from strands_robots.policies import create_policy

robot = Robot('so100')
# Wait for DC runtime to connect and start periodic publishers
time.sleep(3)

policy = create_policy('mock')
for step in range(200):
    obs = robot.get_observation()
    action = policy.get_actions_sync(obs, instruction='pick up the cube')
    robot.apply_action(action)
    if step % 50 == 0:
        print(f'  Step {step}/200', flush=True)

print('Control loop done.', flush=True)
robot.cleanup()
print('Cleanup complete.', flush=True)
" 2>&1 | tee "$LOOP_LOG"

# Give trailing events a moment to arrive
sleep 2

# ── 4. Stop the listener ───────────────────────────────────────────────
kill "$LISTENER_PID" 2>/dev/null || true
wait "$LISTENER_PID" 2>/dev/null || true
LISTENER_PID=""

# ── 5. Validate captured events ────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ZENOH EVENT SUMMARY"
echo "============================================================"

TOTAL=$(grep -c '^\[' "$EVENTS_LOG" 2>/dev/null || echo 0)
STATE_UPDATES=$(grep -c 'event/stateUpdate' "$EVENTS_LOG" 2>/dev/null || echo 0)
OBS_UPDATES=$(grep -c 'event/observationUpdate' "$EVENTS_LOG" 2>/dev/null || echo 0)
PRESENCE=$(grep -c '/presence' "$EVENTS_LOG" 2>/dev/null || echo 0)
HEARTBEATS=$(grep -c '/heartbeat' "$EVENTS_LOG" 2>/dev/null || echo 0)

printf "  %-25s %s\n" "stateUpdate events:" "$STATE_UPDATES"
printf "  %-25s %s\n" "observationUpdate events:" "$OBS_UPDATES"
printf "  %-25s %s\n" "presence events:" "$PRESENCE"
printf "  %-25s %s\n" "heartbeat events:" "$HEARTBEATS"
printf "  %-25s %s\n" "TOTAL:" "$TOTAL"
echo ""

# Show a sample observationUpdate with joint data
SAMPLE_OBS=$(grep 'event/observationUpdate' "$EVENTS_LOG" | tail -1)
if [ -n "$SAMPLE_OBS" ]; then
    echo "  Sample observationUpdate:"
    echo "  $SAMPLE_OBS" | python3 -c "
import sys, json
line = sys.stdin.read().strip()
payload = line.split(': ', 1)[1]
data = json.loads(payload)
params = data.get('params', {})
print(f\"    robot:     {params.get('robot_name')}\")
print(f\"    sim_time:  {params.get('sim_time')}\")
print(f\"    step:      {params.get('step_count')}\")
joints = params.get('joints', {})
for name, val in joints.items():
    print(f\"    {name:>15s}: {val:+.6f} rad\")
" 2>/dev/null || echo "    (could not parse sample)"
    echo ""
fi

# ── 6. Assert minimum thresholds ───────────────────────────────────────
PASS=true

if [ "$TOTAL" -lt 10 ]; then
    echo "FAIL: Expected >= 10 total events, got $TOTAL"
    PASS=false
fi

if [ "$STATE_UPDATES" -lt 5 ]; then
    echo "FAIL: Expected >= 5 stateUpdate events, got $STATE_UPDATES"
    PASS=false
fi

if [ "$PRESENCE" -lt 1 ]; then
    echo "FAIL: Expected >= 1 presence event, got $PRESENCE"
    PASS=false
fi

if [ "$HEARTBEATS" -lt 1 ]; then
    echo "FAIL: Expected >= 1 heartbeat event, got $HEARTBEATS"
    PASS=false
fi

if [ "$OBS_UPDATES" -lt 5 ]; then
    echo "FAIL: Expected >= 5 observationUpdate events, got $OBS_UPDATES"
    PASS=false
fi

# Check no "Failed to publish" in control loop output
PUBLISH_ERRORS=$(grep -c "Failed to publish" "$LOOP_LOG" 2>/dev/null || true)
PUBLISH_ERRORS="${PUBLISH_ERRORS:-0}"
if [ "$PUBLISH_ERRORS" -gt 0 ]; then
    echo "FAIL: Found $PUBLISH_ERRORS 'Failed to publish' errors (missing cleanup?)"
    PASS=false
fi

if [ "$PASS" = true ]; then
    echo "============================================================"
    echo "  ALL CHECKS PASSED"
    echo "============================================================"
    exit 0
else
    echo ""
    echo "============================================================"
    echo "  SOME CHECKS FAILED — see logs above"
    echo "============================================================"
    exit 1
fi
