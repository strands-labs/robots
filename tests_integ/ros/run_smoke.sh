#!/bin/bash
# ROS 2 smoke test in a real ROS distro (not run in CI). From the checkout root:
#   docker run --rm -v "$PWD":/src:ro ros:jazzy-ros-base bash /src/tests_integ/ros/run_smoke.sh
#   docker run --rm -v "$PWD":/src:ro ros:humble-ros-base bash /src/tests_integ/ros/run_smoke.sh
# smoke_robot.py plays a fake ROS robot; smoke_client.py drives it through strands-robots and prints PASS/FAIL.
set -e
source /opt/ros/$ROS_DISTRO/setup.bash
command -v pip3 >/dev/null || (apt-get update -qq && apt-get install -y -qq python3-pip >/dev/null)
mkdir -p /tmp/sr && tar -C /src --exclude=.venv --exclude=.git -cf - . | tar -C /tmp/sr -xf - && cd /tmp/sr
python3 -m pip install -q --break-system-packages --no-deps . >/dev/null 2>&1 || python3 -m pip install -q --no-deps . >/dev/null 2>&1 || { echo "pip cannot build here (old pip): using PYTHONPATH"; export PYTHONPATH=/tmp/sr:$PYTHONPATH; }
python3 -m pip install -q --break-system-packages robot_descriptions >/dev/null 2>&1 || python3 -m pip install -q robot_descriptions >/dev/null
cd /
python3 -c "import numpy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'rmw', __import__('os').environ.get('RMW_IMPLEMENTATION'))"
python3 /src/tests_integ/ros/smoke_robot.py & ROBOT=$!
sleep 2
python3 /src/tests_integ/ros/smoke_client.py; RC=$?
kill $ROBOT
exit $RC
