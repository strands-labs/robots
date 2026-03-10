#!/usr/bin/env python3
"""
Sample 06 — Thor + Isaac Sim Tandem: GPU physics + RTX rendering.

Demonstrates the most powerful strands-robots configuration: Newton physics
running on Thor (AGX Thor, 132GB sm_110) with Isaac Sim rendering on EC2
(L40S 48GB). Communication via Zenoh or the ZMQ subprocess bridge.

Architecture:
    Thor (AGX Thor)          Zenoh          EC2 (L40S)
     Newton physics    <=============>  Isaac Sim render
     4096 parallel env                  RTX ray tracing
     Policy inference                   Photorealistic frames
     7 solver backends                  USD scene management

The ZMQ bridge (strands_robots.isaac.isaac_sim_bridge) enables cross-runtime
communication between Newton's Python environment and Isaac Sim's python.sh.

Level: 3 (Advanced)
Hardware: Thor + EC2 L40S (or two GPUs on same machine)

Usage:
    # On Thor (physics server):
    python samples/06_raytraced_training/thor_isaac_tandem.py --role physics --device cuda:0

    # On EC2 (rendering server):
    /opt/IsaacSim/python.sh samples/06_raytraced_training/thor_isaac_tandem.py --role render

    # Single-machine demo (both on same GPU):
    python samples/06_raytraced_training/thor_isaac_tandem.py --role demo
"""

from __future__ import annotations

import argparse
import time


def run_physics_server(
    num_envs: int = 4096,
    solver: str = "featherstone",
    device: str = "cuda:0",
    num_steps: int = 1000,
) -> dict:
    """Run Newton physics server (Thor side of the tandem).

    In production, this sends state updates to the Isaac Sim render server
    via Zenoh or ZMQ. Here we demonstrate the physics pipeline that would
    be the data source for the rendering server.

    Args:
        num_envs: Parallel environments.
        solver: Newton solver.
        device: CUDA device.
        num_steps: Steps to run.

    Returns:
        Physics server results.
    """
    from strands_robots.newton import NewtonBackend, NewtonConfig

    print("\n🔧 Thor Physics Server")
    print("=" * 60)
    print(f"  Solver:  {solver}")
    print(f"  Envs:    {num_envs:,}")
    print(f"  Device:  {device}")
    print("  Role:    Physics computation + Policy inference")
    print()

    config = NewtonConfig(num_envs=num_envs, solver=solver, device=device)
    backend = NewtonBackend(config)
    backend.create_world(gravity=(0, 0, -9.81))

    # Add robot for locomotion
    backend.add_robot("so100")
    rep = backend.replicate(num_envs=num_envs)
    print(f"  ✅ Scene replicated: {rep.get('env_info', {}).get('bodies_total', '?')} bodies")

    # Warmup
    for _ in range(10):
        backend.step()

    # Simulate and collect state snapshots (these would be sent to render server)
    print(f"  📊 Running {num_steps} steps...")
    state_snapshots = []
    t0 = time.perf_counter()

    for step in range(num_steps):
        backend.step()

        # Every 100 steps, capture state for rendering
        if step % 100 == 0:
            obs = backend.get_observation("so100")
            robot_obs = obs.get("observations", {}).get("so100", {})
            state_snapshots.append({
                "step": step,
                "joint_positions": robot_obs.get("joint_positions"),
                "sim_time": backend._sim_time,
            })

    elapsed = time.perf_counter() - t0
    throughput = int(num_steps * num_envs / elapsed)

    print("\n  ── Physics Server Results ──")
    print(f"  Throughput:  {throughput:,} env-steps/s")
    print(f"  Elapsed:     {elapsed:.3f}s")
    print(f"  Snapshots:   {len(state_snapshots)} (for rendering)")
    print(f"  Sim time:    {backend._sim_time:.4f}s")

    backend.destroy()

    return {
        "throughput": throughput,
        "elapsed": elapsed,
        "snapshots": len(state_snapshots),
    }


def run_render_server() -> dict:
    """Run Isaac Sim rendering server (EC2 side of the tandem).

    In production, this receives state updates from the physics server
    and renders photorealistic frames. Here we demonstrate the rendering
    pipeline that would consume physics data.

    Returns:
        Render server results.
    """
    print("\n🖼️ EC2 Rendering Server")
    print("=" * 60)
    print("  GPU:     L40S 48GB")
    print("  Role:    RTX ray tracing + photorealistic rendering")
    print()

    try:
        from strands_robots.isaac import IsaacSimBackend, is_isaac_sim_available

        if not is_isaac_sim_available():
            print("  ⚠️ Isaac Sim not available on this machine")
            print("  💡 Run on isaac-sim.yml EC2 runner for full demo")
            return {"available": False}

        # In the tandem setup, the render server:
        # 1. Maintains a USD scene matching the physics world
        # 2. Receives state updates (joint positions, object poses)
        # 3. Applies state to USD prims
        # 4. Renders RTX frames
        # 5. Optionally streams frames back or saves to disk

        backend = IsaacSimBackend()
        backend.create_world()
        backend.add_robot("so100")

        # Render a few frames to show the pipeline
        for i in range(10):
            backend.step()
            if i % 3 == 0:
                frame = backend.render(width=640, height=480)
                print(f"  Frame {i}: rendered ({type(frame).__name__})")

        backend.destroy()
        return {"available": True, "frames_rendered": 4}

    except ImportError as e:
        print(f"  ⚠️ Isaac Sim import error: {e}")
        return {"available": False, "error": str(e)}


def run_zmq_bridge_demo() -> dict:
    """Demonstrate the ZMQ subprocess bridge for cross-runtime communication.

    The IsaacSimBridge enables communication between strands-robots' Python
    environment (where Newton runs) and Isaac Sim's python.sh runtime.
    Messages use msgpack-numpy serialization with ~0.3ms overhead.

    Returns:
        Bridge demo results.
    """
    print("\n🌉 ZMQ Subprocess Bridge Demo")
    print("=" * 60)
    print("  Protocol:    ZMQ REQ/REP")
    print("  Serializer:  msgpack + numpy")
    print("  Overhead:    ~0.3ms per message")
    print()

    try:
        from strands_robots.isaac import IsaacSimBridgeClient  # noqa: F401

        # In production, the bridge server runs inside Isaac Sim's python.sh:
        #   /opt/IsaacSim/python.sh -c "
        #     from strands_robots.isaac import IsaacSimBridgeServer
        #     server = IsaacSimBridgeServer(port=5556)
        #     server.serve_forever()
        #   "

        # And the client connects from our Python:
        #   client = IsaacSimBridgeClient(host="ec2-instance", port=5556)
        #   result = client.call("create_world", gravity=(0, 0, -9.81))
        #   result = client.call("add_robot", name="so100")
        #   result = client.call("step")
        #   obs = client.call("get_observation", robot_name="so100")

        print("  ✅ IsaacSimBridgeClient imported successfully")
        print()
        print("  Bridge usage pattern:")
        print("    # Server (Isaac Sim python.sh):")
        print('    server = IsaacSimBridgeServer(port=5556)')
        print('    server.serve_forever()')
        print()
        print("    # Client (strands-robots Python):")
        print('    client = IsaacSimBridgeClient(host="ec2-ip", port=5556)')
        print('    client.call("create_world")')
        print('    client.call("add_robot", name="so100")')
        print('    client.call("step")')
        print('    obs = client.call("get_observation", robot_name="so100")')

        return {"bridge_available": True}

    except ImportError as e:
        print(f"  ⚠️ Bridge not available: {e}")
        return {"bridge_available": False, "error": str(e)}


def run_tandem_demo(device: str = "cuda:0") -> dict:
    """Run a single-machine tandem demo showing the full pipeline.

    Runs Newton physics and reports what would be sent to Isaac Sim
    for rendering. This is a self-contained demo that shows the
    data flow without requiring two separate machines.

    Args:
        device: CUDA device.

    Returns:
        Demo results.
    """
    print("\n🔄 Tandem Demo (Single Machine)")
    print("=" * 60)
    print("  Simulating: Thor (physics) + EC2 (render)")
    print()

    # Phase 1: Physics on "Thor"
    print("  ── Phase 1: Physics (Newton) ──")
    physics = run_physics_server(
        num_envs=1024,  # Smaller for single-machine demo
        solver="featherstone",
        device=device,
        num_steps=500,
    )

    # Phase 2: Bridge info
    print()
    bridge = run_zmq_bridge_demo()

    # Phase 3: Summary
    print("\n\n🏁 Tandem Summary")
    print("=" * 60)
    print(f"  Physics throughput: {physics['throughput']:,} env-steps/s")
    print(f"  State snapshots:    {physics['snapshots']}")
    print(f"  Bridge available:   {bridge.get('bridge_available', False)}")
    print()
    print("  In production tandem mode:")
    print("    1. Thor runs Newton physics at ~2M env-steps/s")
    print("    2. State snapshots sent via Zenoh/ZMQ every 100 steps")
    print("    3. EC2 renders RTX frames from received state")
    print("    4. Frames streamed back for visualization/recording")

    return {"physics": physics, "bridge": bridge}


def main():
    parser = argparse.ArgumentParser(
        description="Sample 06: Thor + Isaac Sim Tandem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--role",
        default="demo",
        choices=["physics", "render", "bridge", "demo"],
        help="Server role (default: demo = single-machine)",
    )
    parser.add_argument("--envs", type=int, default=4096, help="Parallel environments")
    parser.add_argument("--solver", default="featherstone", help="Newton solver")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--steps", type=int, default=1000, help="Physics steps")

    args = parser.parse_args()

    if args.role == "physics":
        run_physics_server(
            num_envs=args.envs,
            solver=args.solver,
            device=args.device,
            num_steps=args.steps,
        )
    elif args.role == "render":
        run_render_server()
    elif args.role == "bridge":
        run_zmq_bridge_demo()
    else:
        run_tandem_demo(device=args.device)


if __name__ == "__main__":
    main()
