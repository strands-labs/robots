"""Train the whole-body bed-reach policy (free-base Inspire-hand G1) with rsl_rl PPO.

Self-contained: builds the manager-based env directly and drives rsl-rl-lib's OnPolicyRunner.
Calls Isaac Lab's deprecation shim so the agent cfg works with the bundled rsl-rl-lib 5.x
(no KeyError: 'class_name'). On finish, exports the policy to JIT + ONNX for deployment.

Run (headless on the Spark GPU):
  cd ~/workspaces/git/IsaacLab && export LD_PRELOAD=$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1 \
    && ./isaaclab.sh -p ~/workspaces/git/robots/examples/isaac_bed_making/rl/train.py \
       --headless --num_envs 2048 --max_iterations 1500
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the bed-reach G1 policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel envs.")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO iterations.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--run_name", type=str, default="", help="Suffix for the run directory.")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Warm-start: load this model_*.pt checkpoint before training (continue/fine-tune).")
parser.add_argument("--robot", type=str, default="g1_edu", choices=["g1_edu", "g1_29dof"],
                    help="g1_edu = 23-DOF EDU (issue #3, transfer-valid); g1_29dof = issue #2 reference.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- everything below runs after the app is live ----
import importlib.metadata as metadata  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.agents import BedReachPPORunnerCfg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    # Select the robot variant: the 23-DOF EDU (issue #3, transfer-valid) or the 29-DOF reference.
    if args_cli.robot == "g1_edu":
        from rl.bed_reach_env_cfg_g1edu import BedReachEduEnvCfg as EnvCfg
        experiment_name = "bed_reach_g1edu"
    else:
        from rl.bed_reach_env_cfg import BedReachEnvCfg as EnvCfg
        experiment_name = "bed_reach_g1"

    env_cfg = EnvCfg()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    agent_cfg = BedReachPPORunnerCfg()
    agent_cfg.experiment_name = experiment_name
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    # Convert the deprecated `policy` cfg -> new actor/critic schema for rsl-rl-lib >= 4.0.
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = stamp if not args_cli.run_name else f"{stamp}_{args_cli.run_name}"
    log_dir = os.path.join(HERE, "logs", agent_cfg.experiment_name, run)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[bed-reach] logging to: {log_dir}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    agent_cfg.device = str(env.unwrapped.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print_dict(agent_cfg.to_dict(), nesting=0)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.resume_from is not None:
        print(f"[bed-reach] warm-starting from checkpoint: {args_cli.resume_from}")
        runner.load(args_cli.resume_from)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Export the final policy for deployment in the demo.
    export_dir = os.path.join(log_dir, "exported")
    runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
    runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
    print(f"[bed-reach] exported policy to: {export_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    # Isaac's replicator orchestrator can hang on a clean close; force exit after artifacts land.
    os._exit(0)
