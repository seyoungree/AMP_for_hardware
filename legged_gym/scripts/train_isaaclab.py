from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
RSL_RL_ROOT = os.path.join(REPO_ROOT, "rsl_rl")
for path in (REPO_ROOT, RSL_RL_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from legged_gym.envs.a1.a1_amp_config import A1AMPCfg, A1AMPCfgPPO


def _parse_args():
    parser = argparse.ArgumentParser(description="Train AMP on an Isaac Lab Unitree A1 task.")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-A1-v0")
    parser.add_argument("--num_envs", type=int, default=A1AMPCfg.env.num_envs)
    parser.add_argument("--seed", type=int, default=A1AMPCfgPPO.seed)
    parser.add_argument("--max_iterations", type=int, default=A1AMPCfgPPO.runner.max_iterations)
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--experiment_name", type=str, default=A1AMPCfgPPO.runner.experiment_name)
    parser.add_argument("--run_name", type=str, default="isaaclab_a1_amp")
    parser.add_argument("--video", action="store_true", help="Enable camera rendering.")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(), AppLauncher


def _set_cfg_attr(cfg, dotted_name: str, value):
    node = cfg
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        if not hasattr(node, part):
            return False
        node = getattr(node, part)
    if not hasattr(node, parts[-1]):
        return False
    setattr(node, parts[-1], value)
    return True


args, AppLauncher = _parse_args()
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry

from legged_gym.isaaclab import A1IsaacLabAmpAdapter
from legged_gym.utils import class_to_dict
from rsl_rl.runners import AMPOnPolicyRunner


def train():
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    _set_cfg_attr(env_cfg, "scene.num_envs", args.num_envs)
    _set_cfg_attr(env_cfg, "seed", args.seed)

    env = gym.make(args.task, cfg=env_cfg)
    amp_env = A1IsaacLabAmpAdapter(env)

    train_cfg = A1AMPCfgPPO()
    train_cfg.seed = args.seed
    train_cfg.runner.max_iterations = args.max_iterations
    train_cfg.runner.experiment_name = args.experiment_name
    train_cfg.runner.run_name = args.run_name

    log_root = os.path.join(REPO_ROOT, "logs", train_cfg.runner.experiment_name)
    log_dir = os.path.join(
        log_root,
        datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name,
    )

    runner = AMPOnPolicyRunner(
        amp_env,
        class_to_dict(train_cfg),
        log_dir=log_dir,
        device=args.rl_device,
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    try:
        train()
    finally:
        simulation_app.close()
