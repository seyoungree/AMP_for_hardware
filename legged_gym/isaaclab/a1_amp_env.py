from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Iterable

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
from rsl_rl.env import VecEnv


COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515], dtype=torch.float)
HIP_OFFSETS = torch.tensor(
    [
        [0.183, 0.047, 0.0],
        [0.183, -0.047, 0.0],
        [-0.183, 0.047, 0.0],
        [-0.183, -0.047, 0.0],
    ],
    dtype=torch.float,
) + COM_OFFSET

A1_JOINT_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


def _walk_path(root: Any, path: Iterable[Any]) -> Any:
    node = root
    for key in path:
        if node is None:
            return None
        if isinstance(node, dict):
            node = node.get(key)
            continue
        if isinstance(key, int) and isinstance(node, (list, tuple)):
            node = node[key]
            continue
        if hasattr(node, key):
            node = getattr(node, key)
            continue
        try:
            node = node[key]
            continue
        except Exception:
            return None
    return node


def load_a1_dof_limits_from_urdf(
    urdf_path: str | None = None,
    joint_order: list[str] | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if urdf_path is None:
        urdf_path = os.path.join(
            LEGGED_GYM_ROOT_DIR, "resources", "robots", "a1", "urdf", "a1.urdf"
        )
    if joint_order is None:
        joint_order = A1_JOINT_ORDER

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limits_by_name: dict[str, tuple[float, float]] = {}

    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        if lower is None or upper is None:
            continue
        limits_by_name[name] = (float(lower), float(upper))

    dof_limits = torch.zeros(len(joint_order), 2, dtype=torch.float, device=device)
    for idx, joint_name in enumerate(joint_order):
        lower, upper = limits_by_name[joint_name]
        mid = (lower + upper) / 2.0
        span = upper - lower
        dof_limits[idx, 0] = mid - 0.5 * span * A1AMPCfg.rewards.soft_dof_pos_limit
        dof_limits[idx, 1] = mid + 0.5 * span * A1AMPCfg.rewards.soft_dof_pos_limit
    return dof_limits


class A1IsaacLabAmpAdapter(VecEnv):
    """A Unitree A1 Isaac Lab adapter that reproduces this repo's AMP observation layout.

    It targets the official Isaac Lab A1 velocity tasks and converts their environment
    state into the old AMP observation interface used by `AMPOnPolicyRunner`.
    """

    def __init__(self, env: Any):
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.device = getattr(self.unwrapped, "device", "cpu")
        self.num_envs = self.unwrapped.num_envs
        self.num_actions = A1AMPCfg.env.num_actions
        self.num_obs = A1AMPCfg.env.num_observations
        self.num_privileged_obs = A1AMPCfg.env.num_privileged_obs
        self.include_history_steps = A1AMPCfg.env.include_history_steps
        self.dt = getattr(self.unwrapped, "step_dt", None) or (
            A1AMPCfg.sim.dt * A1AMPCfg.control.decimation
        )
        self.max_episode_length = getattr(
            self.unwrapped,
            "max_episode_length",
            int(A1AMPCfg.env.episode_length_s / self.dt),
        )

        self.obs_scales = A1AMPCfg.normalization.obs_scales
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
            ],
            device=self.device,
            dtype=torch.float,
        )
        self.default_dof_pos = torch.tensor(
            [
                A1AMPCfg.init_state.default_joint_angles[name]
                for name in A1_JOINT_ORDER
            ],
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)
        self.dof_pos_limits = load_a1_dof_limits_from_urdf(device=self.device)

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.privileged_obs_buf = torch.zeros(
            self.num_envs, self.num_privileged_obs, device=self.device
        )
        self._amp_obs_buf = torch.zeros(self.num_envs, 43, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = getattr(
            self.unwrapped,
            "episode_length_buf",
            torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
        )
        self.extras: dict[str, Any] = {}
        self._last_actions = torch.zeros(
            self.num_envs, self.num_actions, device=self.device, dtype=torch.float
        )

    def _lookup_tensor(
        self,
        obs: Any,
        extras: dict[str, Any],
        *paths: tuple[Any, ...],
    ) -> torch.Tensor | None:
        roots = [extras, obs, self.unwrapped, self.env]
        for root in roots:
            for path in paths:
                value = _walk_path(root, path)
                if isinstance(value, torch.Tensor):
                    return value.to(self.device)
        return None

    def _get_joint_pos(self, obs, extras):
        tensor = self._lookup_tensor(
            obs,
            extras,
            ("joint_pos",),
            ("policy", "joint_pos"),
            ("critic", "joint_pos"),
            ("observations", "joint_pos"),
            ("scene", "robot", "data", "joint_pos"),
        )
        if tensor is None:
            raise KeyError("Could not find joint positions in the Isaac Lab A1 environment.")
        return tensor[:, : self.num_actions]

    def _get_joint_vel(self, obs, extras):
        tensor = self._lookup_tensor(
            obs,
            extras,
            ("joint_vel",),
            ("policy", "joint_vel"),
            ("critic", "joint_vel"),
            ("observations", "joint_vel"),
            ("scene", "robot", "data", "joint_vel"),
        )
        if tensor is None:
            raise KeyError("Could not find joint velocities in the Isaac Lab A1 environment.")
        return tensor[:, : self.num_actions]

    def _get_base_lin_vel(self, obs, extras):
        tensor = self._lookup_tensor(
            obs,
            extras,
            ("base_lin_vel",),
            ("policy", "base_lin_vel"),
            ("critic", "base_lin_vel"),
            ("scene", "robot", "data", "root_lin_vel_b"),
        )
        if tensor is None:
            raise KeyError("Could not find base linear velocity in the Isaac Lab A1 environment.")
        return tensor[:, :3]

    def _get_base_ang_vel(self, obs, extras):
        tensor = self._lookup_tensor(
            obs,
            extras,
            ("base_ang_vel",),
            ("policy", "base_ang_vel"),
            ("critic", "base_ang_vel"),
            ("scene", "robot", "data", "root_ang_vel_b"),
        )
        if tensor is None:
            raise KeyError("Could not find base angular velocity in the Isaac Lab A1 environment.")
        return tensor[:, :3]

    def _get_projected_gravity(self, obs, extras):
        tensor = self._lookup_tensor(
            obs,
            extras,
            ("projected_gravity",),
            ("policy", "projected_gravity"),
            ("critic", "projected_gravity"),
            ("scene", "robot", "data", "projected_gravity_b"),
        )
        if tensor is not None:
            return tensor[:, :3]

        policy_obs = self._lookup_tensor(obs, extras, ("policy",), ("obs",), ("observations", "policy"))
        if policy_obs is not None and policy_obs.shape[-1] >= 9:
            return policy_obs[:, 6:9]
        raise KeyError("Could not find projected gravity in the Isaac Lab A1 environment.")

    def _get_commands(self, obs, extras):
        tensor = self._lookup_tensor(
            obs,
            extras,
            ("commands",),
            ("velocity_commands",),
            ("command",),
            ("policy", "commands"),
            ("critic", "commands"),
            ("command_manager", "_terms", "base_velocity", "command"),
        )
        if tensor is not None:
            return tensor[:, :3]

        policy_obs = self._lookup_tensor(obs, extras, ("policy",), ("obs",), ("observations", "policy"))
        if policy_obs is not None and policy_obs.shape[-1] >= 3:
            return policy_obs[:, :3]
        return torch.zeros(self.num_envs, 3, device=self.device)

    def _get_root_height(self, obs, extras):
        tensor = self._lookup_tensor(
            obs,
            extras,
            ("root_pos_w",),
            ("scene", "robot", "data", "root_pos_w"),
        )
        if tensor is None:
            raise KeyError("Could not find root position in the Isaac Lab A1 environment.")
        return tensor[:, 2:3]

    def _get_actions(self, extras):
        tensor = self._lookup_tensor(extras, extras, ("actions",), ("action",), ("policy_actions",))
        if tensor is not None:
            return tensor[:, : self.num_actions]
        return self._last_actions

    def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = torch.sqrt(
            l_up**2 + l_low**2 + 2 * l_up * l_low * torch.cos(theta_knee)
        )
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
        return torch.stack([off_x, off_y, off_z], dim=-1)

    def foot_positions_in_base_frame(self, foot_angles):
        foot_positions = torch.zeros_like(foot_angles)
        hip_offsets = HIP_OFFSETS.reshape(12).to(self.device)
        for i in range(4):
            foot_positions[:, i * 3 : i * 3 + 3].copy_(
                self.foot_position_in_hip_frame(
                    foot_angles[:, i * 3 : i * 3 + 3], l_hip_sign=(-1) ** i
                )
            )
        return foot_positions + hip_offsets

    def _compute_amp_buffers(self, obs, extras):
        joint_pos = self._get_joint_pos(obs, extras)
        joint_vel = self._get_joint_vel(obs, extras)
        base_lin_vel = self._get_base_lin_vel(obs, extras)
        base_ang_vel = self._get_base_ang_vel(obs, extras)
        projected_gravity = self._get_projected_gravity(obs, extras)
        commands = self._get_commands(obs, extras)
        actions = self._get_actions(extras)
        root_height = self._get_root_height(obs, extras)
        foot_pos = self.foot_positions_in_base_frame(joint_pos)

        privileged_obs = torch.cat(
            (
                base_lin_vel * self.obs_scales.lin_vel,
                base_ang_vel * self.obs_scales.ang_vel,
                projected_gravity,
                commands * self.commands_scale,
                (joint_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                joint_vel * self.obs_scales.dof_vel,
                actions,
            ),
            dim=-1,
        )
        policy_obs = privileged_obs[:, 6:]
        amp_obs = torch.cat(
            (joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, root_height), dim=-1
        )
        return policy_obs, privileged_obs, amp_obs

    def reset(self):
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            obs, extras = reset_out
        else:
            obs, extras = reset_out, {}
        self.obs_buf, self.privileged_obs_buf, self._amp_obs_buf = self._compute_amp_buffers(obs, extras)
        self.extras = extras
        return self.obs_buf, self.privileged_obs_buf

    def step(self, actions: torch.Tensor):
        self._last_actions = actions.to(self.device)
        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        self.obs_buf, self.privileged_obs_buf, self._amp_obs_buf = self._compute_amp_buffers(obs, extras)
        self.rew_buf = rewards.to(self.device)
        self.reset_buf = (terminated | truncated).to(self.device)
        self.extras = extras

        reset_env_ids = extras.get("reset_env_ids")
        if reset_env_ids is None:
            reset_env_ids = torch.nonzero(self.reset_buf, as_tuple=False).flatten()
        else:
            reset_env_ids = reset_env_ids.to(self.device)

        terminal_amp_states = extras.get("terminal_amp_obs")
        if terminal_amp_states is None:
            terminal_amp_states = self._amp_obs_buf[reset_env_ids]
        else:
            terminal_amp_states = terminal_amp_states.to(self.device)

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            reset_env_ids,
            terminal_amp_states,
        )

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self) -> torch.Tensor | None:
        return self.privileged_obs_buf

    def get_amp_observations(self) -> torch.Tensor:
        return self._amp_obs_buf
