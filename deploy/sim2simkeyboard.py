import argparse
import os
import select
import sys
import termios
import threading
import tty
import numpy as np
import torch
import mujoco
import mujoco.viewer


class Config:
    sim_dt = 0.005
    policy_hz = 33
    policy_dt = 1.0 / policy_hz

    default_dof_pos = np.array([
        0.0, 0.9, -1.8,
        0.0, 0.9, -1.8,
        0.0, 0.9, -1.8,
        0.0, 0.9, -1.8,
    ], dtype=np.float32)

    obs_scales = {
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "commands": np.array([2.0, 2.0, 0.25], dtype=np.float32),
    }

    action_scale = 0.25
    clip_observations = 100.0
    clip_actions = 100.0

    stabilize_duration = 1.0

    hold_kp = 60.0
    hold_kd = 4.0
    walk_kp = 50.0
    walk_kd = 3.5

    init_base_height = 0.27

    vx_range = (-1.0, 2.0)
    vy_range = (-0.3, 0.3)
    vyaw_range = (-1.57, 1.57)


def quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def compute_projected_gravity(quat_xyzw: np.ndarray) -> np.ndarray:
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return quat_rotate_inverse(quat_xyzw.astype(np.float32), gravity_world)


def quat_to_euler_xyz(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def build_obs(
    projected_gravity: np.ndarray,
    commands: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    last_action: np.ndarray,
    cfg: #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


# =============================================================================
# Config
# =============================================================================

class SimConfig:
    # timing
    sim_dt = 0.005
    policy_hz = 33
    policy_dt = 1.0 / policy_hz

    default_dof_pos = np.array([
         0.0, 0.9, -1.8,   # FL
         0.0, 0.9, -1.8,   # FR
         0.0, 0.9, -1.8,   # RL
         0.0, 0.9, -1.8,   # RR
    ], dtype=np.float32)

    # crouched pose for stand-up
    init_dof_pos = np.array([
         0.0, 1.2, -2.4,
         0.0, 1.2, -2.4,
         0.0, 1.2, -2.4,
         0.0, 1.2, -2.4,
    ], dtype=np.float32)

    # observation scales
    obs_scales = {
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "commands": np.array([2.0, 2.0, 0.25], dtype=np.float32),
    }

    action_scale = 0.25
    clip_observations = 100.0
    clip_actions = 100.0

    standup_duration = 3.0
    stabilize_duration = 1.0

    stand_kp = 50.0
    stand_kd = 3.5
    walk_kp = 40.0
    walk_kd = 3.0
    tau_limit = 60.0

    vx_range = (-1.0, 2.0)
    vy_range = (-0.3, 0.3)
    vyaw_range = (-1.57, 1.57)

    # start slightly above ground, then let PD settle during crouch/standup
    init_base_height = 0.27

    abort_on_fall = True
    abort_base_height = 0.05
    abort_rpy = 1.2


# =============================================================================
# Helpers
# =============================================================================

def quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quat_to_euler_xyz(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def compute_projected_gravity(quat_xyzw: np.ndarray) -> np.ndarray:
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return quat_rotate_inverse(quat_xyzw.astype(np.float32), gravity_world)


def normalize_obs(obs: np.ndarray, clip_value: float = 100.0) -> np.ndarray:
    return np.clip(obs, -clip_value, clip_value)


def build_obs(
    projected_gravity: np.ndarray,
    commands: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    last_action: np.ndarray,
    config: SimConfig,
    obs_dim: int,
) -> np.ndarray:
    # user specified:
    # 42 = projected_gravity(3) + commands(3) + dof_pos_err(12) + dof_vel(12) + last_action(12)
    if obs_dim != 42:
        raise ValueError(f"Expected 42D policy input, got {obs_dim}")

    obs = []
    obs.extend(list(projected_gravity))
    obs.extend(list(commands * config.obs_scales["commands"]))
    obs.extend(list((dof_pos - config.default_dof_pos) * config.obs_scales["dof_pos"]))
    obs.extend(list(dof_vel * config.obs_scales["dof_vel"]))
    obs.extend(list(last_action))
    return np.array(obs, dtype=np.float32)


class KeyboardController:
    def __init__(self, vx_range=(-1.0, 2.0), vy_range=(-0.3, 0.3), vyaw_range=(-1.57, 1.57)):
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vyaw_range = vyaw_range
        self.lock = threading.Lock()
        self.running = True
        self.exit_requested = False
        self.thread = None

        self.vx_step = 0.1
        self.vyaw_step = 0.1

    def get_velocity(self) -> Tuple[float, float, float]:
        with self.lock:
            return self.vx, self.vy, self.vyaw

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        with self.lock:
            self.vx = float(np.clip(vx, self.vx_range[0], self.vx_range[1]))
            self.vy = float(np.clip(vy, self.vy_range[0], self.vy_range[1]))
            self.vyaw = float(np.clip(vyaw, self.vyaw_range[0], self.vyaw_range[1]))

    def keyboard_thread(self) -> None:
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue

                key = sys.stdin.read(1)
                vx, vy, vyaw = self.get_velocity()

                if key in ("w", "W"):
                    vx += self.vx_step
                elif key in ("s", "S"):
                    vx -= self.vx_step
                elif key in ("a", "A"):
                    vyaw += self.vyaw_step
                elif key in ("d", "D"):
                    vyaw -= self.vyaw_step
                elif key == " ":
                    vx, vy, vyaw = 0.0, 0.0, 0.0
                elif key in ("q", "Q"):
                    self.exit_requested = True
                    self.running = False
                    print("\n[Keyboard] Exit requested...")
                    break
                else:
                    continue

                self.set_velocity(vx, vy, vyaw)
                vx, vy, vyaw = self.get_velocity()
                print(
                    f"\r[Command] vx={vx:+.2f} m/s, vy={vy:+.2f} m/s, yaw={vyaw:+.2f} rad/s",
                    end="",
                    flush=True,
                )
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def start(self) -> None:
        self.thread = threading.Thread(target=self.keyboard_thread, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


# =============================================================================
# Sim controller
# =============================================================================

@dataclass
class JointHandles:
    joint_qpos_addrs: List[int]
    joint_dof_addrs: List[int]
    actuator_ids: List[int]


class Sim2SimController:
    JOINT_NAMES = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    ACTUATOR_NAMES = [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]

    def __init__(self, config: SimConfig, xml_path: str, policy_path: str, headless: bool = False):
        self.config = config
        self.headless = headless

        import mujoco
        import mujoco.viewer
        self.mujoco = mujoco
        self.mujoco_viewer = mujoco.viewer

        print(f"Loading MuJoCo model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.sim_dt

        self.handles = self._resolve_handles()

        print(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()
        self.policy_obs_dim = self._infer_policy_input_dim()
        print(f"Detected policy input dimension: {self.policy_obs_dim}")

        self.last_action = np.zeros(12, dtype=np.float32)
        self.qDes = self.config.init_dof_pos.copy()

        self.policy_decimation = max(1, int(round(config.policy_dt / config.sim_dt)))
        self.policy_counter = 0

        self._print_model_info()
        self.reset_pose()

        print("Sim2Sim controller initialized")

    def _resolve_handles(self) -> JointHandles:
        joint_qpos_addrs = []
        joint_dof_addrs = []
        actuator_ids = []

        for joint_name, actuator_name in zip(self.JOINT_NAMES, self.ACTUATOR_NAMES):
            joint_id = self.model.joint(joint_name).id
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            dof_addr = int(self.model.jnt_dofadr[joint_id])
            actuator_id = int(self.model.actuator(actuator_name).id)

            joint_qpos_addrs.append(qpos_addr)
            joint_dof_addrs.append(dof_addr)
            actuator_ids.append(actuator_id)

        return JointHandles(
            joint_qpos_addrs=joint_qpos_addrs,
            joint_dof_addrs=joint_dof_addrs,
            actuator_ids=actuator_ids,
        )

    def _infer_policy_input_dim(self) -> int:
        state = self.policy.state_dict()
        for key, value in state.items():
            if key.endswith("weight") and getattr(value, "ndim", 0) == 2:
                return int(value.shape[1])
        raise RuntimeError("Could not infer policy input dimension")

    def _print_model_info(self) -> None:
        print("=" * 70)
        print("MuJoCo model info")
        print("=" * 70)
        print(f"nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")
        print(f"timestep={self.model.opt.timestep}")
        print("=" * 70)

    def reset_pose(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        # free base in MuJoCo qpos layout: x y z qw qx qy qz
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = float(self.config.init_base_height)
        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0

        for i, qpos_addr in enumerate(self.handles.joint_qpos_addrs):
            self.data.qpos[qpos_addr] = float(self.config.init_dof_pos[i])

        self.qDes = self.config.init_dof_pos.copy()
        self.last_action[:] = 0.0
        self.policy_counter = 0

        self.mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        quat_wxyz = self.data.qpos[3:7].copy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        projected_gravity = compute_projected_gravity(quat_xyzw)

        dof_pos = np.array(
            [self.data.qpos[addr] for addr in self.handles.joint_qpos_addrs],
            dtype=np.float32,
        )
        dof_vel = np.array(
            [self.data.qvel[addr] for addr in self.handles.joint_dof_addrs],
            dtype=np.float32,
        )
        return projected_gravity, dof_pos, dof_vel

    def send_command(self, target_pos: np.ndarray, kp: float, kd: float) -> None:
        dof_pos = np.array(
            [self.data.qpos[addr] for addr in self.handles.joint_qpos_addrs],
            dtype=np.float32,
        )
        dof_vel = np.array(
            [self.data.qvel[addr] for addr in self.handles.joint_dof_addrs],
            dtype=np.float32,
        )

        tau = kp * (target_pos - dof_pos) - kd * dof_vel
        tau = np.clip(tau, -self.config.tau_limit, self.config.tau_limit)

        for i, actuator_id in enumerate(self.handles.actuator_ids):
            self.data.ctrl[actuator_id] = float(tau[i])

    def step(self) -> None:
        self.mujoco.mj_step(self.model, self.data)

    def _should_abort_fall(self) -> bool:
        if not self.config.abort_on_fall:
            return False

        base_z = float(self.data.qpos[2])
        quat_wxyz = self.data.qpos[3:7].copy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        roll, pitch, _ = quat_to_euler_xyz(quat_xyzw)

        if base_z < self.config.abort_base_height:
            print(f"\nRobot fell: base z too low ({base_z:.3f})")
            return True

        if abs(roll) > self.config.abort_rpy or abs(pitch) > self.config.abort_rpy:
            print(f"\nRobot fell: roll={roll:.2f}, pitch={pitch:.2f}")
            return True

        return False

    def _control_step(self, sim_time: float, keyboard: KeyboardController) -> None:
        # Phase 1: stand up automatically
        if sim_time <= self.config.standup_duration:
            phase = np.tanh(sim_time / 1.2)
            self.qDes = (
                phase * self.config.default_dof_pos
                + (1.0 - phase) * self.config.init_dof_pos
            )
            self.send_command(self.qDes, kp=self.config.stand_kp, kd=self.config.stand_kd)
            return

        # Phase 2: stabilize
        if sim_time <= self.config.standup_duration + self.config.stabilize_duration:
            self.qDes = self.config.default_dof_pos.copy()
            self.send_command(self.qDes, kp=self.config.stand_kp, kd=self.config.stand_kd)
            return

        # Phase 3: policy control
        self.policy_counter += 1
        if self.policy_counter >= self.policy_decimation:
            self.policy_counter = 0

            cmd_vx, cmd_vy, cmd_vyaw = keyboard.get_velocity()
            commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)

            projected_gravity, dof_pos, dof_vel = self.get_state()

            obs = build_obs(
                projected_gravity=projected_gravity,
                commands=commands,
                dof_pos=dof_pos,
                dof_vel=dof_vel,
                last_action=self.last_action,
                config=self.config,
                obs_dim=self.policy_obs_dim,
            )
            obs = normalize_obs(obs, self.config.clip_observations)
            obs_batch = obs[np.newaxis, :].astype(np.float32)

            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs_batch)
                action_tensor = self.policy(obs_tensor)
                if isinstance(action_tensor, tuple):
                    action_tensor = action_tensor[0]
                action = action_tensor.cpu().numpy().flatten().astype(np.float32)

            action = np.clip(action, -self.config.clip_actions, self.config.clip_actions)
            self.last_action = action[:12].copy()
            self.qDes = self.last_action * self.config.action_scale + self.config.default_dof_pos

        self.send_command(self.qDes, kp=self.config.walk_kp, kd=self.config.walk_kd)

    def run(self, keyboard: KeyboardController) -> None:
        motiontime = 0

        if self.headless:
            while True:
                motiontime += 1
                sim_time = motiontime * self.config.sim_dt

                if keyboard.exit_requested:
                    print("\nExit request detected, ending simulation...")
                    break

                self._control_step(sim_time, keyboard)
                self.step()

                if self._should_abort_fall():
                    break

                if motiontime % int(round(1.0 / self.config.sim_dt)) == 0:
                    cmd_vx, cmd_vy, cmd_vyaw = keyboard.get_velocity()
                    print(f"\n[Cmd] vx={cmd_vx:+.2f}, vy={cmd_vy:+.2f}, yaw={cmd_vyaw:+.2f}")
                    print(f"[Sim] t={sim_time:.1f}s, base_z={self.data.qpos[2]:.3f}m")
        else:
            with self.mujoco_viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.lookat[:] = self.data.qpos[:3]
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20

                while viewer.is_running():
                    motiontime += 1
                    sim_time = motiontime * self.config.sim_dt

                    if keyboard.exit_requested:
                        print("\nExit request detected, ending simulation...")
                        break

                    self._control_step(sim_time, keyboard)
                    self.step()

                    if self._should_abort_fall():
                        break

                    viewer.cam.lookat[:] = self.data.qpos[:3]
                    viewer.sync()

                    if motiontime % int(round(1.0 / self.config.sim_dt)) == 0:
                        cmd_vx, cmd_vy, cmd_vyaw = keyboard.get_velocity()
                        print(f"\n[Cmd] vx={cmd_vx:+.2f}, vy={cmd_vy:+.2f}, yaw={cmd_vyaw:+.2f}")
                        print(f"[Sim] t={sim_time:.1f}s, base_z={self.data.qpos[2]:.3f}m")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sim2Sim MuJoCo controller with PD motor control")
    parser.add_argument("--model", type=str, default="policy_1.pt", help="TorchScript policy file")
    parser.add_argument("--xml", type=str, default="scene.xml", help="MuJoCo XML file")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--init-z", type=float, default=None, help="Override initial base height")
    parser.add_argument(
        "--assets-dir",
        type=str,
        default="/home/a2rlab/AMP_for_hardware/deploy/assets/go2",
        help="Directory containing MuJoCo XML",
    )
    parser.add_argument(
        "--policy-dir",
        type=str,
        default="/home/a2rlab/AMP_for_hardware/deploy/exported_policy",
        help="Directory containing policy file",
    )
    args = parser.parse_args()

    config = SimConfig()
    if args.init_z is not None:
        config.init_base_height = float(args.init_z)

    keyboard = KeyboardController(
        vx_range=config.vx_range,
        vy_range=config.vy_range,
        vyaw_range=config.vyaw_range,
    )
    keyboard.start()

    print("\n" + "=" * 70)
    print("Keyboard Control Commands")
    print("=" * 70)
    print("  W: increase forward speed vx")
    print("  S: decrease forward speed vx")
    print("  A: turn left vyaw")
    print("  D: turn right vyaw")
    print("  Space: zero all commands")
    print("  Q: exit")
    print("=" * 70 + "\n")

    xml_path = os.path.join(args.assets_dir, args.xml)
    policy_path = os.path.join(args.policy_dir, args.model)

    controller = Sim2SimController(config, xml_path, policy_path, args.headless)

    try:
        controller.run(keyboard)
    finally:
        keyboard.stop()
        print("\nProgram ended.")Config,
    obs_dim: int,
) -> np.ndarray:
    if obs_dim != 42:
        raise ValueError(f"Expected 42D policy input, got {obs_dim}")

    obs = []
    obs.extend(list(projected_gravity))
    obs.extend(list(commands * cfg.obs_scales["commands"]))
    obs.extend(list((dof_pos - cfg.default_dof_pos) * cfg.obs_scales["dof_pos"]))
    obs.extend(list(dof_vel * cfg.obs_scales["dof_vel"]))
    obs.extend(list(last_action))
    obs = np.array(obs, dtype=np.float32)
    return np.clip(obs, -cfg.clip_observations, cfg.clip_observations)


class KeyboardController:
    def __init__(self, vx_range, vy_range, vyaw_range):
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vyaw_range = vyaw_range

        self.vx_step = 0.1
        self.vy_step = 0.05
        self.vyaw_step = 0.1

        self.running = True
        self.exit_requested = False
        self.lock = threading.Lock()
        self.thread = None

    def get_velocity(self):
        with self.lock:
            return self.vx, self.vy, self.vyaw

    def set_velocity(self, vx, vy, vyaw):
        with self.lock:
            self.vx = float(np.clip(vx, self.vx_range[0], self.vx_range[1]))
            self.vy = float(np.clip(vy, self.vy_range[0], self.vy_range[1]))
            self.vyaw = float(np.clip(vyaw, self.vyaw_range[0], self.vyaw_range[1]))

    def keyboard_thread(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue

                key = sys.stdin.read(1)
                vx, vy, vyaw = self.get_velocity()

                if key in ("w", "W"):
                    vx += self.vx_step
                elif key in ("s", "S"):
                    vx -= self.vx_step
                elif key in ("a", "A"):
                    vyaw += self.vyaw_step
                elif key in ("d", "D"):
                    vyaw -= self.vyaw_step
                elif key in ("z", "Z"):
                    vy += self.vy_step
                elif key in ("c", "C"):
                    vy -= self.vy_step
                elif key == " ":
                    vx, vy, vyaw = 0.0, 0.0, 0.0
                elif key in ("q", "Q"):
                    self.exit_requested = True
                    self.running = False
                    print("\n[Keyboard] Exit requested...")
                    break
                else:
                    continue

                self.set_velocity(vx, vy, vyaw)
                vx, vy, vyaw = self.get_velocity()
                print(
                    f"\r[Command] vx={vx:+.2f} m/s, vy={vy:+.2f} m/s, yaw={vyaw:+.2f} rad/s",
                    end="",
                    flush=True,
                )
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def start(self):
        self.thread = threading.Thread(target=self.keyboard_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


class Sim2SimController:
    JOINT_NAMES = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    ACTUATOR_NAMES = [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]

    def __init__(self, cfg: Config, xml_path: str, policy_path: str, headless: bool = False):
        self.cfg = cfg
        self.headless = headless

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = cfg.sim_dt

        self.joint_qpos_addrs = []
        self.joint_dof_addrs = []
        self.actuator_ids = []

        for joint_name, actuator_name in zip(self.JOINT_NAMES, self.ACTUATOR_NAMES):
            joint_id = self.model.joint(joint_name).id
            self.joint_qpos_addrs.append(int(self.model.jnt_qposadr[joint_id]))
            self.joint_dof_addrs.append(int(self.model.jnt_dofadr[joint_id]))
            self.actuator_ids.append(int(self.model.actuator(actuator_name).id))

        self.ctrl_min = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1].copy()

        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()
        self.policy_obs_dim = self._infer_policy_input_dim()

        self.last_action = np.zeros(12, dtype=np.float32)
        self.qDes = self.cfg.default_dof_pos.copy()

        self.policy_decimation = max(1, int(round(cfg.policy_dt / cfg.sim_dt)))
        self.policy_counter = 0

        self.reset_pose()

    def _infer_policy_input_dim(self) -> int:
        state = self.policy.state_dict()
        for key, value in state.items():
            if key.endswith("weight") and getattr(value, "ndim", 0) == 2:
                return int(value.shape[1])
        raise RuntimeError("Could not infer policy input dimension")

    def reset_pose(self):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.cfg.init_base_height
        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0

        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.qpos[qpos_addr] = float(self.cfg.default_dof_pos[i])

        self.last_action[:] = 0.0
        self.qDes = self.cfg.default_dof_pos.copy()
        self.policy_counter = 0

        mujoco.mj_forward(self.model, self.data)

    def get_dof_pos(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.joint_qpos_addrs], dtype=np.float32)

    def get_dof_vel(self) -> np.ndarray:
        return np.array([self.data.qvel[a] for a in self.joint_dof_addrs], dtype=np.float32)

    def get_state(self):
        quat_wxyz = self.data.qpos[3:7].copy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        projected_gravity = compute_projected_gravity(quat_xyzw)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        return projected_gravity, dof_pos, dof_vel

    def send_pd(self, target_pos: np.ndarray, kp: float, kd: float):
        q = self.get_dof_pos()
        dq = self.get_dof_vel()

        tau = kp * (target_pos - q) - kd * dq
        tau = np.clip(tau, self.ctrl_min, self.ctrl_max)

        for i, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = float(tau[i])

    def _control_step(self, sim_time: float, keyboard: KeyboardController):
        if sim_time <= self.cfg.stabilize_duration:
            self.qDes = self.cfg.default_dof_pos.copy()
            self.send_pd(self.qDes, self.cfg.hold_kp, self.cfg.hold_kd)
            return

        self.policy_counter += 1
        if self.policy_counter >= self.policy_decimation:
            self.policy_counter = 0

            cmd_vx, cmd_vy, cmd_vyaw = keyboard.get_velocity()
            commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)

            projected_gravity, dof_pos, dof_vel = self.get_state()
            obs = build_obs(
                projected_gravity=projected_gravity,
                commands=commands,
                dof_pos=dof_pos,
                dof_vel=dof_vel,
                last_action=self.last_action,
                cfg=self.cfg,
                obs_dim=self.policy_obs_dim,
            )
            obs_batch = obs[np.newaxis, :].astype(np.float32)

            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs_batch)
                action_tensor = self.policy(obs_tensor)
                if isinstance(action_tensor, tuple):
                    action_tensor = action_tensor[0]
                action = action_tensor.cpu().numpy().flatten().astype(np.float32)

            action = np.clip(action, -self.cfg.clip_actions, self.cfg.clip_actions)
            self.last_action = action[:12].copy()
            self.qDes = self.last_action * self.cfg.action_scale + self.cfg.default_dof_pos

        self.send_pd(self.qDes, self.cfg.walk_kp, self.cfg.walk_kd)

    def _print_status(self, sim_time: float, keyboard: KeyboardController):
        dof_pos = self.get_dof_pos()
        err = dof_pos - self.cfg.default_dof_pos
        quat_wxyz = self.data.qpos[3:7].copy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        rpy = quat_to_euler_xyz(quat_xyzw)
        vx, vy, vyaw = keyboard.get_velocity()

        print(
            f"\n[t={sim_time:.2f}] base_z={self.data.qpos[2]:.3f} "
            f"roll={rpy[0]:+.3f} pitch={rpy[1]:+.3f} "
            f"vx={vx:+.2f} vy={vy:+.2f} yaw={vyaw:+.2f} "
            f"max_err={np.abs(err).max():.3f}"
        )

    def run(self, keyboard: KeyboardController):
        motiontime = 0

        if self.headless:
            while True:
                motiontime += 1
                sim_time = motiontime * self.cfg.sim_dt

                if keyboard.exit_requested:
                    print("\nExit request detected, ending simulation...")
                    break

                self._control_step(sim_time, keyboard)
                mujoco.mj_step(self.model, self.data)

                if motiontime % int(round(0.5 / self.cfg.sim_dt)) == 0:
                    self._print_status(sim_time, keyboard)
        else:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.lookat[:] = self.data.qpos[:3]
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20

                while viewer.is_running():
                    motiontime += 1
                    sim_time = motiontime * self.cfg.sim_dt

                    if keyboard.exit_requested:
                        print("\nExit request detected, ending simulation...")
                        break

                    self._control_step(sim_time, keyboard)
                    mujoco.mj_step(self.model, self.data)

                    viewer.cam.lookat[:] = self.data.qpos[:3]
                    viewer.sync()

                    if motiontime % int(round(0.5 / self.cfg.sim_dt)) == 0:
                        self._print_status(sim_time, keyboard)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="scene.xml")
    parser.add_argument("--model", type=str, default="policy_1.pt")
    parser.add_argument("--assets-dir", type=str, default="/home/a2rlab/AMP_for_hardware/deploy/assets/go2")
    parser.add_argument("--policy-dir", type=str, default="/home/a2rlab/AMP_for_hardware/deploy/exported_policy")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    xml_path = os.path.join(args.assets_dir, args.xml)
    policy_path = os.path.join(args.policy_dir, args.model)

    keyboard = KeyboardController(
        vx_range=cfg.vx_range,
        vy_range=cfg.vy_range,
        vyaw_range=cfg.vyaw_range,
    )
    keyboard.start()

    print("\n" + "=" * 70)
    print("Controls")
    print("=" * 70)
    print("W/S : vx")
    print("Z/C : vy")
    print("A/D : yaw")
    print("Space : zero commands")
    print("Q : quit")
    print("=" * 70 + "\n")

    sim = Sim2SimController(cfg, xml_path, policy_path, headless=args.headless)

    try:
        sim.run(keyboard)
    finally:
        keyboard.stop()
        print("\nProgram ended.")