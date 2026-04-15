import argparse
import os
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer
import imageio

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
    command = np.array([1.0, 0.0, 0.0], dtype=np.float32)


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
    cfg: Config,
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


class SimForwardTest:
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
        self.qDes = cfg.default_dof_pos.copy()

        self.policy_decimation = max(1, int(round(cfg.policy_dt / cfg.sim_dt)))
        self.policy_counter = 0

        self.reset_pose()

    def _infer_policy_input_dim(self) -> int:
        state = self.policy.state_dict()
        for key, value in state.items():
            if key.endswith("weight") and getattr(value, "ndim", 0) == 2:
                return int(value.shape[1])
        raise RuntimeError("Could not infer policy input dimension")

    def reset_pose(self) -> None:
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

    def send_pd(self, target_pos: np.ndarray, kp: float, kd: float) -> None:
        q = self.get_dof_pos()
        dq = self.get_dof_vel()

        tau = kp * (target_pos - q) - kd * dq
        tau = np.clip(tau, self.ctrl_min, self.ctrl_max)

        for i, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = float(tau[i])

    def step_control(self, sim_time: float) -> None:
        if sim_time <= self.cfg.stabilize_duration:
            self.qDes = self.cfg.default_dof_pos.copy()
            self.send_pd(self.qDes, self.cfg.hold_kp, self.cfg.hold_kd)
            return

        self.policy_counter += 1
        if self.policy_counter >= self.policy_decimation:
            self.policy_counter = 0

            projected_gravity, dof_pos, dof_vel = self.get_state()
            obs = build_obs(
                projected_gravity=projected_gravity,
                commands=self.cfg.command,
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

    def print_status(self, sim_time: float) -> None:
        dof_pos = self.get_dof_pos()
        err = dof_pos - self.cfg.default_dof_pos
        quat_wxyz = self.data.qpos[3:7].copy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        rpy = quat_to_euler_xyz(quat_xyzw)

        print(f"[t={sim_time:.2f}] base_z={self.data.qpos[2]:.3f} roll={rpy[0]:+.3f} pitch={rpy[1]:+.3f} "
              f"cmd_vx={self.cfg.command[0]:.2f} max_err={np.abs(err).max():.3f}")

    def run(self) -> None:
        motiontime = 0

        if self.headless:
            while True:
                motiontime += 1
                sim_time = motiontime * self.cfg.sim_dt
                self.step_control(sim_time)
                mujoco.mj_step(self.model, self.data)

                if motiontime % int(round(0.5 / self.cfg.sim_dt)) == 0:
                    self.print_status(sim_time)
        else:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.lookat[:] = self.data.qpos[:3]
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20

                renderer = mujoco.Renderer(self.model, width=640, height=480)
                video = imageio.get_writer("go2_mujoco.mp4", fps=int(1/self.cfg.sim_dt))

                while viewer.is_running():
                    motiontime += 1
                    sim_time = motiontime * self.cfg.sim_dt
                    self.step_control(sim_time)
                    mujoco.mj_step(self.model, self.data)

                    renderer.update_scene(self.data)
                    frame = renderer.render()
                    video.append_data(frame)

                    viewer.cam.lookat[:] = self.data.qpos[:3]
                    viewer.sync()

                    if motiontime % int(round(0.5 / self.cfg.sim_dt)) == 0:
                        self.print_status(sim_time)
                video.close()


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

    sim = SimForwardTest(cfg, xml_path, policy_path, headless=args.headless)
    sim.run()