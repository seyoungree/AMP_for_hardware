import os
import time
import numpy as np
import torch

sim_dt = 0.005

default_dof_pos = np.array([
     0.0, 0.9, -1.8,
     0.0, 0.9, -1.8,
     0.0, 0.9, -1.8,
     0.0, 0.9, -1.8,
], dtype=np.float32)

init_dof_pos = np.array([
     0.0, 1.2, -2.4,
     0.0, 1.2, -2.4,
     0.0, 1.2, -2.4,
     0.0, 1.2, -2.4,
], dtype=np.float32)

obs_scales = {
    "dof_pos": 1.0,
    "dof_vel": 0.05,
    "commands": np.array([2.0, 2.0, 0.25], dtype=np.float32),
}

action_scale = 0.25
standup_duration = 3.0
stabilize_duration = 0.5
kp = 50.0float32
kd = 3.5
tau_limit = 60.0
command = np.array([1.5, 0.0, 0.0], dtype=np.float32)


import mujoco
import mujoco.viewer

xml_
path = "/home/a2rlab/AMP_for_hardware/deploy/assets/go2/scene.xml"
policy_path = "/home/a2rlab/AMP_for_hardware/deploy/exported_policy/policy_1.pt"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
model.opt.timestep = sim_dt

policy = torch.jit.load(policy_path, map_location="cpu")
policy.eval()


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

joint_qpos = []
joint_qvel = []
act_ids = []

for j, a in zip(JOINT_NAMES, ACTUATOR_NAMES):
    jid = model.joint(j).id
    joint_qpos.append(model.jnt_qposadr[jid])
    joint_qvel.append(model.jnt_dofadr[jid])
    act_ids.append(model.actuator(a).id)


def get_state():
    quat = data.qpos[3:7].copy()
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])

    g = np.array([0, 0, -1], dtype=np.float32)
    q_w = quat_xyzw[3]
    q_vec = quat_xyzw[:3]

    projected_gravity = (
        g * (2*q_w*q_w - 1)
        - np.cross(q_vec, g) * 2*q_w
        + q_vec * np.dot(q_vec, g) * 2
    )

    dof_pos = np.array([data.qpos[i] for i in joint_qpos], dtype=np.float32)
    dof_vel = np.array([data.qvel[i] for i in joint_qvel], dtype=np.float32)

    return projected_gravity, dof_pos, dof_vel


def build_obs(gravity, commands, dof_pos, dof_vel, last_action):
    obs = []
    obs.extend(list(gravity))
    obs.extend(list(commands * obs_scales["commands"]))
    obs.extend(list((dof_pos - default_dof_pos) * obs_scales["dof_pos"]))
    obs.extend(list(dof_vel * obs_scales["dof_vel"]))
    obs.extend(list(last_action))
    return np.array(obs, dtype=np.float32)


def send_pd(q_des):
    dof_pos = np.array([data.qpos[i] for i in joint_qpos])
    dof_vel = np.array([data.qvel[i] for i in joint_qvel])

    tau = kp * (q_des - dof_pos) - kd * dof_vel
    tau = np.clip(tau, -tau_limit, tau_limit)

    for i, aid in enumerate(act_ids):
        data.ctrl[aid] = float(tau[i])

data.qpos[:] = 0
data.qvel[:] = 0
data.qpos[2] = 0.40
data.qpos[3] = 1.0

for i, addr in enumerate(joint_qpos):
    data.qpos[addr] = init_dof_pos[i]

mujoco.mj_forward(model, data)
last_action = np.zeros(12, dtype=np.float32)

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    step = 0

    while viewer.is_running():
        t += sim_dt
        step += 1

        if t < standup_duration:
            phase = np.tanh(t / 1.2)
            qDes = phase * default_dof_pos + (1 - phase) * init_dof_pos
            send_pd(qDes)

        elif t < standup_duration + stabilize_duration:
            send_pd(default_dof_pos)

        else:
            gravity, dof_pos, dof_vel = get_state()

            obs = build_obs(gravity, command, dof_pos, dof_vel, last_action)
            obs = np.clip(obs, -100, 100)

            with torch.no_grad():
                action = policy(torch.from_numpy(obs[None, :])).numpy().flatten()
            last_action = action[:12]
            qDes = last_action * action_scale + default_dof_pos
            send_pd(qDes)

        mujoco.mj_step(model, data)
        viewer.sync()