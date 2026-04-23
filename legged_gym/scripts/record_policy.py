import os
import cv2
import numpy as np

from isaacgym import gymapi
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Config overrides
    num_record_envs = 16
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, num_record_envs)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    train_cfg.runner.amp_num_preload_transitions = 1

    # Build environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    obs = env.get_observations()

    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Export policy
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        os.makedirs(path, exist_ok=True)
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy to:", path)

    # Output path
    export_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
    )
    os.makedirs(export_dir, exist_ok=True)
    video_path = os.path.join(export_dir, "record_sequence_debug.mp4")

    # Camera setup
    gym, sim = env.gym, env.sim
    num_record = min(env.num_envs, num_record_envs)

    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 720

    camera_handles = [
        gym.create_camera_sensor(env.envs[k], cam_props) for k in range(num_record)
    ]

    cam_offset = np.array([0.0, -1.2, 0.6], dtype=np.float32)
    look_offset = np.array([0.0, 0.0, 0.2], dtype=np.float32)

    # Timing
    seconds_per_env = 4.0
    frames_per_env = int(seconds_per_env / env.dt)
    fps = int(round(1.0 / env.dt))
    num_frames = frames_per_env * num_record

    print(f"Recording {num_record} envs sequentially")
    print(f"{seconds_per_env}s per env | total frames {num_frames} | fps {fps}")

    video = None
    vx_vals = 0.5 * torch.rand(env.num_envs, device=env.device)
    
    for i in range(num_frames):
        env.commands[:] = 0.0
        env.commands[:, 0] = vx_vals

        with torch.no_grad():
            actions = policy(obs.detach())

        obs, _, rew, done, infos, _, _ = env.step(actions.detach())

        env_id = min(i // frames_per_env, num_record - 1)
        env_handle = env.envs[env_id]
        camera_handle = camera_handles[env_id]

        root_pos = env.root_states[env_id, :3].cpu().numpy()
        quat = env.root_states[env_id, 3:7].cpu().numpy()  # (x, y, z, w)

        # extract yaw from quaternion
        x, y, z, w = quat
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        # rotation matrix around z
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

        # define side-view in robot frame (left side)
        local_offset = np.array([0.0, -1.5, 0.7])
        local_look   = np.array([0.0,  0.0, 0.35])

        # rotate into world frame
        cam_pos = root_pos + R @ local_offset
        cam_target = root_pos + R @ local_look

        gym.set_camera_location(
            camera_handle,
            env_handle,
            gymapi.Vec3(*cam_pos.tolist()),
            gymapi.Vec3(*cam_target.tolist()),
        )

        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

        img = gym.get_camera_image(sim, env_handle, camera_handle, gymapi.IMAGE_COLOR)
        if img is None:
            print(f"Warning: no image at frame {i}")
            continue

        img = np.array(img, dtype=np.uint8).reshape(cam_props.height, cam_props.width, 4)
        overlay = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)

        cmd = env.commands[env_id].cpu().numpy()
        reward = rew[env_id].item()
        base_lin_vel = env.base_lin_vel[env_id].cpu().numpy()

        text_rows = [
            (f"env {env_id}", (30, 50), 1.2, (255, 255, 255)),
            (f"cmd: vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}", (30, 100), 0.8, (200, 255, 200)),
            (f"reward: {reward:+.3f}", (30, 140), 0.8, (255, 200, 200)),
            (f"vel: vx={base_lin_vel[0]:+.2f} vy={base_lin_vel[1]:+.2f}", (30, 180), 0.8, (200, 200, 255)),
            (f"frame {i - env_id * frames_per_env + 1}/{frames_per_env}", (30, 220), 0.7, (255, 255, 255)),
        ]

        for text, pos, scale, color in text_rows:
            cv2.putText(
                overlay,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                2,
            )

        if video is None:
            video = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (cam_props.width, cam_props.height),
            )

        video.write(overlay)

    if video:
        video.release()
        print("Saved video:", video_path)

if __name__ == "__main__":
    EXPORT_POLICY = False
    args = get_args()
    play(args)