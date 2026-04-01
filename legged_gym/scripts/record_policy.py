import os
import cv2
import numpy as np

from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    train_cfg.runner.amp_num_preload_transitions = 1

    # Prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Export policy as JIT
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
        print("Exported policy as jit script to:", path)

    # Output paths
    export_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
    )
    frames_dir = os.path.join(export_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    video_path = os.path.join(export_dir, "record.mp4")

    # Camera properties
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 720
    cam_props.enable_tensors = False

    # Access underlying Isaac Gym handles
    gym = env.gym
    sim = env.sim
    env_handle = env.envs[0]

    # Create headless camera sensor
    camera_handle = gym.create_camera_sensor(env_handle, cam_props)

    # Initial camera pose
    cam_offset = np.array([1.2, 0.0, 0.45], dtype=np.float32)
    look_offset = np.array([0.0, 0.0, 0.2], dtype=np.float32)

    video_duration = 10.0
    num_frames = int(video_duration / env.dt)
    fps = int(round(1.0 / env.dt))

    print(f"Gathering {num_frames} frames at {fps} FPS")
    video = None

    camera_rot = 0.0
    camera_rot_per_sec = np.pi / 6.0

    for i in range(num_frames):
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, _, infos, _, _ = env.step(actions.detach())

        # Camera follows robot in a circle
        root_pos = env.root_states[0, :3].detach().cpu().numpy()
        camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)

        rel = 1.2 * np.array(
            [np.cos(camera_rot), np.sin(camera_rot), 0.45], dtype=np.float32
        )
        cam_pos = root_pos + rel
        cam_target = root_pos + look_offset

        gym.set_camera_location(
            camera_handle,
            env_handle,
            gymapi.Vec3(*cam_pos.tolist()),
            gymapi.Vec3(*cam_target.tolist()),
        )

        # Important for camera sensors
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

        # Get RGBA image from camera
        img = gym.get_camera_image(sim, env_handle, camera_handle, gymapi.IMAGE_COLOR)

        if img is None:
            print(f"Warning: failed to get image at frame {i}")
            continue

        # Convert flat buffer to H x W x 4
        img = np.array(img, dtype=np.uint8).reshape(cam_props.height, cam_props.width, 4)

        # Drop alpha, convert RGBA -> BGR for OpenCV
        img_bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)

        frame_path = os.path.join(frames_dir, f"{i:05d}.png")
        cv2.imwrite(frame_path, img_bgr)

        if video is None:
            video = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (cam_props.width, cam_props.height),
            )
            if not video.isOpened():
                raise RuntimeError(f"Failed to open video writer: {video_path}")

        video.write(img_bgr)

    if video is not None:
        video.release()
        print("Saved video to:", video_path)
    else:
        print("No video was created.")

    print("Saved frames to:", frames_dir)


if __name__ == "__main__":
    EXPORT_POLICY = True
    args = get_args()
    play(args)