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


   # ----------------------------
   # Config overrides
   # ----------------------------
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


   # ----------------------------
   # Build environment
   # ----------------------------
   env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
   env.reset()
   obs = env.get_observations()


   # ----------------------------
   # Load policy
   # ----------------------------
   train_cfg.runner.resume = True
   ppo_runner, train_cfg = task_registry.make_alg_runner(
       env=env, name=args.task, args=args, train_cfg=train_cfg
   )
   policy = ppo_runner.get_inference_policy(device=env.device)


   # ----------------------------
   # Export policy
   # ----------------------------
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


   # ----------------------------
   # Output paths
   # ----------------------------
   export_dir = os.path.join(
       LEGGED_GYM_ROOT_DIR,
       "logs",
       train_cfg.runner.experiment_name,
       "exported",
   )
   os.makedirs(export_dir, exist_ok=True)


   video_path = os.path.join(export_dir, "record_sequence_debug.mp4")


   # ----------------------------
   # Camera setup
   # ----------------------------
   cam_props = gymapi.CameraProperties()
   cam_props.width = 1280
   cam_props.height = 720


   gym = env.gym
   sim = env.sim


   num_envs = env.num_envs
   num_record = min(num_envs, num_record_envs)


   camera_handles = []
   for k in range(num_record):
       cam = gym.create_camera_sensor(env.envs[k], cam_props)
       camera_handles.append(cam)


   # 🔵 FIXED CAMERA (no rotation)
   cam_offset = np.array([1.2, 0.0, 0.6], dtype=np.float32)
   look_offset = np.array([0.0, 0.0, 0.2], dtype=np.float32)


   # ----------------------------
   # Timing
   # ----------------------------
   seconds_per_env = 2.0
   frames_per_env = int(seconds_per_env / env.dt)
   fps = int(round(1.0 / env.dt))
   num_frames = frames_per_env * num_record


   print(f"Recording {num_record} envs sequentially")
   print(f"{seconds_per_env}s per env | total frames {num_frames} | fps {fps}")


   video = None


   # ----------------------------
   # Rollout
   # ----------------------------
   for i in range(num_frames):


       # 🔵 OPTIONAL: force same command for all envs
       # env.commands[:] = 0.0
       # env.commands[:, 0] = 1.0  # forward velocity


       with torch.no_grad():
           actions = policy(obs.detach())


       obs, _, rew, done, infos, _, _ = env.step(actions.detach())


       # Select which env to record
       env_id = min(i // frames_per_env, num_record - 1)
       env_handle = env.envs[env_id]
       camera_handle = camera_handles[env_id]


       # Robot state
       root_pos = env.root_states[env_id, :3].cpu().numpy()


       # Camera placement
       cam_pos = root_pos + cam_offset
       cam_target = root_pos + look_offset


       gym.set_camera_location(
           camera_handle,
           env_handle,
           gymapi.Vec3(*cam_pos.tolist()),
           gymapi.Vec3(*cam_target.tolist()),
       )


       gym.step_graphics(sim)
       gym.render_all_camera_sensors(sim)


       img = gym.get_camera_image(
           sim, env_handle, camera_handle, gymapi.IMAGE_COLOR
       )


       if img is None:
           print(f"Warning: no image at frame {i}")
           continue


       img = np.array(img, dtype=np.uint8).reshape(
           cam_props.height, cam_props.width, 4
       )
       img_bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)


       # ----------------------------
       # Extract debug info
       # ----------------------------
       cmd = env.commands[env_id].cpu().numpy()
       reward = rew[env_id].item()


       base_lin_vel = env.base_lin_vel[env_id].cpu().numpy()
       base_ang_vel = env.base_ang_vel[env_id].cpu().numpy()


       # ----------------------------
       # Overlay debug text
       # ----------------------------
       overlay = img_bgr.copy()


       cv2.putText(overlay, f"env {env_id}", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)


       cv2.putText(overlay,
                   f"cmd: vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}",
                   (30, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2)


       cv2.putText(overlay,
                   f"reward: {reward:+.3f}",
                   (30, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,200), 2)


       cv2.putText(overlay,
                   f"vel: vx={base_lin_vel[0]:+.2f} vy={base_lin_vel[1]:+.2f}",
                   (30, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,255), 2)


       cv2.putText(overlay,
                   f"frame {i - env_id * frames_per_env + 1}/{frames_per_env}",
                   (30, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


       # ----------------------------
       # Write video
       # ----------------------------
       if video is None:
           video = cv2.VideoWriter(
               video_path,
               cv2.VideoWriter_fourcc(*"mp4v"),
               fps,
               (cam_props.width, cam_props.height),
           )


       video.write(overlay)


   # ----------------------------
   # Cleanup
   # ----------------------------
   if video:
       video.release()
       print("Saved video:", video_path)




if __name__ == "__main__":
   EXPORT_POLICY = False
   args = get_args()
   play(args)
