from __future__ import annotations

from typing import Any, Callable

import torch

from .vec_env import VecEnv


class IsaacLabAmpVecEnvWrapper(VecEnv):
    """Adapter between an Isaac Lab-style vector environment and this repo's AMP runner.

    The wrapped environment is expected to expose the usual vector-env attributes
    (`num_envs`, `max_episode_length`, `device`) and to return Gymnasium-style step
    tuples:

        obs, rew, terminated, truncated, extras = env.step(actions)

    The `extras` dictionary should contain:
      - `amp_obs`: current AMP observations after the step
      - `terminal_amp_obs`: terminal AMP observations for environments that reset
      - `reset_env_ids`: indices of environments that terminated or truncated

    This keeps the Isaac Lab-specific logic outside of the AMP runner and turns the
    remaining migration work into an environment-side task.
    """

    def __init__(
        self,
        env: Any,
        *,
        num_actions: int,
        num_obs: int,
        num_privileged_obs: int | None,
        dt: float,
        dof_pos_limits: torch.Tensor,
        amp_obs_getter: Callable[[Any, Any, dict[str, Any]], torch.Tensor] | None = None,
        obs_key: str = "policy",
        privileged_obs_key: str = "critic",
        include_history_steps: int | None = None,
    ):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.num_privileged_obs = num_privileged_obs
        self.dt = dt
        self.dof_pos_limits = dof_pos_limits
        self.include_history_steps = include_history_steps
        self.max_episode_length = env.max_episode_length
        self.device = getattr(env, "device", "cpu")
        self.amp_obs_getter = amp_obs_getter
        self.obs_key = obs_key
        self.privileged_obs_key = privileged_obs_key

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.privileged_obs_buf = None
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs, self.num_privileged_obs, device=self.device
            )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = getattr(
            env,
            "episode_length_buf",
            torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
        )
        self.extras: dict[str, Any] = {}
        self._amp_obs_buf = torch.zeros(self.num_envs, 1, device=self.device)

    def _extract_obs(self, obs: Any):
        if isinstance(obs, dict):
            policy_obs = obs[self.obs_key]
            privileged_obs = obs.get(self.privileged_obs_key)
        else:
            policy_obs = obs
            privileged_obs = None
        return policy_obs, privileged_obs

    def _extract_amp_obs(self, obs: Any, extras: dict[str, Any]) -> torch.Tensor:
        if "amp_obs" in extras:
            return extras["amp_obs"]
        if self.amp_obs_getter is None:
            raise KeyError(
                "Missing `amp_obs` in the environment extras and no `amp_obs_getter` "
                "was provided to IsaacLabAmpVecEnvWrapper."
            )
        return self.amp_obs_getter(self.env, obs, extras)

    def reset(self):
        reset_output = self.env.reset()
        if isinstance(reset_output, tuple):
            obs, extras = reset_output
        else:
            obs, extras = reset_output, {}
        policy_obs, privileged_obs = self._extract_obs(obs)
        self.obs_buf = policy_obs
        self.privileged_obs_buf = privileged_obs
        self._amp_obs_buf = self._extract_amp_obs(obs, extras)
        self.extras = extras
        return self.obs_buf, self.privileged_obs_buf

    def step(self, actions: torch.Tensor):
        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = terminated | truncated

        policy_obs, privileged_obs = self._extract_obs(obs)
        self.obs_buf = policy_obs
        self.privileged_obs_buf = privileged_obs
        self.rew_buf = rewards
        self.reset_buf = dones
        self.extras = extras
        self._amp_obs_buf = self._extract_amp_obs(obs, extras)

        reset_env_ids = extras.get("reset_env_ids")
        if reset_env_ids is None:
            reset_env_ids = torch.nonzero(dones, as_tuple=False).flatten()

        terminal_amp_states = extras.get("terminal_amp_obs")
        if terminal_amp_states is None:
            terminal_amp_states = self._amp_obs_buf[reset_env_ids]

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
