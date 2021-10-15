import gym
from numpy import vectorize
from tasks.base.vec_task import VecTask
from gym import spaces

from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch
import torch
import numpy as np

class CustomEnv(gym.Env, VecTask):

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = self.act_space
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = self.obs_space

    def step(self, actions):
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self.task.step(actions_tensor)

        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device), self.task.rew_buf.to(self.rl_device), self.task.reset_buf.to(self.rl_device), self.task.extras
    def reset(self):
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions], dtype=torch.float32, device=self.rl_device))

        # step the simulator
        self.task.step(actions)
        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
    def render(self, mode='human'):
        return
    def close (self):
        return