from datetime import datetime
import os
import time

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.rl_pytorch.sac import ReplayBeffer


class SAC:

    def __init__(self,
                 vec_env,
                 actor_critic_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=None,
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False
                 ):

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = asymmetric

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.step_size = learning_rate
        self.is_testing = is_testing
        self.current_learning_iteration = 0
        self.num_learning_epochs = num_learning_epochs

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # SAC components
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, asymmetric=asymmetric)
        self.actor_critic.to(self.device)

        # Initialize the optimizer
        q_lr = 3e-3
        value_lr = 3e-3
        policy_lr = 3e-3
        self.value_optimizer = optim.Adam(self.actor_critic.value_net.parameters(), lr=value_lr)
        self.q1_optimizer = optim.Adam(self.actor_critic.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.actor_critic.q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.actor_critic.policy_net.parameters(), lr=policy_lr)

        self.buffer = ReplayBeffer(50000)
        # hyperparameters
        self.gamma = gamma
        self.tau = 0.01
        self.batch_size = 256

        # Load the target value network parameters
        for target_param, param in zip(self.actor_critic.target_value_net.parameters(), self.actor_critic.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # SAC parameters
        self.state_dim = self.vec_env.observation_space.shape[0]
        self.action_dim = self.vec_env.action_space.shape[0]

        self.apply_reset = apply_reset

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        if self.is_testing:
            while True:
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                    # Compute the action
                    actions = self.actor_critic.act_inference(current_obs)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    current_obs.copy_(next_obs)
        else:
            Return = []
            action_range = torch.Tensor([self.action_space.low, self.action_space.high]).to('cuda:0')

            for it in range(self.current_learning_iteration, num_learning_iterations):
                score = 0
                # Rollout
                for _ in range(500):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    states = current_obs

                    actions = self.actor_critic.act(current_obs)
                    action_in =  actions * (action_range[1] - action_range[0]) / 2.0 +  (action_range[1] + action_range[0]) / 2.0
                    # Step the vec_environment
                    next_states, reward, done, _ = self.vec_env.step(action_in)

                    self.buffer.push((states, actions, reward, next_states, done))
                    states = next_states

                    score += reward
                    # if done:
                    #     break
                    if self.buffer.buffer_len() > 500:
                        self.update(self.batch_size)

                print("episode:{}, buffer_capacity:{}".format(it, self.buffer.buffer_len()))
                self.writer.add_scalar('Reward/Reward', score.mean(), it)
                Return.append(score)
                score = 0

    def update(self, batch_size):
        
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        new_action, log_prob = self.actor_critic.evaluate(state)

        # V value loss
        value = self.actor_critic.value_net(state)
        new_q1_value = self.actor_critic.q1_net(state, new_action)
        new_q2_value = self.actor_critic.q2_net(state, new_action)
        next_value = torch.min(new_q1_value, new_q2_value) - log_prob
        value_loss = F.mse_loss(value, next_value.detach())

        # Soft q  loss
        q1_value = self.actor_critic.q1_net(state, action)
        q2_value = self.actor_critic.q2_net(state, action)
        target_value = self.actor_critic.target_value_net(next_state)
        target_q_value = reward + done * self.gamma * target_value
        q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
        q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())

        # Policy loss
        policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value)).mean()

        # Update Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update v
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update Soft q
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_value_loss.backward()
        q2_value_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.actor_critic.target_value_net.parameters(), self.actor_critic.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

