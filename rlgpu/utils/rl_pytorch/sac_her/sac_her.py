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
                 num_learning_epochs,
                 demonstration_buffer_len = 50000,
                 replay_buffer_len = 600000,
                 gamma=0.99,
                 init_noise_std=1.0,
                 learning_rate=3e-4,
                 tau = 0.005,
                 alpha = 4,
                 reward_scale = 1,
                 batch_size = 256,
                 schedule="fixed",
                 desired_kl=None,
                 model_cfg=None,
                 device='cuda:0',
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
        self.demonstration_buffer_len = demonstration_buffer_len
        self.replay_buffer_len = replay_buffer_len

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
        q_lr = learning_rate
        value_lr = learning_rate
        policy_lr = learning_rate
        # ur5_package reward_scale: 4
        self.reward_scale = reward_scale
        self.value_optimizer = optim.Adam(self.actor_critic.value_net.parameters(), lr=value_lr)
        self.q1_optimizer = optim.Adam(self.actor_critic.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.actor_critic.q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.actor_critic.policy_net.parameters(), lr=policy_lr)

        self.buffer = ReplayBeffer(self.replay_buffer_len, self.demonstration_buffer_len)
        # hyperparameters
        self.gamma = gamma
        self.target_entropy = np.log(vec_env.num_actions)
        self.tau = tau
        self.alpha_log = torch.tensor((0.2,), dtype=torch.float32,
                                requires_grad=True, device=self.device)  # trainable parameter
        # self.alpha = torch.tensor((1,), dtype=torch.float32,
        #                         requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=learning_rate)

        self.batch_size = batch_size
        self.criterion = torch.nn.SmoothL1Loss()

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
            states = current_obs

            for it in range(self.current_learning_iteration, num_learning_iterations):
                score = 0
                current_obs = self.vec_env.reset()
                goal = self.vec_env.get_desired_goal()
                # Rollout
                for T in range(self.num_learning_epochs):
                    actions = self.actor_critic.act(torch.cat([states, goal], -1))
                    next_states, reward, done, goal, _ = self.vec_env.step(actions)
                    # implement reward scale
                    reward *= self.reward_scale

                    self.buffer.push((torch.cat([states, goal], -1), actions, reward, torch.cat([next_states, goal], -1), done))

                    states = next_states
                    goal = self.vec_env.get_desired_goal()

                for T in range(self.num_learning_epochs):
                    # future
                    indexes = np.random.randint(T, self.num_learning_epochs, size = 8)
                    for index in indexes:
                        achieved_goal, buffer = self.buffer.get_achieved_goal()

                        achieved_reward = self.vec_env.get_achieved_reward(achieved_goal, buffer[0])
                        self.buffer.push((torch.cat([buffer[0], achieved_goal], -1), buffer[1], achieved_reward, torch.cat([buffer[3], achieved_goal], -1), buffer[4]))

                for _ in range(self.num_learning_epochs):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()

                    if self.buffer.buffer_len() >= self.demonstration_buffer_len + 1 and self.buffer.buffer_len() >= self.batch_size:
                        self.update(self.batch_size)
                    
                print("episode:{}, score:{}, buffer_capacity:{}".format(it, score.mean(), self.buffer.buffer_len()))
                self.writer.add_scalar('Reward/Reward', score.mean(), it)
                self.writer.add_scalar('Reward/Alpha', self.alpha_log.exp().detach().mean(), it)
                Return.append(score)
                score = 0
                
                if it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def update(self, batch_size):
        
        state, action, reward, next_state, done = self.buffer.sample(batch_size)

        #-------------------------------
        # SAC2018 Origin implementation
        #-------------------------------

        # new_action, log_prob = self.actor_critic.evaluate(state)
        # # V value loss
        # value = self.actor_critic.value_net(state)
        # new_q1_value = self.actor_critic.q1_net(state, new_action)
        # new_q2_value = self.actor_critic.q2_net(state, new_action)
        # next_value = torch.min(new_q1_value, new_q2_value) - self.alpha * log_prob
        # value_loss = F.mse_loss(value, next_value.detach())
        # # Soft Q loss
        # q1_value = self.actor_critic.q1_net(state, action)
        # q2_value = self.actor_critic.q2_net(state, action)
        # target_value = self.actor_critic.target_value_net(next_state)
        # target_q_value = reward + self.gamma * target_value
        # # target_q_value = reward + self.gamma * target_value

        # q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
        # q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())

        # # Policy loss
        # policy_loss = (self.alpha * log_prob - torch.min(new_q1_value, new_q2_value)).mean()

        # # Update Policy
        # self.policy_optimizer.zero_grad()
        # policy_loss.backward()
        # self.policy_optimizer.step()

        # # Update v
        # self.value_optimizer.zero_grad()
        # value_loss.backward()
        # self.value_optimizer.step()

        # # Update Soft q
        # self.q1_optimizer.zero_grad()
        # self.q2_optimizer.zero_grad()
        # q1_value_loss.backward()
        # q2_value_loss.backward()
        # self.q1_optimizer.step()
        # self.q2_optimizer.step()

        # # Update target networks
        # for target_param, param in zip(self.actor_critic.target_value_net.parameters(), self.actor_critic.value_net.parameters()):
        #     target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        #--------------------------
        # SpinningUp implementation
        #--------------------------

        # # Set up function for computing Q function
        # q1_value = self.actor_critic.q1_net(state, action)
        # q2_value = self.actor_critic.q2_net(state, action)
        # action2, log_prob2 = self.actor_critic.evaluate(next_state)
        # target_q1_value = self.actor_critic.target_q1_net(next_state, action2)
        # target_q2_value = self.actor_critic.target_q2_net(next_state, action2)
        # backup = reward + self.gamma * (torch.min(target_q1_value, target_q2_value) - 0.2 * log_prob2)

        # q1_value_loss = self.criterion(q1_value, backup)
        # q2_value_loss = self.criterion(q2_value, backup)

        # # Update Soft q
        # self.q1_optimizer.zero_grad()
        # self.q2_optimizer.zero_grad()
        # q1_value_loss.backward(retain_graph=True)
        # q2_value_loss.backward()
        # self.q1_optimizer.step()
        # self.q2_optimizer.step()

        # # Set up function for computing SAC pi loss
        # new_action, log_prob = self.actor_critic.evaluate(state)

        # q1_pi_value = self.actor_critic.q1_net(state, new_action)
        # q2_pi_value = self.actor_critic.q2_net(state, new_action)

        # # Policy loss
        # policy_loss = (0.2 * log_prob - torch.min(q1_pi_value, q2_pi_value)).mean()

        # # Update Policy
        # self.policy_optimizer.zero_grad()
        # policy_loss.backward()
        # self.policy_optimizer.step()

        # # Update target networks
        # for target_param, param in zip(self.actor_critic.target_q1_net.parameters(), self.actor_critic.q1_net.parameters()):
        #     target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        # for target_param, param in zip(self.actor_critic.target_q2_net.parameters(), self.actor_critic.q2_net.parameters()):
        #     target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        #--------------------------
        # SAC2019 Origin implementation
        #--------------------------
        alpha = self.alpha_log.exp().detach()

        with torch.no_grad():
            action2, log_prob2 = self.actor_critic.evaluate(next_state)
            target_q1_value = self.actor_critic.target_q1_net(next_state, action2)
            target_q2_value = self.actor_critic.target_q2_net(next_state, action2)
            backup = reward + (1 - done) * self.gamma * (torch.min(target_q1_value, target_q2_value) - alpha * log_prob2)

        q1_value = self.actor_critic.q1_net(state, action)
        q2_value = self.actor_critic.q2_net(state, action)

        q1_value_loss = self.criterion(q1_value, backup)
        q2_value_loss = self.criterion(q2_value, backup)

        # Update Soft q
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_value_loss.backward(retain_graph=True)
        q2_value_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.actor_critic.target_q1_net.parameters(), self.actor_critic.q1_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        for target_param, param in zip(self.actor_critic.target_q2_net.parameters(), self.actor_critic.q2_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        '''loss of alpha (temperature parameter automatic adjustment)'''
        new_action, log_prob = self.actor_critic.evaluate(state)

        alpha_loss = (- self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        '''loss of actor'''
        alpha = self.alpha_log.exp().detach()

        with torch.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
        
        q1_pi_value = self.actor_critic.q1_net(state, new_action)
        q2_pi_value = self.actor_critic.q2_net(state, new_action)

        # Policy loss
        policy_loss = (alpha * log_prob - torch.min(q1_pi_value, q2_pi_value)).mean()

        # Update Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()