from abc import abstractclassmethod
from codecs import strict_errors
from datetime import datetime
import os
import time

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque

import torch
from torch._C import BenchmarkExecutionStats, DeviceObjType
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
                #  demonstration_buffer_len = 50000,
                 demonstration_buffer_len = 0,
                 replay_buffer_len = 1000000,
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

        # self.twin_optimizer = optim.Adam(self.actor_critic.twin_net.parameters(), lr=policy_lr)

        self.buffer = ReplayBeffer(self.replay_buffer_len, self.demonstration_buffer_len)
        # hyperparameters
        self.gamma = gamma
        self.target_entropy = np.log(vec_env.num_actions)
        self.tau = tau
        self.alpha_log = torch.tensor((np.log(0.2),), dtype=torch.float32,
                                requires_grad=True, device=self.device)  # trainable parameter
        # self.alpha = torch.tensor((1,), dtype=torch.float32,
        #                         requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=learning_rate)

        self.abstract_states = torch.tensor(([0, 0],), dtype=torch.float32,
                                requires_grad=True, device=self.device)

        self.batch_size = batch_size
        self.criterion = torch.nn.SmoothL1Loss()
        self.twin_criterion = torch.nn.CrossEntropyLoss()

        # Load the target value network parameters
        for target_param, param in zip(self.actor_critic.target_value_net.parameters(), self.actor_critic.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # SAC parameters
        self.state_dim = self.vec_env.observation_space.shape[0]
        self.action_dim = self.vec_env.action_space.shape[0]

        self.apply_reset = apply_reset

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path), strict = False)
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path), strict = False)
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

                    # domain_para, force = self.vec_env.get_twin_module_data()
                    # self.abstract_states = self.actor_critic.act_abstract_states(current_obs[:, 9:12], force)
                    # self.abstract_states = torch.softmax(self.abstract_states, dim=1)

                    # print(self.abstract_states)
                    current_obs.copy_(next_obs)
        else:
            Return = []
            best_score_mean, best_it = 0, 0
            action_range = torch.Tensor([self.action_space.low, self.action_space.high]).to('cuda:0')
            states = current_obs

            for it in range(self.current_learning_iteration, num_learning_iterations):
                score = 0
                current_obs = self.vec_env.reset()
                # Rollout
                for _ in range(self.num_learning_epochs):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    if self.buffer.buffer_len() >= self.demonstration_buffer_len:
                        actions = self.actor_critic.act(states)
                    else:
                        actions = self.vec_env.get_reverse_actions()
                        print(actions[0])
                        # actions = self.actor_critic.act(states)
                    # action_in =  actions * (action_range[1] - action_range[0]) / 2.0 + (action_range[1] + action_range[0]) / 2.0
                    # Step the vec_environment
                    next_states, reward, done, _ = self.vec_env.step(actions)
                    domain_para, force = self.vec_env.get_twin_module_data()
                    # implement reward scale
                    reward *= self.reward_scale

                    if self.buffer.buffer_len() < self.demonstration_buffer_len:
                        self.buffer.push_demonstration_data((states, actions, reward, next_states, done), 200)
                    else:
                        self.buffer.push((states, actions, reward, next_states, done))
                    states = next_states

                    if self.buffer.buffer_len() >= self.demonstration_buffer_len:
                        score += reward
                    # if done:
                    #     break
                    if self.buffer.buffer_len() >= self.demonstration_buffer_len + 1 and self.buffer.buffer_len() >= self.batch_size:
                        self.update(self.batch_size)
                        self.update_twin_module(states, domain_para, force)
                    
                print("episode:{}, score:{}, buffer_capacity:{}".format(it, score.mean(), self.buffer.buffer_len()))
                self.writer.add_scalar('Reward/Reward', score.mean(), it)
                self.writer.add_scalar('Reward/Alpha', self.alpha_log.exp().detach().mean(), it)
                self.writer.add_scalar('Reward/TwinLoss', self.twin_loss.detach().mean(), it)
                self.writer.add_scalar('Reward/Std', self.actor_critic.std.detach().mean(), it)

                if score.mean() >= best_score_mean:
                    best_it = it - 1
                    best_score_mean = score.mean()
                else:
                    if best_score_mean >= 1500 and torch.rand(1) < 0.25:
                        self.actor_critic.load_state_dict(torch.load(os.path.join(self.log_dir, 'model_{}.pt'.format(best_it))))
                        print("return to best model: " + 'model_{}.pt'.format(best_it))
                        self.actor_critic.train()

                print(best_score_mean)
                score = 0

                if it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def update_twin_module(self, states, domain_para, force):
        useful_state = states[:, 9:12]
        
        self.abstract_states = self.actor_critic.act_abstract_states(useful_state, force)

        self.twin_loss = self.twin_criterion(self.abstract_states, domain_para)

        self.twin_optimizer.zero_grad()
        self.twin_loss.backward()
        self.twin_optimizer.step()

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

