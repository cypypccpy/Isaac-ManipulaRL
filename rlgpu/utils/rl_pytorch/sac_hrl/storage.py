from os import name
import gym
import torch
import random
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal

class ReplayBeffer():
    def __init__(self, buffer_maxlen, demonstration_buffer_maxlen):
        self.buffer = []
        self.demonstration_buffer = collections.deque(maxlen=demonstration_buffer_maxlen)
        self.controller_buffer = collections.deque(maxlen=buffer_maxlen)
        self.meta_controller_buffer = collections.deque(maxlen=10000)
        self.demonstration_buffer_maxlen = demonstration_buffer_maxlen
    
    def push_controller(self, data):
        self.controller_buffer.append(data)

    def push_demonstration_data(self, data, iter):
        for i in range(iter):
            self.demonstration_buffer.append(data)

    def push_meta_controller(self, data):
        self.meta_controller_buffer.append(data)


    def sample_meta_controller(self, batch_size):
        state_list = torch.FloatTensor([])
        action_list = torch.FloatTensor([])
        reward_list = torch.FloatTensor([])
        next_state_list = torch.FloatTensor([])
        done_list = torch.FloatTensor([])
        
        self.buffer = []
        self.buffer.extend(self.demonstration_buffer)
        self.buffer.extend(self.learning_buffer)
        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience
            # state, action, reward, next_state, done
            if state_list.shape[0] == 0:
                state_list = s
                action_list = a
                reward_list = r
                next_state_list = n_s
                done_list = d
            else:
                state_list = torch.cat((state_list, s), dim = 0)
                action_list = torch.cat((action_list, a), dim = 0)
                reward_list = torch.cat((reward_list, r), dim = 0)
                next_state_list = torch.cat((next_state_list, n_s), dim = 0)
                done_list = torch.cat((done_list, d), dim = 0)
        
        return state_list, \
               action_list, \
               reward_list.unsqueeze(-1), \
               next_state_list, \
               done_list.unsqueeze(-1)

    def sample_controller(self, batch_size):
        state_list = torch.FloatTensor([])
        action_list = torch.FloatTensor([])
        reward_list = torch.FloatTensor([])
        next_state_list = torch.FloatTensor([])
        done_list = torch.FloatTensor([])
        
        self.buffer = []
        self.buffer.extend(self.demonstration_buffer)
        self.buffer.extend(self.learning_buffer)
        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience
            # state, action, reward, next_state, done
            if state_list.shape[0] == 0:
                state_list = s
                action_list = a
                reward_list = r
                next_state_list = n_s
                done_list = d
            else:
                state_list = torch.cat((state_list, s), dim = 0)
                action_list = torch.cat((action_list, a), dim = 0)
                reward_list = torch.cat((reward_list, r), dim = 0)
                next_state_list = torch.cat((next_state_list, n_s), dim = 0)
                done_list = torch.cat((done_list, d), dim = 0)
        
        return state_list, \
               action_list, \
               reward_list.unsqueeze(-1), \
               next_state_list, \
               done_list.unsqueeze(-1)

    def buffer_len(self):
        self.buffer = []
        self.buffer.extend(self.demonstration_buffer)
        self.buffer.extend(self.controller_buffer)
        return len(self.buffer)
