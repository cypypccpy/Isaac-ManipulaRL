from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torch.distributions import Normal

# Value Net
class ValueNet(nn.Module):
    def __init__(self, state_dim, edge=3e-3):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


# Soft Q Net
class SoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SoftQNet, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 64)
        self.linear5 = nn.Linear(64, 1)
        self.hw = nn.Hardswish()

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.hw(self.linear4(x))
        x = self.linear5(x)

        return x


# Policy Net
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 64)
        self.hw = nn.Hardswish()

        self.mean_linear = nn.Linear(64, action_dim)
        self.log_std_linear = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.hw(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

class TwinNet(nn.Module):
    def __init__(self, state_dim, force_dim, edge=3e-3):
        super(TwinNet, self).__init__()
        # self.linear1 = nn.Linear(state_dim + force_dim, 64)
        self.linear1 = nn.Linear(force_dim, 64)

        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 2)

    def forward(self, state, force):
        # x = F.relu(self.linear1(torch.cat([state, force], 1)))
        x = F.relu(self.linear1(force, 1))

        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

        
