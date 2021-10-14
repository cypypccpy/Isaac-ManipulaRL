import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torchvision.models import squeezenet
from utils.rl_pytorch.ppo.resmlp import ResMLP
from utils.rl_pytorch.ppo.mynetwork import MyNetWork

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

class ActorCritic(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
        super(ActorCritic, self).__init__()

        self.asymmetric = asymmetric

        self.actor = MyNetWork(output=actions_shape[0])

        self.critic = MyNetWork(output=1)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        self.log_actions_mean = actions_mean
        self.log_value = value

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, observations, states, actions):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
