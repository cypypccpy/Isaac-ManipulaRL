from isaacgym.gymutil import AxesGeometry
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torchvision.models import squeezenet
from utils.rl_pytorch.sac.mynetwork import ValueNet, PolicyNet, SoftQNet, TwinNet, DiscretePolicyNet

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch.distributions import Normal, Categorical

class MetaActorCritic(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
        super(MetaActorCritic, self).__init__()

        self.asymmetric = asymmetric

        # initialize networks
        self.meta_q1_net = SoftQNet(obs_shape[0], obs_shape[0]).cuda()
        self.meta_q2_net = SoftQNet(obs_shape[0], obs_shape[0]).cuda()
        self.meta_policy_net = DiscretePolicyNet(obs_shape[0], obs_shape[0]).cuda()
        self.meta_target_q1_net = SoftQNet(obs_shape[0], obs_shape[0]).cuda()
        self.meta_target_q2_net = SoftQNet(obs_shape[0], obs_shape[0]).cuda()

    def forward(self):
        raise NotImplementedError

    def act(self, states):
        mean, log_std = self.policy_net(states)
        self.std = log_std.exp()
        normal = Normal(mean, self.std)

        z = normal.sample()
        actions = torch.tanh(z)

        return actions.detach()

    def act_inference(self, states):
        mean, log_std = self.policy_net(states)

        actions = torch.tanh(mean)

        return actions.detach()

    def evaluate_discrete(self, states, epsilon=1e-6):
        action_probabilities = self.meta_policy_net(states)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

class ActorCritic(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
        super(ActorCritic, self).__init__()

        self.asymmetric = asymmetric

        # initialize networks
        self.value_net = ValueNet(obs_shape[0]).cuda()
        self.target_value_net = ValueNet(obs_shape[0]).cuda()
        self.q1_net = SoftQNet(obs_shape[0], actions_shape[0]).cuda()
        self.q2_net = SoftQNet(obs_shape[0], actions_shape[0]).cuda()
        self.policy_net = PolicyNet(obs_shape[0], actions_shape[0]).cuda()
        self.target_q1_net = SoftQNet(obs_shape[0], actions_shape[0]).cuda()
        self.target_q2_net = SoftQNet(obs_shape[0], actions_shape[0]).cuda()

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

    def forward(self):
        raise NotImplementedError

    def act(self, states):
        mean, log_std = self.policy_net(states)
        self.std = log_std.exp()
        normal = Normal(mean, self.std)

        z = normal.sample()
        actions = torch.tanh(z)

        return actions.detach()

    def act_inference(self, states):
        mean, log_std = self.policy_net(states)

        actions = torch.tanh(mean)

        return actions.detach()

    def evaluate(self, states, epsilon=1e-6):

        mean, log_std = self.policy_net(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = torch.randn_like(mean, requires_grad=True)

        action = torch.tanh(mean + std * noise)
        log_prob = normal.log_prob(mean + std * noise) - torch.log(1 - action.pow(2) + epsilon)
        # log_prob = torch.mean(log_prob, dim=1, keepdim=True)

        return action, log_prob

    def act_abstract_states(self, states, force):
        abs_states = self.twin_net(states, force)

        return abs_states


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
