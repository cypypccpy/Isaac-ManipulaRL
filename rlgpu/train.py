#!/usr/bin/env python3

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import random

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_ppo import process_ppo
from utils.process_sac import process_sac

import torch


def train():
    task, env = parse_task(args, cfg, cfg_train, sim_params)
    # rl_algorithm = process_ppo(args, env, cfg_train, logdir)
    rl_algorithm = process_ppo(args, env, cfg_train, logdir)

    rl_algorithm_iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        rl_algorithm_iterations = args.max_iterations
        
    rl_algorithm.run(num_learning_iterations=rl_algorithm_iterations, log_interval=cfg_train["learn"]["save_interval"])


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
