# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#from rl_pytorch.ppo import PPO, ActorCritic
#from utils.rl_pytorch.ppo import PPO, ActorCritic
from numpy import log
from utils.rl_pytorch.sac import SAC, ActorCritic

def process_sac(args, env, cfg_train, logdir):
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if not is_testing:
        is_testing = args.test
    if args.resume > 0:
        chkpt = args.resume

    """Set up the PPO system for training or inferencing."""
    sac = SAC(vec_env=env,
              actor_critic_class=ActorCritic,
              num_learning_epochs=learn_cfg["noptepochs"],
              log_dir=logdir,
              is_testing=is_testing
              )

    logdir_name = logdir
    # logdir_name = logdir
    print(is_testing)
    if is_testing:
        logdir = logdir_name
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        sac.test("{}/model_283.pt".format(logdir))

    elif chkpt > 0:
        load_model_log = logdir_name
        print("Loading model from {}/model_{}.pt".format(load_model_log, chkpt))
        sac.load("{}/model_283.pt".format(load_model_log))

    return sac
