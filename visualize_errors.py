# HACK:
import argparse
import os
import shutil
import torch

from einops import rearrange
from torch.distributions.one_hot_categorical import OneHotCategorical
from pathlib import Path
from utils import load_mamba_model, load_mawm_model, _wrap, bins2continuous, discretize_into_bins, compute_compounding_errors
from utils import _compute_mamba_errors, _compute_mawm_errors
from smac.env import StarCraft2Env
from copy import deepcopy
from collections import deque
import numpy as np
import logging

from configs.flatland.TimetableConfigs import AllAgentLauncherConfig
from configs import SimpleObservationConfig, NearRewardConfig, DeadlockPunishmentConfig, RewardsComposerConfig
from configs.flatland.RewardConfigs import FinishRewardConfig
from env.flatland.params import SeveralAgents, PackOfAgents, LotsOfAgents
from environments import Env, FlatlandType, FLATLAND_OBS_SIZE, FLATLAND_ACTION_SIZE
from configs.EnvConfigs import StarCraftConfig, EnvCurriculumConfig

from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig

from agent.memory.DreamerMemory import DreamerMemory, ObsDataset

import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="starcraft", help='Flatland or SMAC env')
    parser.add_argument('--map_name', type=str, default="2m_vs_1z", help='Specific setting')
    parser.add_argument('--tokenizer', type=str, default='vq')
    parser.add_argument('--mamba_model_path', type=str, default=None)
    parser.add_argument('--marie_model_path', type=str, default=None)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=15)
    
    parser.add_argument('--ce_for_r', action='store_true')
    parser.add_argument('--ce_for_av', action='store_true')
    parser.add_argument('--ce_for_end', action='store_true')
    
    parser.add_argument('--temperature', type=float, default=1.0)
    
    return parser.parse_args()


# prepare environment
def get_env_info(configs, env):
    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = env.n_actions
        config.NUM_AGENTS = env.n_agents
    
    print(f'Observation dims: {env.n_obs}')
    print(f'Action dims: {env.n_actions}')
    print(f'Num agents: {env.n_agents}')
    env.close()


def get_env_info_flatland(configs):
    for config in configs:
        config.IN_DIM = FLATLAND_OBS_SIZE
        config.ACTION_SIZE = FLATLAND_ACTION_SIZE


def prepare_starcraft_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = StarCraftConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}


def prepare_flatland_configs(env_name):
    if env_name == FlatlandType.FIVE_AGENTS:
        env_config = SeveralAgents(RANDOM_SEED + 100)
    elif env_name == FlatlandType.TEN_AGENTS:
        env_config = PackOfAgents(RANDOM_SEED + 100)
    elif env_name == FlatlandType.FIFTEEN_AGENTS:
        env_config = LotsOfAgents(RANDOM_SEED + 100)
    else:
        raise Exception("Unknown flatland environment")
    obs_builder_config = SimpleObservationConfig(max_depth=3, neighbours_depth=3,
                                                 timetable_config=AllAgentLauncherConfig())
    reward_config = RewardsComposerConfig((FinishRewardConfig(finish_value=10),
                                           NearRewardConfig(coeff=0.01),
                                           DeadlockPunishmentConfig(value=-5)))
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    get_env_info_flatland(agent_configs)
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": reward_config,
            "obs_builder_config": obs_builder_config}

# ----------------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    RANDOM_SEED = 12345
    if args.env == Env.FLATLAND:
        raise NotImplementedError("Currently, visulization does not support FLATLAND env.")
    elif args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.map_name)
    else:
        raise Exception("Unknown environment")
    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    configs["learner_config"].tokenizer_type = args.tokenizer
    configs["controller_config"].tokenizer_type = args.tokenizer
    
    configs["learner_config"].use_ce_for_r = args.ce_for_r
    configs["learner_config"].use_ce_for_end = False  # args.ce_for_end
    configs["learner_config"].use_ce_for_av_action = args.ce_for_av
    
    rewards_prediction_config = {
        'loss_type': 'hlgauss',
        'min_v': -1., 
        'max_v': 3.,
        'bins': 50,
    }
    
    configs["learner_config"].rewards_prediction_config = rewards_prediction_config
    configs["learner_config"].HORIZON = args.horizon
    configs["learner_config"].update()
    
    device = configs["learner_config"].DEVICE
    
    # loading mamba model
    mamba_model = None
    if args.mamba_model_path is not None:
        replay_prefix = "mamba"
        mamba_model = load_mamba_model(configs["learner_config"], args.mamba_model_path)
        
        @torch.no_grad()
        def mamba_select_actions(obser, avail_action, prev_actions, prev_rnn_state):
            obser = obser.unsqueeze(0)
            avail_action = avail_action.unsqueeze(0)
            
            state = mamba_model["model"](obser, prev_actions, prev_rnn_state, None)
            feats = state.get_features()
            
            action, pi = actor(feats)
            pi[avail_action == 0] = -1e10
            pi = pi / args.temperature  # softmax temperature

            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample()
            
            return action, deepcopy(state)
    
    # loading marie model
    marie_model = None
    if args.marie_model_path is not None:
        replay_prefix = "marie"
        marie_model = load_mawm_model(configs["learner_config"], args.marie_model_path)
        
        if configs["learner_config"].use_stack:
            stack_obs = deque(maxlen=configs["learner_config"].stack_obs_num)
            for _ in range(configs["learner_config"].stack_obs_num):
                stack_obs.append(
                    torch.zeros(configs["learner_config"].NUM_AGENTS, configs["learner_config"].IN_DIM, device=configs["learner_config"].DEVICE, dtype=torch.float32)
                )
        
        @torch.no_grad()
        def marie_select_actions(obser, avail_action):
            if not configs["learner_config"].use_bin:
                feats = marie_model["tokenizer"].encode_decode(obser, True, True)
            else:
                tokens = discretize_into_bins(obser, configs["learner_config"].bins)
                feats = bins2continuous(tokens, configs["learner_config"].bins)
            
            if configs["learner_config"].use_stack:
                stack_obs.append(feats)
                feat = rearrange(torch.stack(list(stack_obs), dim=0), 'b n e -> n (b e)')
            else:
                feat = feats
            
            action, pi = actor(feat)
            pi[avail_action == 0] = -1e10
            pi = pi / args.temperature  # softmax temperature
            
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample()
            
            return action
            
    
    # initialize raw env
    env = StarCraft2Env(
        map_name=args.map_name,
        continuing_episode=True,
        seed=RANDOM_SEED,
        replay_prefix=replay_prefix + f"_{args.map_name}",
        replay_dir="/mnt/data/optimal/zhangyang/SC2_Replay",
    )

    '''
    evaluation for MARIE
    '''
    print("Start evaluation for MARIE...")
    actor = marie_model["actor"]
    replay_prefix = "marie"

    marie_trajs = []
    ## reset env
    for idx in range(args.eval_episodes):
        # for storation of sampled traj
        observations = []
        actions = []
        stored_rewards = []
        av_actions = []
        #######

        rewards = []
        prev_rnn_state = None
        prev_actions = None
        
        obs, _ = env.reset()
        obs = torch.tensor(np.array(obs)).to(device)
        done = False
        
        # sample a trajectory
        while not done:
            av_action = torch.tensor(env.get_avail_actions()).to(device)

            if replay_prefix == "mamba":
                action, prev_rnn_state = mamba_select_actions(obs, av_action, prev_actions, prev_rnn_state)
                prev_actions = action.clone()
                action = action.squeeze(0)
            elif replay_prefix == 'marie':
                action = marie_select_actions(obs, av_action)
            else:
                action_dist = OneHotCategorical(probs=av_action / av_action.sum(-1, keepdim=True))
                action = action_dist.sample()
            
            reward, done, info = env.step([ac.argmax() for i, ac in enumerate(action)])
            
            observations.append(obs)
            actions.append(action)
            stored_rewards.append(torch.ones(action.shape[0], 1, device=device, dtype=torch.float32) * reward)
            av_actions.append(av_action)
            
            obs = torch.tensor(np.array(env.get_obs())).to(device)
            
            rewards.append(reward)
        
        print(
            f"Visualize {idx}th episode - "
            + f"take {len(rewards)} timesteps | "
            + f"battle won: {info['battle_won']} | "
            + f"meet episode limit = {info.get('episode_limit', False)} | "
            + f"returns: {np.sum(rewards)}"
        )

        sample = {
            "observations": torch.stack(observations, dim=0).to(configs["learner_config"].DEVICE),
            "actions": torch.stack(actions, dim=0).to(configs["learner_config"].DEVICE),
            "rewards": torch.stack(stored_rewards, dim=0).to(configs["learner_config"].DEVICE),
            "av_actions": torch.stack(av_actions, dim=0).to(configs["learner_config"].DEVICE),
        }
        marie_trajs.append(sample)


    '''
    evaluation for MAMBA
    '''
    print("Start evaluation for MAMBA...")
    actor = mamba_model["actor"]
    replay_prefix = "mamba"

    mamba_trajs = []
    ## reset env
    for idx in range(args.eval_episodes):
        # for storation of sampled traj
        observations = []
        actions = []
        stored_rewards = []
        av_actions = []
        #######

        rewards = []
        prev_rnn_state = None
        prev_actions = None
        
        obs, _ = env.reset()
        obs = torch.tensor(np.array(obs)).to(device)
        done = False
        
        # sample a trajectory
        while not done:
            av_action = torch.tensor(env.get_avail_actions()).to(device)

            if replay_prefix == "mamba":
                action, prev_rnn_state = mamba_select_actions(obs, av_action, prev_actions, prev_rnn_state)
                prev_actions = action.clone()
                action = action.squeeze(0)
            else:
                action = marie_select_actions(obs, av_action)
            
            reward, done, info = env.step([ac.argmax() for i, ac in enumerate(action)])

            observations.append(obs)
            actions.append(action)
            stored_rewards.append(torch.ones(action.shape[0], 1, device=device, dtype=torch.float32) * reward)
            av_actions.append(av_action)

            obs = torch.tensor(np.array(env.get_obs())).to(device)
            
            rewards.append(reward)
        
        print(
            f"Visualize {idx}th episode - "
            + f"take {len(rewards)} timesteps | "
            + f"battle won: {info['battle_won']} | "
            + f"meet episode limit = {info.get('episode_limit', False)} | "
            + f"returns: {np.sum(rewards)}"
        )

        sample = {
            "observations": torch.stack(observations, dim=0).to(configs["learner_config"].DEVICE),
            "actions": torch.stack(actions, dim=0).to(configs["learner_config"].DEVICE),
            "rewards": torch.stack(stored_rewards, dim=0).to(configs["learner_config"].DEVICE),
            "av_actions": torch.stack(av_actions, dim=0).to(configs["learner_config"].DEVICE),
        }
        mamba_trajs.append(sample)
    
    # env.save_replay()
    env.close()

    trajs = mamba_trajs + marie_trajs
    import random
    random.shuffle(trajs)

    eval_errors_steps = 100
    test_horizons = args.horizon
    n_agents = configs["learner_config"].NUM_AGENTS

    ## error
    marie_obs_l1_errors = []
    marie_obs_l2_errors = []
    marie_r_l1_errors = []

    mamba_obs_l1_errors = []
    mamba_obs_l2_errors = []
    mamba_r_l1_errors = []
    
    for i in range(eval_errors_steps):
        print(f'Current test step: {i + 1}')
        # sample segment for computing errors
        sampled_traj = random.choice(trajs)
        length = sampled_traj["observations"].shape[0]
        start = np.random.randint(1, length - test_horizons)
        end = start + test_horizons
        
        # for MARIE
        splitted_segment4marie = {k: v[start:end] for k, v in sampled_traj.items()}
        
        # for MAMBA
        splitted_segment4mamba = {
            "observations": sampled_traj["observations"][start:end],
            "actions": sampled_traj["actions"][start - 1 : end],
            "rewards": sampled_traj["rewards"][start:end],
            "av_actions": sampled_traj["av_actions"][start:end],
            "last": torch.zeros_like(sampled_traj["rewards"][start:end], device=sampled_traj["observations"].device)
        }

        _, marie_error_dict = _compute_mawm_errors(marie_model, splitted_segment4marie, test_horizons)
        _, mamba_error_dict = _compute_mamba_errors(mamba_model, splitted_segment4mamba, test_horizons)

        marie_obs_l1_errors.append(marie_error_dict['obs_l1_errors'])
        marie_obs_l2_errors.append(marie_error_dict['obs_l2_errors'])
        marie_r_l1_errors.append(marie_error_dict['r_errors'])

        mamba_obs_l1_errors.append(mamba_error_dict['obs_l1_errors'])
        mamba_obs_l2_errors.append(mamba_error_dict['obs_l2_errors'])
        mamba_r_l1_errors.append(mamba_error_dict['r_errors'])

    marie_obs_l1_errors = np.stack(marie_obs_l1_errors, axis=0)
    marie_obs_l2_errors = np.stack(marie_obs_l2_errors, axis=0)
    marie_r_l1_errors = np.stack(marie_r_l1_errors, axis=0).squeeze(-1)

    mamba_obs_l1_errors = np.stack(mamba_obs_l1_errors, axis=0)
    mamba_obs_l2_errors = np.stack(mamba_obs_l2_errors, axis=0)
    mamba_r_l1_errors = np.stack(mamba_r_l1_errors, axis=0).squeeze(-1)
    
    ipdb.set_trace()

    import matplotlib.pyplot as plt

    steps = np.arange(test_horizons) + 1    
    fig = plt.figure(figsize=(16, 4))
    for agent_id in range(n_agents):
        plt.subplot(1, 3, agent_id + 1)

        mean_marie_errors = marie_obs_l2_errors.mean(0)[:, agent_id]
        cumulative_marie_errors = np.cumsum(mean_marie_errors)
        
        mean_mamba_errors = mamba_obs_l1_errors.mean(0)[:, agent_id]
        cumulative_mamba_errors = np.cumsum(mean_mamba_errors)

        plt.plot(steps, cumulative_marie_errors, color='blue', label=f'MARIE')
        plt.fill_between(steps, cumulative_marie_errors - marie_obs_l1_errors.std(0)[:, agent_id], cumulative_marie_errors + marie_obs_l1_errors.std(0)[:, agent_id], color='blue', alpha=0.2,)

        plt.plot(steps, cumulative_mamba_errors, color='red', label=f'MAMBA')
        plt.fill_between(steps, cumulative_mamba_errors - mamba_obs_l1_errors.std(0)[:, agent_id], cumulative_mamba_errors + mamba_obs_l1_errors.std(0)[:, agent_id], color='red', alpha=0.2,)
        
        plt.xlabel('Horizons', fontsize=14)
        plt.ylabel('Accumulated Errors', fontsize=14)
        plt.title(f'Agent {agent_id + 1}', fontsize=14)
        plt.grid(True)

    handles, labels = plt.gca().get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=6, fontsize=14)
    fig.tight_layout()
    plt.subplots_adjust(top=0.90)

    plt.savefig(f'cum_errors.pdf', bbox_inches='tight')

    plt.close()

    ipdb.set_trace()
    print('finish...')


    stored_dict = {
        'marie_obs_l1_errors': marie_obs_l1_errors,
        'marie_obs_l2_errors': marie_obs_l2_errors,
        'marie_r_l1_errors': marie_r_l1_errors,

        'mamba_obs_l1_errors': mamba_obs_l1_errors,
        'mamba_obs_l2_errors': mamba_obs_l2_errors,
        'mamba_r_l1_errors': mamba_r_l1_errors,
    }
    
    import pickle
    with open('cum_errors.pkl', 'wb') as f:
        pickle.dump(stored_dict, f)
