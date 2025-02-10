import argparse
import os
import shutil
import torch

from einops import rearrange, repeat
from torch.distributions.one_hot_categorical import OneHotCategorical
from pathlib import Path
from utils import load_mamba_model, load_mawm_model, _wrap, bins2continuous, discretize_into_bins
from utils import obs_bins2continuous, symexp, symlog, obs_split_into_bins
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
from configs.EnvConfigs import StarCraftConfig, EnvCurriculumConfig, PettingZooConfig, FootballConfig, MAMujocoConfig

from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig

# for MPE
from configs.dreamer.mpe.MpeLearnerConfig import MPEDreamerLearnerConfig
from configs.dreamer.mpe.MpeControllerConfig import MPEDreamerControllerConfig

# for GRF
from configs.dreamer.football.GRFLearnerConfig import GRFDreamerLearnerConfig
from configs.dreamer.football.GRFControllerConfig import GRFDreamerControllerConfig

# for MAMuJoCo
from configs.dreamer.mamujoco.mamujocoLearnerConfig import MAMujocoDreamerLearnerConfig
from configs.dreamer.mamujoco.mamujocoControllerConfig import MAMujocoDreamerControllerConfig

from agent.memory.DreamerMemory import DreamerMemory, ObsDataset

import ipdb

## ---------- GRF ------------------
env_num_agents = {
    "academy_pass_and_shoot_with_keeper": 2,
    "academy_run_pass_and_shoot_with_keeper": 2,
    "academy_3_vs_1_with_keeper": 3,
    "academy_counterattack_easy": 4,
    "academy_counterattack_hard": 4,
    "academy_corner": 11,
    "academy_single_goal_versus_lazy": 11,

    # additional env_names: 
    # https://github.com/google-research/football/blob/9a9e35bcd1929a82c2b91eeed777e5e571f29d38/gfootball/doc/scenarios.md?plain=1#L20
    "academy_empty_goal_close": 3,
    "academy_empty_goal": 3,
    "academy_run_to_score": 3,
    "academy_run_to_score_with_keeper": 3,
}
## --------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="starcraft", help='Flatland or SMAC env')
    parser.add_argument('--env_name', type=str, default="5_agents", help='Specific setting')
    parser.add_argument('--tokenizer', type=str, default='vq')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--eval_episodes', type=int, default=1)
    
    parser.add_argument('--ce_for_r', action='store_true')
    parser.add_argument('--ce_for_av', action='store_true')
    parser.add_argument('--ce_for_end', action='store_true')
    
    parser.add_argument('--temperature', type=float, default=1.0)
    
    return parser.parse_args()


# prepare environment
def get_env_info(configs, env):
    if not env.discrete:
        assert hasattr(env, 'individual_action_space')
        individual_action_space = env.individual_action_space
    else:
        individual_action_space = None

    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = env.n_actions
        config.NUM_AGENTS = env.n_agents
        config.CONTINUOUS_ACTION = not env.discrete
        config.ACTION_SPACE = individual_action_space
    
    print(f'Observation dims: {env.n_obs}')
    print(f'Action dims: {env.n_actions}')
    print(f'Num agents: {env.n_agents}')
    print(f'Continuous action for control? -> {not env.discrete}')
    
    if hasattr(env, 'individual_action_space'):
        print(f'Individual action space: {env.individual_action_space}')

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

def prepare_pettingzoo_configs(env_name, continuous_action = True):
    agent_configs = [MPEDreamerControllerConfig(), MPEDreamerLearnerConfig()]
    env_config = PettingZooConfig(env_name, RANDOM_SEED, continuous_action)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_football_configs(env_name):
    agent_configs = [GRFDreamerControllerConfig(), GRFDreamerLearnerConfig()]
    env_config = FootballConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_mamujoco_configs(scenario, agent_config):
    agent_configs = [MAMujocoDreamerControllerConfig(), MAMujocoDreamerLearnerConfig()]
    env_config = MAMujocoConfig(scenario = scenario, seed = RANDOM_SEED, agent_conf = agent_config)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

# ----------------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    RANDOM_SEED = 12345
    # if args.env == Env.FLATLAND:
    #     raise NotImplementedError("Currently, visulization does not support FLATLAND env.")
    # elif args.env == Env.STARCRAFT:
    #     configs = prepare_starcraft_configs(args.map_name)
    # else:
    #     raise Exception("Unknown environment")
    
    if args.env == Env.FLATLAND:
        raise NotImplementedError("Currently, visulization does not support FLATLAND env.")
    elif args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
    elif args.env == Env.PETTINGZOO:
        configs = prepare_pettingzoo_configs(args.env_name, continuous_action=True)
    elif args.env == Env.GRF:
        configs = prepare_football_configs(args.env_name)
    elif args.env == Env.MAMUJOCO:
        configs = prepare_mamujoco_configs(args.env_name, args.agent_conf)
    else:
        raise Exception("Unknown environment")

    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    configs["learner_config"].seed = RANDOM_SEED

    configs["learner_config"].tokenizer_type = args.tokenizer
    configs["controller_config"].tokenizer_type = args.tokenizer
    
    configs["learner_config"].use_ce_for_r = args.ce_for_r
    configs["learner_config"].use_ce_for_end = args.ce_for_end
    configs["learner_config"].use_ce_for_av_action = args.ce_for_av
    
    if args.env != Env.MAMUJOCO:
        rewards_prediction_config = {
            'loss_type': 'hlgauss',
            'min_v': -1., 
            'max_v': 3.,
            'bins': 50,
        }
        
        configs["learner_config"].rewards_prediction_config = rewards_prediction_config
    
    device = configs["learner_config"].DEVICE
    
    # create databuffer
    config = configs["learner_config"]
    mamba_replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, config.NUM_AGENTS,
                                        config.DEVICE, config.ENV_TYPE, 'inf')
    
    # loading model
    if "mamba" in args.model_path and "Trans-mamba" not in args.model_path:
        replay_prefix = "mamba"
        model = load_mamba_model(configs["learner_config"], args.model_path)
        
        @torch.no_grad()
        def select_actions(obser, avail_action, prev_actions, prev_rnn_state):
            obser = obser.unsqueeze(0)
            avail_action = avail_action.unsqueeze(0)
            
            state = model["model"](obser, prev_actions, prev_rnn_state, None)
            feats = state.get_features()
            
            action, pi = actor(feats)
            pi[avail_action == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample()
            
            return action, deepcopy(state)
        
    else:
        replay_prefix = "marie"
        model = load_mawm_model(configs["learner_config"], args.model_path)
        
        if configs["learner_config"].use_stack:
            stack_obs = deque(maxlen=configs["learner_config"].stack_obs_num)
            for _ in range(configs["learner_config"].stack_obs_num):
                stack_obs.append(
                    torch.zeros(configs["learner_config"].NUM_AGENTS, configs["learner_config"].IN_DIM, device=configs["learner_config"].DEVICE, dtype=torch.float32)
                )
        
        @torch.no_grad()
        def select_actions(obser, avail_action):
            if not configs["learner_config"].use_bin:
                feats = model["tokenizer"].encode_decode(obser, True, True)
            else:
                if args.env == Env.GRF:
                    obser = symlog(obser)
                
                    tokens = obs_split_into_bins(observations, configs["learner_config"].bins, low=-3., high=3.)
                    feats = obs_bins2continuous(tokens, configs["learner_config"].bins, low=-3., high=3.)

                    feats = symexp(feats)
                else:
                    tokens = discretize_into_bins(obser, configs["learner_config"].bins)
                    feats = bins2continuous(tokens, configs["learner_config"].bins)
            
            if configs["learner_config"].use_stack:
                stack_obs.append(feats)
                feat = rearrange(torch.stack(list(stack_obs), dim=0), 'b n e -> n (b e)')
            else:
                feat = feats
            

            if configs["learner_config"].CONTINUOUS_ACTION:
                action, log_probs = actor(feat, deterministic = True)
            else:
                action, pi = actor(feat)
                if avail_action is not None:
                    pi[avail_action == 0] = -1e10  # logits

                pi = pi / args.temperature  # softmax temperature
                action_dist = OneHotCategorical(logits=pi)
                action = action_dist.sample()

            return action
    
    actor = model["actor"]
    
    # initialize raw env
    if args.env == Env.STARCRAFT:
        env = StarCraft2Env(
            map_name=args.env_name,
            continuing_episode=True,
            seed=RANDOM_SEED,
            replay_prefix=replay_prefix + f"_{args.env_name}",
            replay_dir="/mnt/data/optimal/zhangyang/SC2_Replay",
        )
    elif args.env == Env.GRF:
        import gfootball.env as football_env
        kwargs = {}
        kwargs['dump_frequency'] = 1
        kwargs['extra_players'] = None
        kwargs['logdir'] = f'grf_replays/{args.env_name}' # None
        kwargs['channel_dimensions'] = (96, 72)
        kwargs['representation'] = "simple115v2"
        kwargs['rewards'] = "scoring,checkpoints"
        kwargs["stacked"] = False

        kwargs['write_full_episode_dumps'] = True # False
        kwargs['write_goal_dumps'] = True # False
        kwargs['write_video'] = True # False
        kwargs['render'] = False
        kwargs['other_config_options'] = None

        kwargs['env_name'] = args.env_name
        
        if args.env_name in ["academy_run_to_score", "academy_run_to_score_with_keeper"]:
            kwargs['number_of_left_players_agent_controls'] = 0 # env_num_agents[env_name]
            kwargs['number_of_right_players_agent_controls'] = env_num_agents[args.env_name]
        else:
            kwargs['number_of_left_players_agent_controls'] = env_num_agents[args.env_name]
            kwargs['number_of_right_players_agent_controls'] = 0

        if "channel_dimensions" in kwargs:
            kwargs["channel_dimensions"] = tuple(kwargs["channel_dimensions"])
        if "logdir" in kwargs and kwargs["logdir"] is None:
            kwargs["logdir"] = ""
        if "other_config_options" in kwargs and kwargs["other_config_options"] is None:
            kwargs["other_config_options"] = {}

        env = football_env.create_environment(**kwargs)
        # env = configs['env_config'][0].create_env()

    # reset env
    observations = []
    actions = []
    stored_rewards = []
    av_actions = []
    
    for idx in range(args.eval_episodes):
        if configs["learner_config"].use_stack:
            stack_obs = deque(maxlen=configs["learner_config"].stack_obs_num)
            for _ in range(configs["learner_config"].stack_obs_num):
                stack_obs.append(
                    torch.zeros(configs["learner_config"].NUM_AGENTS, configs["learner_config"].IN_DIM, device=configs["learner_config"].DEVICE, dtype=torch.float32)
                )

        rewards = []
        prev_rnn_state = None
        prev_actions = None
        
        obs = env.reset()
        obs = torch.tensor(np.array(obs)).to(device)
        done = False
        
        # sample a trajectory
        while not done:
            if args.env == Env.STARCRAFT:
                av_action = torch.tensor(env.get_avail_actions()).to(device)
            else:
                av_action = None

            if replay_prefix == "mamba":
                action, prev_rnn_state = select_actions(obs, av_action, prev_actions, prev_rnn_state)
                prev_actions = action.clone()
                action = action.squeeze(0)
            else:
                action = select_actions(obs, av_action)
            # ipdb.set_trace()
            if args.env == Env.STARCRAFT:
                reward, done, info = env.step([ac.argmax() for i, ac in enumerate(action)])
            
            elif args.env == Env.GRF:
                next_obs, reward, done, info = env.step([ac.cpu().numpy().argmax() for i, ac in enumerate(action)])
                reward = torch.from_numpy(reward).to(device = device, dtype=torch.float32)

            if idx == 0:
                observations.append(obs)
                actions.append(action)
                stored_rewards.append(torch.ones(action.shape[0], 1, device=device, dtype=torch.float32) * reward)
                av_actions.append(av_action)
            
            if args.env == Env.STARCRAFT:
                obs = torch.tensor(np.array(env.get_obs())).to(device)
            else:
                obs = torch.from_numpy(next_obs).to(device = device, dtype=torch.float32)
            
            rewards.append(reward.cpu().numpy().mean() if type(reward) == torch.Tensor else reward)
            
        print(
            f"Visualize {idx}th episode - "
            + f"take {len(rewards)} timesteps | "
            + f"meet episode limit = {info.get('episode_limit', False)} | "
            + f"returns: {np.sum(rewards)} | "
            + f"Score: {info['score_reward']}"
        )
    
    # env.save_replay()
    env.close()
    
    ## for attention visualization
    sample = {
        "observation": torch.stack(observations, dim=0).to(configs["learner_config"].DEVICE),
        "action": torch.stack(actions, dim=0).to(configs["learner_config"].DEVICE),
        "reward": torch.stack(stored_rewards, dim=0).to(configs["learner_config"].DEVICE),
        "av_action": torch.stack(av_actions, dim=0).to(configs["learner_config"].DEVICE) if av_action is not None else None,
    }
    
    def segment_sample(sample, start, horizon):
        new_sample = {}
        for k, v in sample.items():
            new_sample[k] = v[start : start + horizon].unsqueeze(0) if v is not None else None
        return new_sample
    
    vis_sample = segment_sample(sample, 20, 5)
    
    save_path = Path(os.path.dirname(os.path.abspath(__file__))).parent / "attn_vis" / "trans_attn" / "final"
    
    print(f"Saving visualization to {save_path}...")

    
    # ipdb.set_trace()

    # transformer = model['model']
    # tokenizer = model['tokenizer']
    
    # transformer.visualize_attn(vis_sample, tokenizer, save_path)
    
    # print('finish...')
