import argparse
import os
import shutil
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
from pathlib import Path
from utils import load_mamba_model, load_marie_model, _wrap, compute_compounding_errors

from configs.flatland.TimetableConfigs import AllAgentLauncherConfig
from env.flatland.params import SeveralAgents, PackOfAgents, LotsOfAgents
from environments import Env, FlatlandType, FLATLAND_OBS_SIZE, FLATLAND_ACTION_SIZE
from configs.EnvConfigs import StarCraftConfig, EnvCurriculumConfig

from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig

import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="flatland", help='Flatland or SMAC env')
    parser.add_argument('--env_name', type=str, default="5_agents", help='Specific setting')
    parser.add_argument('--tokenizer', type=str, default='vq')
    
    parser.add_argument('--ce_for_r', action='store_true')
    parser.add_argument('--ce_for_av', action='store_true')
    parser.add_argument('--ce_for_end', action='store_true')
    
    parser.add_argument('--marie_load_path', type=str, default=None)
    parser.add_argument('--mamba_load_path', type=str, default=None)
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
    
    RANDOM_SEED = 12345
    if args.env == Env.FLATLAND:
        configs = prepare_flatland_configs(args.env_name)
    elif args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
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

    # loading 
    check_mode = 'dual'
    if args.mamba_load_path is None:
        assert args.mawm_load_path is not None
        check_mode = 'marie'
    
    if args.mawm_load_path is None:
        assert args.mamba_load_path is not None
        check_mode = 'mamba'

    models = []
    if check_mode == 'dual':
        mamba_model = load_mamba_model(configs["learner_config"], args.mamba_load_path)
        mawm_model = load_marie_model(configs["learner_config"], args.mawm_load_path)
        models.append(mamba_model)
        models.append(mawm_model)
        
    elif check_mode == 'marie':
        mawm_model = load_marie_model(configs["learner_config"], args.mawm_load_path)
        models.append(mawm_model)
        
    elif check_mode == 'mamba':
        mamba_model = load_mamba_model(configs["learner_config"], args.mamba_load_path)
        models.append(mamba_model)
    
    # initialize env
    env = configs["env_config"][0].create_env()
    obs = _wrap(env.reset())
    
    observations = []
    actions = []
    rewards = []
    av_actions = []
    done = False
    # sample a trajectory
    while not done:
        av_action = torch.tensor(env.get_avail_actions())

        # random select actions
        action_dist = OneHotCategorical(probs=av_action / av_action.sum(-1, keepdim=True))
        action = action_dist.sample()
        
        next_obs, reward, done, info = env.step([ac.argmax() for i, ac in enumerate(action)])
        
        observations.append(obs)
        actions.append(action)
        rewards.append(_wrap(reward))
        av_actions.append(av_action)
        
        obs = _wrap(next_obs)
        done = all([v for k, v in done.items()])
        
    env.close()
        
    print(f"Sampled a {len(observations)}-long traj")
    
    sample = {
        "observations": torch.stack(observations, dim=0).to(configs["learner_config"].DEVICE),
        "actions": torch.stack(actions, dim=0).to(configs["learner_config"].DEVICE),
        "rewards": torch.stack(rewards, dim=0).to(configs["learner_config"].DEVICE),
        "av_actions": torch.stack(av_actions, dim=0).to(configs["learner_config"].DEVICE),
    }
    
    compute_compounding_errors(models, sample, configs["learner_config"].HORIZON)

    # exp = Experiment(steps=args.steps,
    #                  episodes=50000,
    #                  random_seed=RANDOM_SEED,
                    #  env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                    #                                 obs_builder_config=configs["obs_builder_config"],
                    #                                 reward_config=configs["reward_config"]),
    #                  controller_config=configs["controller_config"],
    #                  learner_config=configs["learner_config"])

    # train_dreamer(exp, n_workers=args.n_workers)
