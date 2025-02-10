
import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime

from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from configs.EnvConfigs import StarCraftConfig
from environments import Env

import wandb

offline_dataset_path = {
    '2m_vs_1z_20K': "/xx/xx.pkl",
    '2m_vs_1z_50K': "/xx/xx.pkl",
    'so_many_baneling_30K': "/xx/xx.pkl", 
    'so_many_baneling_50K': "/xx/xx.pkl",
}


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--env', type=str, default='starcraft')
    parser.add_argument('--map', type=str, default='3m')
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=500)
    
    parser.add_argument('--tokenizer', type=str, default='vq')
    parser.add_argument('--decay', type=float, default=0.8)

    parser.add_argument('--ce_for_av', action='store_true')
    parser.add_argument('--ce_for_end', action='store_true')
    
    ## wandb mode
    parser.add_argument('--mode', type=str, default='disabled')
    
    return parser

def main(args):
    learner_config = DreamerLearnerConfig()
    
    if args.env == 'starcraft':
        env_config = StarCraftConfig(args.map, 1234)
        env = env_config.create_env()
    
    learner_config.ENV_TYPE = Env(args.env)
    learner_config.IN_DIM = env.n_obs
    learner_config.ACTION_SIZE = env.n_actions
    learner_config.NUM_AGENTS = env.n_agents
    
    ## prepare for loading offline dataset
    learner_config.load_path = offline_dataset_path.get(f"{args.map}_{args.num_steps // 1000}K", None)
    assert learner_config.load_path is not None, f"No {args.map}_{args.num_steps // 1000}K offline dataset"
    learner_config.is_preload = True
    learner_config.use_external_rew_model = False
    learner_config.use_stack = False
    
    ## setting tokenizer type
    learner_config.tokenizer_type = args.tokenizer
    learner_config.ema_decay = args.decay
    learner_config.sample_temperature = 'inf'

    ## setting world model parameters
    learner_config.use_classification = args.ce_for_end
    learner_config.use_ce_for_av_action = args.ce_for_av
    post_fix = ""
    if args.ce_for_end:
        post_fix += "ce_on_end"
    
    if args.ce_for_av:
        post_fix += "_ce_on_av_"
    
    ## make run directory
    save_path = Path("pretrained_weights") / f'new_pretrained_wm' / (f"marie_{args.map}_{learner_config.tokenizer_type}_{args.num_steps // 1000}K_obs{learner_config.nums_obs_token}_" + post_fix + datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3])
    save_path.mkdir(parents=True, exist_ok=True)
    learner_config.RUN_DIR = str(save_path)
    print(f"Run is saved to {str(save_path)}...")
    
    ## wandb initialize
    wandb.init(
        config=learner_config.to_dict(),
        mode=args.mode,
        project="xxx",
        group="(pretrain) marie",
        name=f"marie_{args.map}_{learner_config.tokenizer_type}_{args.num_steps // 1000}K_obs{learner_config.nums_obs_token}" + post_fix,
    )
    
    # do training
    learner = learner_config.create_learner()
    learner.train_wm_offline(args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MARIE pre-training with offline dataset', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)