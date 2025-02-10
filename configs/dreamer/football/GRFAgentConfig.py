from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.Config import Config
from agent.models.tokenizer import StateEncoderConfig
from agent.models.transformer import PerceiverConfig, TransformerConfig
from configs.dreamer.DreamerAgentConfig import DreamerConfig

from functools import partial

RSSM_STATE_MODE = 'discrete'


class GRFDreamerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.ACTION_HIDDEN = 64

        ## discretize params
        self.use_bin = False
        self.bins = 512

        # tokenizer params
        self.nums_obs_token = 16 # 4
        self.hidden_sizes = [128, 128]
        self.alpha = 1.0
        self.EMBED_DIM = 32 # 128
        self.OBS_VOCAB_SIZE = 512 # 512

        self.alpha = 10.
        self.ema_decay = 0.8

        self.encoder_config_fn = partial(StateEncoderConfig,
            nums_obs_token=self.nums_obs_token, 
            hidden_sizes=self.hidden_sizes,
            alpha=1.0,
            z_channels=self.EMBED_DIM * self.nums_obs_token
        )
        
        # world model params
        self.HORIZON = 15  # 15
        self.TRANS_EMBED_DIM = 256 # 256
        self.HEADS = 4
        self.perattn_HEADS = 4
        self.DROPOUT = 0.1
        
        # lack "num_latents" which should be equal to NUM_AGENTS
        self.perattn_config = partial(PerceiverConfig,
            dim=self.TRANS_EMBED_DIM,
            latent_dim=self.TRANS_EMBED_DIM,
            depth=2,
            cross_heads=8, # 1
            cross_dim_head=64,
            latent_heads=8,
            latent_dim_head=64,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

        self.trans_config = partial(TransformerConfig,
            # tokens_per_block=self.nums_obs_token + 1 + 1,
            max_blocks=self.HORIZON,
            attention='causal',
            num_layers=10, # 10
            num_heads=self.HEADS,
            embed_dim=self.TRANS_EMBED_DIM,
            embed_pdrop=self.DROPOUT,
            resid_pdrop=self.DROPOUT,
            attn_pdrop=self.DROPOUT,
        )

        # 这里修改一下
        self.FEAT = self.STOCHASTIC + self.DETERMINISTIC
        # self.FEAT = self.EMBED_DIM * self.nums_obs_token
        self.critic_FEAT = self.TRANS_EMBED_DIM # * self.nums_obs_token # self.TRANS_EMBED_DIM
        self.GLOBAL_FEAT = self.FEAT + self.EMBED

        ## debug
        self.use_stack = True
        self.stack_obs_num = 4
