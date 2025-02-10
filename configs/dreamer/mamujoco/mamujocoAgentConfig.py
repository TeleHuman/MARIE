from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from agent.models.tokenizer import StateEncoderConfig
from configs.dreamer.DreamerAgentConfig import DreamerConfig

from functools import partial


class MAMujocoDreamerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.ACTION_SIZE = 9
        self.ACTION_LAYERS = 3
        self.ACTION_HIDDEN = 128  # 256

        self.use_bin = False
        self.bins = 256
        self.action_bins = 256

        self.use_valuenorm = True
        self.use_huber_loss = True
        self.use_clipped_value_loss = True
        self.huber_delta = 10.0

        # VQ parameters
        self.nums_obs_token = 16 # 4
        self.hidden_sizes = [512, 512]
        self.alpha = 1.0
        self.EMBED_DIM = 64 # 32
        self.OBS_VOCAB_SIZE = 512

        self.alpha = 10.
        self.ema_decay = 0.8

        self.encoder_config_fn = partial(StateEncoderConfig,
            nums_obs_token=self.nums_obs_token, 
            hidden_sizes=self.hidden_sizes,
            alpha=1.0,
            z_channels=self.EMBED_DIM * self.nums_obs_token
        )


        ## debug
        self.use_stack = False
        self.stack_obs_num = 5
