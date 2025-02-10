from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.dreamer.DreamerAgentConfig import DreamerConfig

from functools import partial

RSSM_STATE_MODE = 'discrete'


class MPEDreamerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.ACTION_SIZE = 9
        self.ACTION_LAYERS = 1
        self.ACTION_HIDDEN = 128  # 256

        self.use_bin = True
        self.bins = 256
        self.action_bins = 256

        ## debug
        self.use_stack = False
        self.stack_obs_num = 5
