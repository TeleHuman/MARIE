from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.distributions import OneHotCategorical
from networks.transformer.layers import AttentionEncoder
from networks.dreamer.utils import build_model
from networks.dreamer.mlp_base import MLPBase

def get_init_method(initialization_method):
    """Get the initialization method.
    Args:
        initialization_method: (str) initialization method
    Returns:
        initialization method: (torch.nn) initialization method
    """
    return nn.init.__dict__[initialization_method]

def init(module, weight_init, bias_init, gain=1):
    """Init module.
    Args:
        module: (torch.nn) module
        weight_init: (torch.nn) weight init
        bias_init: (torch.nn) bias init
        gain: (float) gain
    Returns:
        module: (torch.nn) module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class FixedNormal(torch.distributions.Normal):
    """Modify standard PyTorch Normal."""

    def log_probs(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

class Categorical(nn.Module):
    """A linear layer followed by a Categorical distribution."""

    def __init__(
        self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01
    ):
        super(Categorical, self).__init__()
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return x

EPS = 0.01

class DiagGaussian(nn.Module):
    """A linear layer followed by a Diagonal Gaussian distribution."""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        initialization_method="orthogonal_",
        gain=0.01,
        args=None,
    ):
        super(DiagGaussian, self).__init__()

        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if args is not None:
            self.std_x_coef = args["std_x_coef"]
            self.std_y_coef = args["std_y_coef"]
        else:
            self.std_x_coef = 1.0
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef

        eps = torch.finfo(action_mean.dtype).eps
        action_std = action_std.clamp(min=eps)

        return FixedNormal(action_mean, action_std)

class FeatureNormedActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size = 64,
                 use_feature_normalization: bool = True, activation=nn.ReLU):
        super().__init__()
        
        self.use_feature_normalization = use_feature_normalization
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.fc1 = nn.Sequential(nn.Linear(obs_dim, hidden_size), activation(), nn.LayerNorm(hidden_size) if self.use_feature_normalization else nn.Identity())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), activation(), nn.LayerNorm(hidden_size) if self.use_feature_normalization else nn.Identity())
        self.act_layer = nn.Linear(hidden_size, action_dim)
    
    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act_layer(x)
        
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()
        return action, x 

class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU):
        super().__init__()

        self.feedforward_model = build_model(in_dim, out_dim, layers, hidden_size, activation)

    def forward(self, state_features):
        x = self.feedforward_model(state_features)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()
        return action, x

class StochasticPolicy(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU,
                 continuous_action: bool = False, continuous_action_space = None):
        super().__init__()

        self.continuous_action = continuous_action
        if self.continuous_action:
            assert continuous_action_space is not None
            self.continuous_action_space = continuous_action_space
            action_dim = self.continuous_action_space.shape[0]
            self.initialization_method = "orthogonal_"
            self.gain = 0.01

            self.action_out = DiagGaussian(
                hidden_size, action_dim, self.initialization_method, self.gain
            )
        else:
            action_dim = out_dim
            self.initialization_method = "orthogonal_"
            self.gain = 0.01

            self.action_out = Categorical(
                hidden_size, action_dim, self.initialization_method, self.gain
            )
            # raise NotImplementedError("Currently not supported for Discrete action control.")
        
        # self.base = nn.Sequential(
        #     build_model(in_dim, hidden_size, layers, hidden_size, activation),
        #     activation(),
        # )
        self.base = MLPBase(in_dim, [hidden_size] * layers)
        self.dist_entropy = None

    def forward(self, state_features, deterministic: bool = False):
        actor_features = self.base(state_features)
        if self.continuous_action:
            action_dist = self.action_out(actor_features)

            self.dist_entropy = action_dist.entropy().detach()

            actions = (
                action_dist.mode()
                if deterministic
                else action_dist.sample()
            )
            action_log_probs = action_dist.log_probs(actions)

            return actions, action_log_probs

        else:
            logits = self.action_out(actor_features)
            action_dist = OneHotCategorical(logits=logits)
            action = action_dist.sample()
            return action, logits
    
    # used for continuous actions
    def evaluate_actions(self, state_features, actions):
        actor_features = self.base(state_features)
        action_dist = self.action_out(actor_features)
        
        action_log_probs = action_dist.log_probs(actions)
        dist_entropy = action_dist.entropy().mean()

        return action_log_probs, dist_entropy, action_dist


class AttentionActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU):
        super().__init__()
        self.feedforward_model = build_model(hidden_size, out_dim, 1, hidden_size, activation)
        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size)
        self.embed = nn.Linear(in_dim, hidden_size)

    def forward(self, state_features):
        n_agents = state_features.shape[-2]
        batch_size = state_features.shape[:-2]
        embeds = F.relu(self.embed(state_features))
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        attn_embeds = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
        x = self.feedforward_model(attn_embeds)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()
        return action, x


class RNNActor(nn.Module):
    def __init__(self, in_dim, out_dim, n_agents, hidden_size = 512, layers = 3, use_original_obs: bool = False, activation=nn.ReLU) -> None:
        super().__init__()
        self.use_original_obs = use_original_obs
        self.n_agents = n_agents

        self.embed = build_model(in_dim, hidden_size, layers, activation=activation)

        self.lstm_dim = 512
        self.lstm = nn.LSTMCell(1024, self.lstm_dim)
        self.hx, self.cx = None, None

        self.actor_linear = nn.Linear(512, out_dim)

    def __repr__(self) -> str:
        return "actor_critic"

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None) -> None:
        device = burnin_observations.device
        self.hx = torch.zeros(n, self.lstm_dim, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, device=device)
        if burnin_observations is not None:
            assert burnin_observations.ndim == 3 and burnin_observations.size(0) == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding[:, i])

    def forward(self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None):
        assert inputs.ndim == 2  # input shape: (batch size * n_agents, input_dim)
        assert -1 <= inputs.min() <= 1 and -1 <= inputs.max() <= 1
        assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0) and mask_padding.any())
        x = inputs[mask_padding] if mask_padding is not None else inputs

        b, e = x.shape
        # x = x.mul(2).sub(1)
        x = self.embed(x)

        if mask_padding is None:
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        else:
            self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x, (self.hx[mask_padding], self.cx[mask_padding]))

        logits_actions = self.actor_linear(self.hx)  # rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')

        return logits_actions