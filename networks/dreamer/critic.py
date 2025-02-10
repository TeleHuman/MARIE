import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.dreamer.mlp_base import MLPLayer, MLPBase

from networks.dreamer.utils import build_model
from networks.transformer.layers import AttentionEncoder


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_size, layers=2, activation=nn.ELU):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.activation = activation
        self.feedforward_model = build_model(in_dim, 1, layers, hidden_size, activation)

    def forward(self, state_features):
        return self.feedforward_model(state_features)

class FeatureNormedCritic(nn.Module):
    def __init__(self, in_dim, hidden_size = 64,
                 use_feature_normalization: bool = True, activation=nn.ReLU):
        super().__init__()
        
        self.use_feature_normalization = use_feature_normalization
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(in_dim)

        self.embed_layer = nn.Sequential(nn.Linear(in_dim, 256), activation(), nn.LayerNorm(256) if self.use_feature_normalization else nn.Identity())
        self._attention_stack = AttentionEncoder(1, 256, 256)
        
        self.fc1 = nn.Sequential(nn.Linear(256, hidden_size), activation(), nn.LayerNorm(hidden_size) if self.use_feature_normalization else nn.Identity())
        # self.fc2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), activation(), nn.LayerNorm(hidden_size) if self.use_feature_normalization else nn.Identity())
        self.v_out = nn.Linear(hidden_size, 1)
    
    def forward(self, obs):
        n_agents = obs.shape[-2]
        batch_size = obs.shape[:-2]
        
        if self.use_feature_normalization:
            x = self.feature_norm(obs)
        else:
            x = obs
        
        # project into embeds for self-attention
        embeds = self.embed_layer(x)
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        
        x = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
        x = self.fc1(x)
        # x = self.fc2(x)
        
        return self.v_out(x)


class AugmentedCritic(nn.Module):
    def __init__(self, in_dim, hidden_size, activation=nn.ReLU):
        super().__init__()
        self.feedforward_model = build_model(hidden_size, 1, 1, hidden_size, activation)
        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size)
        self.embed = nn.Linear(in_dim, hidden_size)
        # self.prior = build_model(in_dim, 1, 3, hidden_size, activation)

    def forward(self, state_features):
        n_agents = state_features.shape[-2]
        batch_size = state_features.shape[:-2]
        embeds = F.relu(self.embed(state_features))
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        attn_embeds = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
        return self.feedforward_model(attn_embeds)

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

def get_init_method(initialization_method):
    """Get the initialization method.
    Args:
        initialization_method: (str) initialization method
    Returns:
        initialization method: (torch.nn) initialization method
    """
    return nn.init.__dict__[initialization_method]

def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor

def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)

    return mod

class FeatureNormedAugmentedCritic(nn.Module):
    def __init__(self, in_dim, hidden_size, layers, feat_norm = True, activation = 'relu',
                 initialization_method = 'orthogonal_'):
        super().__init__()
        
        self.use_feature_normalization = feat_norm
        self.initialization_method = initialization_method
        self.activation = activation
        self.hidden_sizes = [hidden_size] * layers

        init_method = get_init_method(self.initialization_method)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(in_dim)

        self.base = MLPBase(
            hidden_size, self.hidden_sizes,
            use_feature_normalization = self.use_feature_normalization,
            initialization_method = self.initialization_method,
            activation_func=self.activation
        )

        self.embed = MLPLayer(
            in_dim, [hidden_size], self.initialization_method, self.activation
        )
        self.embed.fc.pop(2)

        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size, norm_first=True)
        self._attention_stack = initialize_weights(self._attention_stack)

        self.v_out = init_(nn.Linear(self.hidden_sizes[-1], 1))


    def forward(self, input_state):
        n_agents = input_state.shape[-2]
        batch_size = input_state.shape[:-2]


        # cent_obs = check(cent_obs).to(**self.tpdv)
        # rnn_states = check(rnn_states).to(**self.tpdv)
        # masks = check(masks).to(**self.tpdv)

        if self.use_feature_normalization:
            input_state = self.feature_norm(input_state)

        embeds = self.embed(input_state)
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        critic_feat = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))

        critic_feat = self.base(critic_feat)
        values = self.v_out(critic_feat)

        return values

class VNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, in_dim, hidden_size, layers, feat_norm = True, activation = 'relu',
                 initialization_method = 'orthogonal_'):
        """Initialize VNet model.
        """
        super(VNet, self).__init__()
        self.hidden_sizes = [hidden_size] * layers
        self.initialization_method = initialization_method
        self.activation = activation
        self.feat_norm = feat_norm
        init_method = get_init_method(self.initialization_method)

        self.base = MLPBase(in_dim, self.hidden_sizes, use_feature_normalization=self.feat_norm,
                            initialization_method=self.initialization_method,
                            activation_func=self.activation)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

    def forward(self, cent_obs):
        """Compute actions from the given inputs.
        Args:
            cent_obs: (torch.Tensor) observation inputs into network.
        Returns:
            values: (torch.Tensor) value function predictions.
        """

        critic_features = self.base(cent_obs)
        values = self.v_out(critic_features)
        
        return values