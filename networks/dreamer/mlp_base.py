import torch.nn as nn

"""MLP modules."""

def get_active_func(activation_func):
    """Get the activation function.
    Args:
        activation_func: (str) activation function
    Returns:
        activation function: (torch.nn) activation function
    """
    if activation_func == "sigmoid":
        return nn.Sigmoid()
    elif activation_func == "tanh":
        return nn.Tanh()
    elif activation_func == "relu":
        return nn.ReLU()
    elif activation_func == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_func == "selu":
        return nn.SELU()
    elif activation_func == "hardswish":
        return nn.Hardswish()
    elif activation_func == "identity":
        return nn.Identity()
    else:
        assert False, "activation function not supported!"

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

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func):
        """Initialize the MLP layer.
        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list) list of hidden layer sizes.
            initialization_method: (str) initialization method.
            activation_func: (str) activation function.
        """
        super(MLPLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [
            init_(nn.Linear(input_dim, hidden_sizes[0])),
            active_func,
            nn.LayerNorm(hidden_sizes[0]),
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])),
                active_func,
                nn.LayerNorm(hidden_sizes[i]),
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLPBase(nn.Module):
    """A MLP base module."""

    def __init__(self, obs_dim, hidden_sizes, use_feature_normalization=True, initialization_method="orthogonal_", activation_func="relu"):
        super(MLPBase, self).__init__()

        self.use_feature_normalization = use_feature_normalization
        self.initialization_method = initialization_method
        self.activation_func = activation_func
        self.hidden_sizes = hidden_sizes

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func
        )

    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x