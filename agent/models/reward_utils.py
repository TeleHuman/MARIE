import torch

import torch 
import torch.special 
import torch.nn as nn 
import torch.nn.functional as F

class HLGaussLoss(nn.Module):
    '''
    return (..., num_bins) prob labels
    '''
    def __repr__(self):
        return f"HLGaussLoss(min={self.min_value}, max={self.max_value}, bins={self.num_bins}, sigma={self.sigma})"

    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace(
            min_value, max_value, num_bins + 1, dtype=torch.float32
        )

        self.output_dim = self.num_bins
        
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, self.transform_to_probs(target))
    
    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        self.support = self.support.to(target.device)

        cdf_evals = torch.special.erf(
            (self.support - target.unsqueeze(-1))
            / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)
    
    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        self.support = self.support.to(probs.device)
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)
    
class TwoHotCategoricalLoss(nn.Module):
    '''
    return (..., num_bins + 1) prob labels
    '''
    def __repr__(self):
        return f"TwoHotCategoricalLoss(min={self.min_value}, max={self.max_value}, bins={self.num_bins})"

    def __init__(self, min_value: float, max_value: float, num_bins: int, **kwargs):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.support = torch.linspace(
            min_value, max_value, num_bins + 1, dtype=torch.float32
        )

        self.output_dim = self.num_bins + 1
        
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, self.transform_to_probs(target))
    
    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        self.support = self.support.to(target.device)

        shape = target.shape
        target = target.reshape(-1)
        probs = torch.zeros(target.shape + (self.num_bins + 1, ), dtype=torch.float32, device=target.device)
        indices = (self.support < target.unsqueeze(-1)).sum(-1).tolist()
        for i, idx in enumerate(indices):
            probs[i, idx - 1] = torch.abs(self.support[idx] - target[i]) / (self.support[idx] - self.support[idx - 1])
            probs[i, idx] = torch.abs(self.support[idx - 1] - target[i]) / (self.support[idx] - self.support[idx - 1])

        probs = probs.view(*shape, self.num_bins + 1)
        return probs
    
    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        self.support = self.support.to(probs.device)
        return torch.sum(probs * self.support, dim=-1)

losses_dict = {
    'hlgauss': HLGaussLoss,
    'twohot': TwoHotCategoricalLoss,
}