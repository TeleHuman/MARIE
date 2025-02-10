"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

# from dataset import Batch
from .nets import StateEncoder, StateDecoder
from utils import LossWithIntermediateLosses

import ipdb
import wandb

@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor

class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int,
                 encoder: StateEncoder, decoder: StateDecoder) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder

        self.num_tokens = encoder.config.nums_obs_token

        # quantized codebook dim: (vocab_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

    def __repr__(self) -> str:
        return "state_based_tokenizer"
    
    def forward(self, obs: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(obs, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def encode(self, obs: torch.Tensor, should_preprocess: bool = False):
        if should_preprocess:
            obs = self.preprocess_input(obs)
      
        shape = obs.shape # (..., N, Obs_dim), N -> Nums of agents
        obs = obs.reshape(-1, *shape[-1:])
        # z = torch.cat(self.encoder(obs).chunk(self.num_tokens, dim=-1), dim=-2)
        z = rearrange(self.encoder(obs), 'b (n e) -> b n e', n=self.num_tokens, e=self.embed_dim)

        b, n, e = z.shape  # n = tokens per obs
        z_flattened = rearrange(z, 'b n e -> (b n) e')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        tokens = dist_to_embeddings.argmin(dim=-1)
        
        z_q = rearrange(self.embedding(tokens), '(b n) e -> b n e', b=b, n=n, e=e).contiguous()
        tokens = rearrange(tokens, '(b n) -> b n', b=b, n=n).contiguous()

        # Reshape to original
        z = z.reshape(*shape[:-1], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-1], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-1], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)
    
    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False):
        shape = z_q.shape # (..., N, num_tokens, embed_dim)
        z_q = z_q.view(-1, *shape[-2:])
        z_q = rearrange(z_q, 'b n e -> b (n e)')

        rec = self.decoder(z_q)
        # rec = rec.reshape(*shape[:-2], *rec.shape[1:])
        rec = rec.reshape(*shape[:-2], rec.shape[-1])
        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec
    
    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    ###TODO###
    ## SMAC obs in [-1, 1] range
    def preprocess_input(self, x: torch.Tensor):
        return x
    

    def postprocess_output(self, y: torch.Tensor):
        return y
    
    def compute_loss(self, batch, **kwargs: Any) -> LossWithIntermediateLosses:
        '''
        batch:
        - observation: (B, T, N, obs_dim)
        - actions: (B, T, N, act_dim)  (one-hot manner)
        - av_action: (B, T, N, act_dim)
        - reward: (B, T, N, 1)
        - done: (B, T, N, 1)
        - filled: (B, T)
        '''
        assert len(batch['observation'].shape) == 4, "Expected nums of dimensions of batch['obs'] is 4"
        observations = self.preprocess_input(rearrange(batch['observation'], 'b t n o -> (b t) n o'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        # - iris uses 1.0 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()
        
        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        # reconstruction_loss = F.mse_loss(reconstructions, observations)

        rec_error = torch.linalg.norm(reconstructions.detach() - observations.detach(), dim=-1).detach().mean()

        loss = commitment_loss + reconstruction_loss

        loss_dict = {
            'tokenizer/commitment_loss': commitment_loss.item(),
            'tokenizer/reconstruction_loss': reconstruction_loss.item(),
            'tokenizer/rec_error_L2_per_obs': rec_error.item(),
            'tokenizer/total_loss': loss.item(),
        }

        return loss, loss_dict
        # return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss)