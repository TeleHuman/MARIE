import torch.nn as nn
import torch

from .vector_quantize_pytorch import FSQ, VectorQuantize

import torch
import torch.nn as nn

from einops import rearrange

import ipdb

class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_tokens: int, hidden_size: int = 512, **vq_kwargs):
        super().__init__()

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim * num_tokens)
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * num_tokens, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, in_dim)
        )

        self.codebook = VectorQuantize(dim=embed_dim, **vq_kwargs)
        return
    
    def encode(self, x, should_preprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices, _ = self.codebook(x)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        z_quantized = self.codebook.get_output_from_indices(indices)
        return z_quantized, indices
        

    def decode(self, indices, should_postprocess: bool = False):
        z_quantized = self.codebook.get_output_from_indices(indices)
        rec = self.decoder(z_quantized)

        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec
    
    @torch.no_grad()
    def encode_decode(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        z_q, indices = self.encode(x, should_preprocess)
        rec = self.decode(indices, should_postprocess)
        return rec

    def forward(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices, commit_loss = self.codebook(x)
        
        x = x.reshape(*shape[:-1], -1)
        rec = self.decoder(x)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        
        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec, indices, commit_loss
     
    def preprocess_input(self, x):
        return x
    
    def postprocess_output(self, y):
        '''
        clamp into [-1, 1]
        '''
        # return y.clamp(-1., 1.)
        return y
    
    def compute_loss(self, x, alpha = 10.):
        out, indices, cmt_loss = self(x, True, True)
        rec_loss = (out - x).abs().mean()
        loss = rec_loss + alpha * cmt_loss
        
        active_rate = indices.detach().unique().numel() / self.codebook.codebook_size * 100
        
        loss_dict = {
            "vq/cmt_loss": cmt_loss.item(),
            "vq/rec_loss": rec_loss.item(),
            "vq/active": active_rate,
        }
        
        return loss, loss_dict


class SimpleFSQAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, num_tokens: int, levels, **fsq_kwargs) -> None:
        super().__init__()

        self.num_tokens = num_tokens
        self.levels = levels
        self.embed_dim = len(levels)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, len(levels) * num_tokens)
        )

        self.decoder = nn.Sequential(
            nn.Linear(len(levels) * num_tokens, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, in_dim)
        )

        self.codebook = FSQ(levels, **fsq_kwargs)
        
    def encode(self, x, should_preprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices = self.codebook(x)
        z_quantized = self.codebook.indices_to_codes(indices)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        z_quantized = z_quantized.reshape(*shape[:-1], self.num_tokens, self.embed_dim)
        return z_quantized, indices
        

    def decode(self, indices, should_postprocess: bool = False):
        shape = indices.shape
        indices = rearrange(indices, "... h -> (...) h")

        z_quantized = self.codebook.indices_to_codes(indices)
        z_quantized = rearrange(z_quantized, "... h d -> (...) (h d)")

        rec = self.decoder(z_quantized)

        rec = rec.reshape(*shape[:-1], -1)

        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec
    
    @torch.no_grad()
    def encode_decode(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        z_q, indices = self.encode(x, should_preprocess)
        rec = self.decode(indices, should_postprocess)
        return rec

    def forward(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices = self.codebook(x)
        
        x = x.reshape(*shape[:-1], -1)
        rec = self.decoder(x)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        
        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec, indices


    def preprocess_input(self, x):
        return x
    
    def postprocess_output(self, y):
        # return y.clamp(-1., 1.)
        return y
    
    def compute_loss(self, x):
        out, indices = self(x, True, True)
        loss = (out - x).abs().mean()

        active_rate = indices.detach().unique().numel() / self.codebook.codebook_size * 100
        
        loss_dict = {
            "vq/rec_loss": loss.item(),
            "vq/active": active_rate,
        }
        
        return loss, loss_dict