# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""Auxiliary dense diffuse-albedo and photometric-reliability predictions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseAuxiliaryHead(nn.Module):
    """Predict bounded albedo and reliability from the final decoder tokens."""

    def __init__(self, net, rho_max=1.0):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.rho_max = float(rho_max)
        self.proj = nn.Linear(net.dec_embed_dim, 4 * self.patch_size ** 2)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, decout, img_shape):
        height, width = img_shape
        tokens = decout[-1]
        batch = tokens.shape[0]
        patch = self.proj(tokens).transpose(-1, -2).reshape(
            batch, 4 * self.patch_size ** 2,
            height // self.patch_size, width // self.patch_size)
        dense = F.pixel_shuffle(patch, self.patch_size).permute(0, 2, 3, 1)
        albedo = self.rho_max * torch.sigmoid(dense[..., :3])
        reliability = torch.sigmoid(dense[..., 3])
        return {'albedo': albedo, 'photometric_reliability': reliability}
