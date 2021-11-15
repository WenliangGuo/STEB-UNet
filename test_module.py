from model import PatchEmbed
import torch.nn as nn
import torch

patch_size = 4
in_chans = 3
embed_dim = 96
norm_layer=nn.LayerNorm

x = torch.randn(1,3,256,256)

patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer)
y = patch_embed(x)
print(y.shape)
