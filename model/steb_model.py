from torch.nn import functional as F
from model import PatchEmbed, BasicLayer, PatchMerging
import torch.nn as nn
import torch
from mmcv_custom import load_checkpoint
import math
from .unet_parts import *
from .segvit import Encoder, Transformer
from .swin_transformer import BasicLayer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def position_embedding(x, embed_dim, patch_size, drop):
    H, W = x.size(2), x.size(3)
    img_size = to_2tuple([H, W])
    patch_size = to_2tuple(patch_size)
    patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

    absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
    trunc_normal_(absolute_pos_embed, std=.02)
    absolute_pos_embed = F.interpolate(absolute_pos_embed, size=(H, W), mode='bicubic').cuda()
    x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
    Drop = nn.Dropout(drop)
    return Drop(x)

class STEB_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=2,embed_dim=64, window_size=7,mlp_ratio=4.,
                 qkv_bias=True,qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,bilinear=True):

        super(STEB_UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.embed_dim = embed_dim
        self.patch_size = [patch_size, patch_size]
        self.drop = drop_rate

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer)

        self.transformer1 = BasicLayer(
                dim=64,
                depth=6,
                num_heads=8,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.2,
                norm_layer=norm_layer,
                downsample=None)
        
        self.patch_merge1 = PatchMerging(dim = 64, norm_layer=norm_layer)

        self.transformer2 = BasicLayer(
                dim=64,
                depth=6,
                num_heads=8,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.2,
                norm_layer=norm_layer,
                downsample=None)

        self.patch_merge2 = PatchMerging(dim=128, norm_layer=norm_layer)

        self.transformer3 = BasicLayer(
                dim=128,
                depth=6,
                num_heads=8,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.2,
                norm_layer=norm_layer,
                downsample=None)
        
        self.fuse1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding = 0)

        self.fuse2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding = 0)

        self.fuse3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding = 0)
        
        self.inc = Down(in_channels, 64)
        
        self.down1 = Down(128, 128)
        
        self.down2 = Down(256, 256)
        
        self.down3 = Down(512, 512)

        factor = 2 if bilinear else 1
        
        self.pool2 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        B, C, H, W = x.shape 

        trans_in = self.patch_embed(x) # 64 128 128

        trans_in = position_embedding(trans_in, self.embed_dim, self.patch_size, self.drop)

        trans1_vec = self.transformer1(trans_in, int(H/2), int(W/2))     #128*128 64
 
        trans1 = trans1_vec.permute(0,2,1).view(B, 64, int(H/2), int(W/2))   #64 128 128
        
        trans2_vec =self.transformer2(trans1_vec, int(H/2), int(W/2))        #128*128 64

        trans2_vec_merge = self.patch_merge1(trans2_vec, int(H/2), int(W/2)) #64*64 128

        trans2 = trans2_vec_merge.permute(0,2,1).view(B, 128, int(H/4), int(W/4)) #128 64 64
    
        trans3_vec =self.transformer3(trans2_vec_merge, int(H/4), int(H/4))  # 64*64 128

        trans3_vec_merge = self.patch_merge2(trans3_vec, int(H/4), int(W/4)) #32*32 256

        trans3 = trans3_vec_merge.permute(0,2,1).view(B, 256, int(H/8), int(W/8)) #256 32 32
        
        x1 = self.inc(x)
        
        xx1 = self.fuse1(torch.cat([trans1,x1], dim = 1))

        x2 = self.down1(xx1)

        xx2 = self.fuse2(torch.cat([trans2,x2], dim = 1))
        
        x3 = self.down2(xx2)
        
        xx3 = self.fuse3(torch.cat([trans3,x3], dim = 1))
        
        x4 = self.down3(xx3)
        
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        
        x = self.up2(x, x3)
        
        x = self.up3(x, x2)
    
        x = self.up4(x, x1)
        
        logits = self.outc(x)

        out = F.sigmoid(logits)
        
        return out