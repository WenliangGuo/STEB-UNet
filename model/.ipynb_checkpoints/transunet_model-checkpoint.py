import torch.nn.functional as F
import math
from .transunet_parts import *
from .segvit import SegViT

class TransUNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(TransUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.transformer1 = SegViT(
            image_size = 512,
            patch_size = 2,
            dim = 64,
            depth = 2,
            heads = 4,
            mlp_dim = 128, 
        ) #dim: 512*512*3 -> 128*128*64
        self.sequp1 = SeqUpsample(in_channels = 64, out_channels = 64, scale_factor=2) #dim: 128*128*128-> 256*256*64

        self.transformer2 = SegViT(
            image_size = 256,
            patch_size = 4,
            dim = 128,
            depth = 2,
            heads = 3,
            mlp_dim = 128, 
        ) #dim: 512*512*3 -> 64*64*128
        self.sequp2 = SeqUpsample(in_channels = 128, out_channels = 128, scale_factor=2) #dim: 64*64*256-> 128*128*128

        self.transformer3 = SegViT(
            image_size = 256,
            patch_size = 8,
            dim = 256,
            depth = 2,
            heads = 4,
            mlp_dim = 64, 
        ) #dim: 512*512*3 -> 32*32*256
        self.sequp3 = SeqUpsample(in_channels = 256, out_channels = 256, scale_factor=2) #dim: 32*32*512-> 64*64*256

        self.inc = DoubleConv(in_channels, 64)
        
        self.pool1 = nn.MaxPool2d(2)
        self.down1 = DoubleConv(128, 128)
        
        self.pool2 = nn.MaxPool2d(2) 
        self.down2 = DoubleConv(256, 256)
        
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(512, 512)

        factor = 2 if bilinear else 1
        
        self.pool2 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)
    
    def forward(self, x):
        seq1 =self.transformer1(x)
        f1 = seq1.view(seq1.shape[0],int(math.sqrt(seq1.shape[1])),int(math.sqrt(seq1.shape[1])),seq1.shape[2])
        print("f1生成完毕")
        seq2 =self.transformer2(x)
        f2 = seq1.view(seq2.shape[0],int(math.sqrt(seq2.shape[1])),int(math.sqrt(seq2.shape[1])),seq2.shape[2])
        print("f2生成完毕")
        seq3 =self.transformer3(x)
        f3 = seq1.view(seq3.shape[0],int(math.sqrt(seq3.shape[1])),int(math.sqrt(seq3.shape[1])),seq3.shape[2])
        print("f3生成完毕")

        x1 = self.inc(x)
        x2 = self.down1(torch.cat(f1,x1))
        print("Down1生成完毕")
        x3 = self.down2(torch.cat(f2,x2))
        print("Down2生成完毕")
        x4 = self.down3(torch.cat(f3,x3))
        print("Down3生成完毕")
        x5 = self.down4(x4)
        print("Down4生成完毕")
        x = self.up1(x5, x4)
        print("up1生成完毕")
        x = self.up2(x, x3)
        print("up2生成完毕")
        x = self.up3(x, x2)
        print("up3生成完毕")
        x = self.up4(x, x1)
        print("up4生成完毕")
        logits = self.outc(x)
        print("输出生成完毕")
        return logits