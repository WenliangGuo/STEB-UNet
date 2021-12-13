from torch.nn import functional as F
import math
from .transunet_parts import *
from .segvit import Encoder, Transformer

class TransUNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(TransUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.encoder = Encoder(
            dim = 64,
            image_size = 128,
            patch_size = 2,
            channels = 3
        )
        self.transformer1 = Transformer(
            dim = 64,
            depth = 6,
            heads = 8,
            mlp_dim = 256, 
            dim_head = 64
        ) 

        self.transformer2 = Transformer(
            dim = 64,
            depth = 6,
            heads = 8,
            mlp_dim = 512, 
            dim_head = 64
        )
        self.seqdown1 = Down(in_channels= 64, out_channels= 128)

        self.transformer3 = Transformer(
            dim = 128,
            depth = 6,
            heads = 8,
            mlp_dim = 1024, 
            dim_head = 64
        ) 
        self.seqdown2 = Down(in_channels= 128, out_channels= 256)

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

        seq0 = self.encoder(x)
        # print("f_in: {}".format(x.shape))

        seq1 =self.transformer1(seq0)
    
        f1 = seq1.view(seq1.shape[0], seq1.shape[2], int(math.sqrt(seq1.shape[1])),int(math.sqrt(seq1.shape[1])))
        #print("f1: {}".format(f1.shape))
        
        seq2 =self.transformer2(seq1)
        f2 = seq2.view(seq2.shape[0], seq2.shape[2], int(math.sqrt(seq2.shape[1])),int(math.sqrt(seq2.shape[1])))
        #print("f2: {}".format(f2.shape))
        
        f3 = self.seqdown1(f2)
        seq3 =self.transformer3(f3.view(f3.shape[0],f3.shape[2]**2,f3.shape[1]))
        f4 = seq3.view(seq3.shape[0], seq3.shape[2], int(math.sqrt(seq3.shape[1])),int(math.sqrt(seq3.shape[1]))) 
        #print("f3: {}".format(f3.shape))

        f5 = self.seqdown2(f4)
        
        x1 = self.inc(x)
        #print("x1: {}".format(x1.shape))

        xx1 = self.fuse1(torch.cat([f1,x1], dim = 1))

        x2 = self.down1(xx1)
       # print("x2: {}".format(x2.shape))
        
        xx2 = self.fuse2(torch.cat([f3,x2], dim = 1))

        x3 = self.down2(xx2)
       # print("x3: {}".format(x3.shape))

        xx3 = self.fuse3(torch.cat([f5,x3], dim = 1))
        
        x4 = self.down3(xx3)
        #print("x4: {}".format(x4.shape))
        
        x5 = self.down4(x4)
        #print("x5: {}".format(x5.shape))
        
        x = self.up1(x5, x4)
        
        x = self.up2(x, x3)
        
        x = self.up3(x, x2)
    
        x = self.up4(x, x1)
        
        logits = self.outc(x)

        out = F.sigmoid(logits)
        
        return out