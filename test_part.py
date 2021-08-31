from model.segvit import SegViT
from model.transunet_model import TransUNet
import torch
import os 
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

image_size = 512
patch_size = 2
in_channel = 3
out_channel = 64

net = SegViT(
    image_size = image_size,
    patch_size = patch_size,
    dim = out_channel,
    channels = in_channel,
    depth = 1,
    heads = 16,
    mlp_dim = 2048, 
    dropout = 0.1,
    emb_dropout = 0.1
)

#net = TransUNet(in_channels = 3, out_channels = 1)

img = torch.randn(1, in_channel, image_size, image_size)
preds = net(img) 
net.cuda()

#net = torch.nn.DataParallel(net, device_ids = [0, 1])

shape = list(preds.shape)
preds = preds.view(shape[0],int(math.sqrt(shape[1])),int(math.sqrt(shape[1])),shape[2])
print(preds.shape)