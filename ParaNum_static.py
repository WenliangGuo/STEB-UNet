from thop import profile
from thop import clever_format
from model import TransUNet
import torch 
import matplotlib.pyplot as plt
import numpy as np


'''
def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


model = TransUNet(in_channels=3, out_channels = 1, bilinear=True)
num = params_count(model[i])
print(num*1.0/1000000)
'''
model = TransUNet(in_channels=3, out_channels = 1, bilinear=True)
input = torch.randn(1, 3, 512, 512) 
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")