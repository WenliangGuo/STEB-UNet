from model import TransUNet
from model import Swin_TransUNet
import matplotlib.pyplot as plt
import numpy as np

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

model = Swin_TransUNet(in_channels=3, out_channels = 1, bilinear=True)
num = params_count(model)
print(num*1.0/1000000)
