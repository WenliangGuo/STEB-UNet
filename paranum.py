from model import TransUNet, Swin_TransUNet
from SETR.transformer_seg import SETRModel 
from setr.SETR import SETR_Naive
import matplotlib.pyplot as plt
import numpy as np
def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

model = SETR_Naive(
    img_dim = 128,
    patch_dim = 2,
    num_channels = 3,
    num_classes = 2,
    embedding_dim = 64,
    num_heads = 8,
    num_layers = 18,
    hidden_dim = 1024,
    dropout_rate=0.1,
    attn_dropout_rate=0.0
    )
num = params_count(model)
print(num*1.0/1000000)
