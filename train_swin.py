import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import os
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import Swin_TransUNet

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import eval
from collections import OrderedDict

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def cal_iou(a, b):
    """
    Returns the IoU of two bounding boxes 
    得到bbox的坐标
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    if torch.cuda.is_available():
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

# ------- 2. set the directory of training dataset --------

model_name = 'Swin_TransUNet' 

train_data = os.path.join(os.getcwd(), '../The cropped image tiles and raster labels/train_all' + os.sep)
tra_image_dir = os.path.join('image' + os.sep)
tra_label_dir = os.path.join('label' + os.sep)

valid_data = os.path.join(os.getcwd(), '../The cropped image tiles and raster labels/val' + os.sep)
val_image_dir = os.path.join('image' + os.sep)
val_label_dir = os.path.join('label' + os.sep)

image_ext = '.png'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epoch_num = 550
batch_size_train = 16
batch_size_val = 1
train_num = 0
val_num = 0

ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_epoch = 5
Loss_list = []

tra_img_name_list = glob.glob(train_data + tra_image_dir + '*' + image_ext)
val_img_name_list = glob.glob(valid_data + val_image_dir + '*' + image_ext)

tra_lbl_name_list = []
val_lbl_name_list = []

for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(train_data + tra_label_dir + imidx + label_ext)

for img_path in val_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	val_lbl_name_list.append(valid_data + val_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("train images: ", len(val_img_name_list))
print("train labels: ", len(val_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

train_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(256),
        ToTensorLab(flag=0)]))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

valid_dataset = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        ToTensorLab(flag=0)]))
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size_val, shuffle=False, num_workers=1)

# ------- 3. define model --------
# define the net
net = Swin_TransUNet(in_channels=3, out_channels = 1)

model_list = os.listdir(model_dir)

if len(model_list) != 0: #load the latest model
    model_list.sort(key=lambda x:os.path.getmtime(os.path.join(model_dir,x)))
    latest_file = model_list[-1]
    print("Previous training is interrupted. Begin training from {}.".format(latest_file))
    # original saved file with DataParallel
    state_dict = torch.load(os.path.join(model_dir,latest_file))
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)

net = nn.DataParallel(net)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(train_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d= net(inputs_v)
        loss = bce_loss(d, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data

        # del temporary outputs and loss
        del d, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f" % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val))


    if (epoch+1) % save_epoch== 0:
        # print("Validating ......")
        # iou_sum = 0
        # ite = 0

        # for i_val, data_val in enumerate(valid_dataloader):
        #     ite = ite + 1
        #     inputs_val = data_val['image']
        #     inputs_val = inputs_val.type(torch.FloatTensor)

        #     if torch.cuda.is_available():
        #         inputs_val = Variable(inputs_val.cuda())
        #     else:
        #         inputs_val = Variable(inputs_val)

        #     d_val= net(inputs_val)

        #     # normalization
        #     pred = d_val[:,0,:,:]
        #     pred = normPRED(pred)

        #     #calculate iou and accumulate
        #     iou_sum = iou_sum + cal_iou(inputs_val, pred)
        #     del d_val
        
        # print("Mean iou: {}".format(iou_sum / ite))
        torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f.pth" % (epoch+1, running_loss / ite_num4val))
        

    Loss_list.append(running_loss / ite_num4val)
    running_loss = 0.0
    running_tar_loss = 0.0
    net.train()  # resume train
    ite_num4val = 0
        
    x = range(0, len(Loss_list))
    y = Loss_list
    plt.plot(x, y, '.-')
    plt.xlabel('Test loss vs. ite_num')
    plt.ylabel('Test loss')
    plt.savefig("loss/Swin_TransUNet_loss.png".format(str(epoch+1)))
