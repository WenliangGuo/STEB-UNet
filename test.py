import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
from SETR.transformer_seg import SETRModel
from collections import OrderedDict
import numpy as np
from PIL import Image
import glob
import warnings
warnings.filterwarnings("ignore")

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import TransUNet, Swin_TransUNet
import copy
import cv2 as cv
import time

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(raw_image_path, pred, save_dir, blend=False):

    predict = pred
    predict = predict.squeeze()
    img_pred = predict.cpu().numpy()
    img_pred = Image.fromarray(img_pred*255).convert('RGB')
    img_name = raw_image_path.split(os.sep)[-1]
    raw_image = cv.imread(raw_image_path)
    #img_up = cv.resize(img_pred, (raw_image.shape[0],raw_image.shape[1]), interpolation= cv.INTER_LINEAR)
    img_up = np.array(img_pred.resize((raw_image.shape[1],raw_image.shape[0]),resample=Image.BILINEAR))
    img_up = img_up[:,:,0]

    if blend == True:
        colors = [(0,0,0),(128,0,0)]
        seg_img = np.zeros((np.shape(raw_image)[0], np.shape(raw_image)[1], 3))
           
        for c in range(2):
            seg_img[:,:,0] += ((img_up[:,: ] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((img_up[:,: ] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((img_up[:,: ] == c )*( colors[c][2] )).astype('uint8')

        img_mark = Image.fromarray(np.uint8(seg_img))
        raw_image = Image.fromarray(np.uint8(raw_image))
        
        img_up = Image.blend(raw_image, img_mark, 0.7)
    
    else:
        img_up = Image.fromarray(np.uint8(img_up))

    img_up.save(os.path.join(save_dir, img_name))

def main():

    # --------- 1. get image path and name ---------
    image_dir = "../The cropped image tiles and raster labels/test/image/"
    prediction_dir = "../results/simple_test/WHU/swin_transu_bce/"
    model_dir = "/home/xiaoxiao/gwl/TransUNet/saved_models/WHU-dataset/Swin_TransUNet_bce/Swin_TransUNet_bce_itr_1780_train_0.047400206327438354.pth"
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print("测试集图像数量：", len(img_name_list))

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(128),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------

    # net = SETRModel(patch_size=(32, 32), 
    #                 in_channels=3, 
    #                 out_channels=1, 
    #                 hidden_size=1024, 
    #                 num_hidden_layers=6, 
    #                 num_attention_heads=8, 
    #                 decode_features=[512, 256, 128, 64])
    
    net = Swin_TransUNet(in_channels=3, out_channels = 1)
    net = nn.DataParallel(net) # multi-GPU

    # state_dict = torch.load(model_dir)
    # # create new OrderedDict that does not contain `module.`
    # checkpoint = OrderedDict()
    # for k, v in state_dict['state_dict'].items():
    #     name = k[7:] # remove `module.`
    #     checkpoint[name] = v
    # net.load_state_dict(checkpoint)

    checkpoint = torch.load(model_dir)
    net.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    tot_time = 0

    with torch.no_grad():
        for i_test, data_test in enumerate(test_salobj_dataloader):

            print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            #old_img = copy.deepcopy(inputs_test)
            since = time.time()
            d= net(inputs_test)
            # normalization
            pred = d[:,0,:,:]
            pred = normPRED(pred)

            tot_time += time.time() - since

            # save results to test_results folder
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            
            save_output(img_name_list[i_test],pred,prediction_dir)
            #save_output(img_name_list[i_test], pred, prediction_dir, blend= True)

            del d,pred,
    print(tot_time)
if __name__ == "__main__":
    main()
