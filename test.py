import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import TransUNet, Swin_TransUNet

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    image_dir = "../Massachusetts-dataset/test/image/"
    #image_dir = "../The cropped image tiles and raster labels/test/image/"
    prediction_dir = "../results/transfer_test/Mass/setr/"
    model_dir = "../TransUNet/saved_models/WSU-dataset/SETR/SETR_bce_itr_300_train_0.089532.pth"
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

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

    net = SETRModel(patch_size=(32, 32), 
                    in_channels=3, 
                    out_channels=1, 
                    hidden_size=1024, 
                    num_hidden_layers=6, 
                    num_attention_heads=8, 
                    decode_features=[512, 256, 128, 64])
    net = nn.DataParallel(net) # multi-GPU

    checkpoint = torch.load(model_dir)
    net.load_state_dict(checkpoint['state_dict'])
    
    #net.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    
    with torch.no_grad():
        for i_test, data_test in enumerate(test_salobj_dataloader):

            print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d= net(inputs_test)

            # normalization
            pred = d[:,0,:,:]
            pred = normPRED(pred)

            # save results to test_results folder
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            save_output(img_name_list[i_test],pred,prediction_dir)

            del d

if __name__ == "__main__":
    main()
