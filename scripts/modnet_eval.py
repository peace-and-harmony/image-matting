#!/usr/bin/python

import math
import cv2, os, sys, random, copy
import shutil
import glob
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from functools import reduce
from skimage import io, transform, color

from argparse import ArgumentParser, SUPPRESS
from tqdm.notebook import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms.functional as TF
from torchvision import transforms

import scipy
from scipy.ndimage import morphology
import matplotlib.pyplot as plt
from PIL import Image

from MODNet.src.models.modnet import MODNet
from IPython import get_ipython

if 'google.colab' in str(get_ipython()):
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-o", "--out", help="Input path -this should be your inference folder", type=str, default=None)
    args.add_argument("-i", "--images", help="Path to tour original images", type=str, default=None)
    args.add_argument("-l", "--labels", help="Path to your binary labels", type=str, default=None)
    args.add_argument("-m", "--model", help="Path to your binary labels", type=str, default=None)
    args.add_argument("-b", "--batch", help="Training batch size", type=int, default=1)

    return parser

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def iou(y_true, y_pred,  smooth=0.001):
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    res = (intersection + smooth) / (union + smooth)
    return res


def save_output(image_name,pred,d_dir, lbl):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()


    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)


    #Added masked and concatenated result
    im = cv2.imread(image_name)
    mask = cv2.cvtColor(pb_np, cv2.COLOR_RGB2GRAY)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)

    image = im.copy()
    im[:,:, 3] = mask

    if lbl != None:
      #grab metrics
      y_true = cv2.imread(lbl)
      res = iou(y_true, pb_np)
    else:
      res=[]

    mask_alp = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
    combined = np.concatenate([im, mask_alp, image], axis=1)
    combined = cv2.cvtColor(combined, cv2.COLOR_BGRA2RGBA)

    fname = os.path.basename(os.path.splitext(image_name)[0])
    cv2.imwrite(f'{d_dir}{fname}.png', combined)

    return res

def main():
    args = build_argparser().parse_args()
    ious = []

    # --------- 1. get image path and name ---------
    model_name='modnet'

    img_name_list = sorted(glob.glob(f'{args.images}/*'))
    lbl_name_list = sorted(glob.glob(f'{args.labels}/*'))

    sample_size = len(img_name_list)
    batch_size=args.batch

    # --------- 2. dataloader ---------
    #1. dataloader
    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # --------- 3. model define ---------

    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet.load_state_dict(torch.load(args.model))#Handle unexpected state dict keys
        modnet.cuda()
    else:
        modnet.load_state_dict(torch.load(args.model, map_location='cpu'))
    modnet.eval()

    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print('Use GPU...')
        modnet = modnet.cuda()
    else:
        print('Use CPU...')
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

    # --------- 4. inference for each image ---------
    # im_names = os.listdir(input_path)
    for i_test, im_name in tqdm(enumerate(img_name_list), total = sample_size):
        print('Process image: {0}'.format(im_name))

        # read image
        try:
          im = Image.open(os.path.join(args.images, im_name))
        except:
          print('UnidentifiedImageError error is hrere')
          continue

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        im_rh, im_rw = (512, 512)

        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda(), False)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        # matte = matte[0][0].data.cpu().numpy()
        matte_name = im_name.split('.')[0] + '.png'

        pred = normPRED(matte)
        # save results to test_results folder
        if not os.path.exists(args.out):
            os.makedirs(args.out, exist_ok=True)

        if args.labels == None:
          save_output(img_name_list[i_test],pred,args.out + '/', args.labels)
        else:
          res = save_output(img_name_list[i_test],pred,args.out + '/', lbl_name_list[i_test])
          ious.append(res)
        del matte


    if ious:
      iou_mean = np.mean(np.array(ious), dtype=np.float64)
      tdp = round(iou_mean, 3)
      print(f'----------\nmiou: {tdp} \n----------')


if __name__ == '__main__':
  main()
