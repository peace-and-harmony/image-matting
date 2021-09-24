#!/usr/bin/python

import math
import cv2, os, sys, random, copy
import shutil
import glob
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from functools import reduce
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
from MODNet.src.trainer import supervised_training_iter

# modify trimap-generator output path and pass error trimap
os.chdir('/content/trimap_generator/')
subprocess.call(["sed -i 's#./images/results/#/content/trimap_generator/images/results#g' trimap_module.py"], shell=True)
subprocess.call(["s#sys.exit()#pass#g' trimap_module.py"], shell=True)
# subprocess.call(["sed -i 's#print("ERROR: non-binary image (grayscale)"); pass#pass#g' trimap_module.py"], shell=True)

from trimap_generator.trimap_module import trimap as tr
from trimap_generator.trimap_module import extractImage
from trimap_generator.trimap_module import checkImage



class ImagesDataset(Dataset):
    """
    image (torch.autograd.Variable): input RGB image
                                     its pixel values should be normalized
    trimap (torch.autograd.Variable): trimap used to calculate the losses
                                      its pixel values can be 0, 0.5, or 1
                                      (foreground=1, background=0, unknown=0.5)
    gt_matte (torch.autograd.Variable): ground truth alpha matte
                                      its pixel values are between [0, 1]

    """
    def __init__(self, root, transform=None, w=1024, h=576):
        self.root = root
        self.transform = transform
        self.normalize_colorjit = transforms.Compose([
                                             transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.3),
                                             transforms.RandomInvert(p=0.5),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             ])
        self.w = w
        self.h = h
        self.imgs = sorted(os.listdir(os.path.join(self.root, 'image'))) # '/content/valid_data/image'
        self.alphas = sorted(os.listdir(os.path.join(self.root, 'mask'))) # '/content/valid_data/mask'
        assert len(self.imgs) == len(self.alphas), 'the number of dataset is different, please check it.'

    def getTrimap(self, alpha, idx):
        image   = alpha
        name    = self.alphas[idx]
        size    = 25 # how many pixel extension do you want to dilate
        number  = 1  # numbering purpose
        tr(image, name, size, number, erosion=3)
        trimap = cv2.imread('/content/trimap_generator/images/results/{}px_'.format(size) + name + '_{}.png'.format(number), cv2.IMREAD_UNCHANGED)
        # mapping trimap to [0, 0.5, 1]
        trimap = np.where(trimap==127, 0.5, trimap)
        trimap = np.where(trimap==255, 1, trimap)

        # remove generated png after loading
        os.remove('/content/trimap_generator/images/results/{}px_'.format(size) + name + '_{}.png'.format(number))

        return trimap

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.root, 'image', self.imgs[idx]))
        alpha = cv2.imread(os.path.join(self.root, 'mask', self.alphas[idx]), cv2.IMREAD_GRAYSCALE)

        # random seed for transformers( randomCrop of image should match with
        seed = np.random.randint(2147483647)   # corresponding gt_matte
        random.seed(seed)                      # and generated trimap
        torch.manual_seed(seed)
        if self.transform is not None:         # random seed apply to img
          img = self.transform(img)
          img = self.normalize_colorjit(img)   # normaliza original image

        random.seed(seed)
        torch.manual_seed(seed)
        if self.transform is not None:         # random seed apply to alpha
          alpha = self.transform(alpha)
          alpha_tri = alpha.mul(255).byte()
          alpha_tri = alpha_tri.cpu().numpy().transpose((1, 2, 0))
          alpha_tri = np.squeeze(alpha_tri, axis=2)
          trimap = self.getTrimap(alpha_tri, idx)

          trimap = np.expand_dims(trimap, axis=0)

        return self.imgs[idx], img, trimap, alpha

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-p", "--path", help="Input path -this should be the folder storing checkpoint.pth", type=str, default=None)
    args.add_argument("-ts", "--trainpath", help="Path to tour original images", type=str, default=None)
    args.add_argument("-s", "--efreq", help="Epoch frequency with wich to save the model stat", type=int, default=1)
    args.add_argument("-e", "--enum", help="Number of overall epochs", type=int, default=80)
    args.add_argument("-b", "--batch", help="Training batch size", type=int, default=16)
    args.add_argument("-w", "--nworkers", help="Number of workers", type=int, default=0)
    args.add_argument("-l", "--learningrate", help="Define learning rate", type=float, default=0.01)
    args.add_argument("--resume",help='return training from the last model and optimizer state', action="store_true")

    return parser




def main():
    """ resume=True if not first runing else False  """

    args = build_argparser().parse_args()

    torch_transforms = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Resize(544),
         transforms.RandomCrop(512),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.RandomRotation(degrees=(0, 180)),
        #  transforms.RandomInvert() # should implemented separately only for original image of training dataset
         # move to self.normalize_colorjit
         ]
         )
    # initialize para
    ite_num = 0
    epoch_start = 0
    batch_num = 0
    # dataloader

    # os.mkdir(args.path)
    log_dir = os.path.join(args.path, 'logs')
    try:
     os.makedirs(log_dir)
    except OSError:
      pass
    writer = SummaryWriter(log_dir=log_dir)

    save_model_dir = args.path
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    dataset = ImagesDataset(args.trainpath, torch_transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.nworkers, pin_memory=True)

    # initial optimizer
    optimizer = torch.optim.SGD(modnet.parameters(), lr=args.learningrate, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.125 * args.enum),
                                                   gamma=0.1)  # step_size lr reduce freq， default: reduce/10 iters

    if args.resume:
      VModel = sorted(os.listdir(save_model_dir))[-1]  # default [-1]. [12] takes 000013.ckpt
      check = os.path.join(save_model_dir, VModel)
      print(f'Loading model--{check}')

      state = torch.load(check) # saved for rhys'
      modnet.load_state_dict(state['state_dict'])# saved for rhys'

      print("Loading optimizer and lr_scheduler from saved model...")
      optimizer.load_state_dict(state['optimizer'])
      # lr_scheduler.load_state_dict(state['lr_scheduler'])
      lr_scheduler.load_state_dict(state['lr_scheduler'])

      #load general parameters
      ite_num = state['ite_num']
      epoch_start = state['epoch']

    else:
      try:
        check = '/content/MODNet/pretrained/modnet_webcam_portrait_matting.ckpt'
        modnet.load_state_dict(torch.load(check))
      except:
        print('Train from Scrach! Good Luck')
        pass

    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
      print('Use GPU...')
      modnet = modnet.cuda()

    for epoch in range(epoch_start+1, args.enum+1):
        modnet.train()
        batch_num = 0

        # refresh loss list every epoch
        mattes = []
        semantics = []
        details = []

        for idx, (img_file, image, trimap, gt_matte) in enumerate(dataloader, start=1):
            ite_num = ite_num + 1
            batch_num = batch_num + 1

            trimap = trimap.float().cuda()
            image = image.cuda()
            gt_matte = gt_matte.cuda()

            semantic_loss, detail_loss, matte_loss = \
                supervised_training_iter(modnet, optimizer, image, trimap, gt_matte)
            info = f"Batch_num: {batch_num} epoch: {epoch}/{args.enum} semantic_loss: {semantic_loss}, detail_loss: {detail_loss}, matte_loss： {matte_loss}"
            print(info)
            mattes.append(float(matte_loss))
            semantics.append(float(semantic_loss))
            details.append(float(detail_loss))
            # tensorboard writer
            writer.add_scalar('semantic_loss_idx', semantic_loss, idx)
            writer.add_scalar('detail_loss_idx', detail_loss, idx)
            writer.add_scalar('matte_loss_idx', matte_loss, idx)

        avg_matte = float(np.mean(mattes))
        avg_semantic = float(np.mean(semantics))
        avg_detail = float(np.mean(details))

        print(f"epoch: {epoch}/{args.enum}, average_matte_loss: {avg_matte}")
        lr_scheduler.step()

        writer.add_scalar('ave_matte', avg_matte, epoch)
        writer.add_scalar('avg_semantic', avg_semantic, epoch)
        writer.add_scalar('avg_detail', avg_detail, epoch)

        if epoch % args.efreq == 0:

            state = {
                'epoch': epoch,
                'ite_num':ite_num,
                'state_dict': modnet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'last_lr': optimizer.param_groups[0]['lr'],
                'lr_scheduler': lr_scheduler.state_dict(),
                }

            torch.save(state, save_model_dir + '/' + "_bcev2_itr_%d_tar_%3f.pth" % (ite_num, epoch))
            modnet.train()  # resume train
        writer.flush()

if __name__ == '__main__':
    main()
