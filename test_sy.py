import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging

import matplotlib.pyplot as plt


def structure_loss(pred, mask):
    # (1) real mask preprocessing
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # (2) cross entropy loss between pred and mask
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')

    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model):
    # [1] make test loader
    data_path = 'data_sample'
    image_root = os.path.join(data_path, 'images')
    gt_root = os.path.join(data_path, 'masks')
    #model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)

    # [2] make test loader
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()

        # [1] check gt [res,res]
        gt = np.asarray(gt, np.float32)
        print(f'original gt = {gt.shape}')
        gt /= (gt.max() + 1e-8)
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        target_flat = np.reshape(target, (-1))  # [batch, res*res]

        print(f' target_flat = {target_flat.shape}')

        # [2] image and prdict


        image = image.cuda()
        res, res1 = model(image)
        res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        # Getting Dice Score #
        input_flat = np.reshape(input, (-1))  # [batch, res*res]
        print(f' input_flat = {input_flat.shape}')
        

        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth) # dice = every pixel by pixel

        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice # adding dice score to get mean  DSC
    return DSC / num1




def main(opt):

    print(f' step 1. build models')
    #logging.basicConfig(filename='train_log.log',
    #                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    #                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    model = PolypPVT().cuda()
    dataset_dice = test(model)





if __name__ == '__main__':
    dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    model_name = 'PolypPVT'
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')
    parser.add_argument('--train_save', type=str,
                        default='./model_pth/' + model_name + '/')
    parser.add_argument('--domain', type=str,
                        default='polyp', help='domain adaptation')
    opt = parser.parse_args()
    main(opt)
