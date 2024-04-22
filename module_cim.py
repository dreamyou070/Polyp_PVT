from lib.pvt import PolypPVT
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




def main(opt):

    print(f' step 1. build models')
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    model = PolypPVT().cuda()
    weight_dtype = torch.float32
    model.to(dtype=weight_dtype)

    print(f' step 2. cim module')
    # [2] cim module
    backbone_model = model.backbone

    # [3] input
    input = torch.randn(1, 3, 256, 256).to(dtype=weight_dtype, device='cuda')
    vit_output = backbone_model(input)
    #low_level_feature =




if __name__ == '__main__':
    dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ################## model_name #############################
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
    opt = parser.parse_args()
    main(opt)
