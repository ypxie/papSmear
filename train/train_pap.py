# -*- coding: utf-8 -*-

import os, sys, pdb
import argparse
import torch

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)

from papSmear.proj_utils.local_utils import mkdirs
from papSmear.datasets.papsmear import papSmearData as Dataset
from papSmear.cfgs.config_pap import cfg
from papSmear.darknet import Darknet19
from papSmear.train_yolo import train_eng

def set_args():
    # Arguments setting
    parser = argparse.ArgumentParser(description = 'Pap Smear Bounding Box Detection')

    parser.add_argument('--device_id',  type=int, default=5, help='which device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size.')
    parser.add_argument('--img_size',   default=[256, 320, 352], help='output image size')

    parser.add_argument('--maxepoch',        type=int,   default=20000,    help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',              type=float, default = 2.0e-4, help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay',        type=float, default = 0.1,    help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay_epochs', default= [6000, 12000],       help='decay the learning rate at this epoch')
    parser.add_argument('--momentum',        type=float, default=0.9,      help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay',    type=float, default=0,        help='weight decay for training')

    parser.add_argument('--reuse_weights',   action='store_true', default=False, help='continue from last checkout point')
    parser.add_argument('--load_from_epoch', type=int, default= 800, help='load from epoch')

    parser.add_argument('--display_freq',    type=int, default= 50, help='plot the results every {} batches')
    parser.add_argument('--save_freq',       type=int, default= 500,  help='how frequent to save the model')
    parser.add_argument('--model_name',      type=str, default='yolo_pap')

    args = parser.parse_args()
    return args


if  __name__ == '__main__':
    args = set_args()
    DatasetDir = "/data/.data1/pingjun/Datasets/PapSmear"
    model_root = os.path.join(DatasetDir, 'models')
    data_root = os.path.join(DatasetDir, 'data/training')

    # Dataloader setting
    dataloader = Dataset(data_root, args.batch_size, img_shape = (256, 256))
    # Set Darknet
    net = Darknet19(cfg)

    # CUDA Settings
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        net.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    # print ('>> START training ')
    train_eng(dataloader, model_root, args.model_name, net, args)
