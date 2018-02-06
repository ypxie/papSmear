# -*- coding: utf-8 -*-

import os, sys, pdb
sys.path.insert(0, '..')

import argparse
import torch
from papSmear.proj_utils.local_utils import mkdirs
from papSmear.datasets.papclassify import papClassify as Dataset
from papSmear.cfgs.config_pap import cfg
from papSmear.models.res_model import papResnet
from papSmear.train_cls import train_cls

proj_root = os.path.join('..')
model_root = os.path.join(proj_root, 'Model')
mkdirs([model_root])

home = os.path.expanduser('~')
dropbox = os.path.join(home, 'Dropbox')

## ----------------training parameters setting--------------------------
data_root = os.path.join(home, 'DataSet','yoloSeg', 'train','papcls')
save_root = os.path.join(data_root,'rectangle')
mkdirs(save_root)
model_name = 'pap_cls'

def set_args():
    # Arguments setting
    parser = argparse.ArgumentParser(description = 'Pap Smear Bounding Box Detection')

    parser.add_argument('--device_id',  type=int, default=0, help='which device')
    parser.add_argument('--batch_size', type=int, default= 32, help='batch size.')
    parser.add_argument('--img_size',   default = [128], help='output image size')

    parser.add_argument('--start_seg',       type=int,   default = 100,    help='number of epochs before we train the seg part')
    parser.add_argument('--maxepoch',        type=int,   default = 20000,    help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',              type=float, default = 2.0e-3, help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay',        type=float, default = 0.1,    help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay_epochs', default= [6000, 12000],       help='decay the learning rate at this epoch')
    parser.add_argument('--momentum',        type=float, default=0.9,      help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay',    type=float, default=0,        help='weight decay for training')

    parser.add_argument('--reuse_weights',   action='store_true', default=True, help='continue from last checkout point')
    parser.add_argument('--load_from_epoch', type=int, default= 2300, help='load from epoch')

    parser.add_argument('--display_freq',    type=int, default= 100, help='plot the results every {} batches')
    parser.add_argument('--save_freq',       type=int, default= 50,  help='how frequent to save the model')
    parser.add_argument('--model_name',      type=str, default=model_name)
    
    args = parser.parse_args()
    return args

if  __name__ == '__main__':
    args = set_args()
    # DatasetDir = "/data/.data1/pingjun/Datasets/PapSmear"

    net = papResnet(num_classes=2)
    print(net)

    dataloader = Dataset(data_root, args.batch_size, img_shape = (128, 128), aug_rate=50)
    #dataloader.overlayImgs(save_root)

    # CUDA Settings
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        net.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    # print ('>> START training ')
    train_cls(dataloader, model_root, args.model_name, net, args)
