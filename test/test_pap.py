# -*- coding: utf-8 -*-

import os, sys

import os, sys, pdb
sys.path.insert(0, '..')

from papSmear.darknet import Darknet19
from papSmear.datasets.papsmear import testingData as Dataset
from papSmear.cfgs.config_pap import cfg
from papSmear.test_yolo import test_eng
from papSmear.proj_utils.local_utils import mkdirs

import argparse
import torch

def set_args():
    parser = argparse.ArgumentParser(description = 'Testing code for pap smear detection')
    parser.add_argument('--batch_size',      type=int, default=1) # has to be 1 for testing.
    parser.add_argument('--device_id',       type=int, default=0)
    parser.add_argument('--resize_ratio',    type=list, default=[0.6])
    parser.add_argument('--load_from_epoch', type=int, default= 280)
    parser.add_argument('--model_name',      type=str, default='yolo_pap_seg')
    args = parser.parse_args()

    return args

if  __name__ == '__main__':
    proj_root = os.path.join('..')
    model_root = os.path.join(proj_root, 'Model')
    mkdirs([model_root])

    home = os.path.expanduser('~')
    data_root = os.path.join(home, 'DataSet','papSmear')

    save_root = os.path.join(data_root,'rectangle')
    mkdirs(save_root)

    # Setting model, testing data and result path
    #DatasetDir = "/data/.data1/pingjun/Datasets/PapSmear"
    #model_root = os.path.join(DatasetDir, 'models')
    #data_root =  os.path.join(DatasetDir, 'data/testing')
    #save_root =  os.path.join(DatasetDir, 'data/Results')
    #mkdirs(save_root)

    # Config arguments
    args = set_args()

    # Dataloader setting
    dataloader = Dataset(data_root, args.batch_size, resize_ratio=args.resize_ratio, test_mode=True)
    # Set Darknet
    net = Darknet19(cfg)

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        net.cuda(args.device_id)
        #import torch.backends.cudnn as cudnn
        #cudnn.benchmark = True

    print('>> START testing ')
    model_name ='{}'.format(args.model_name)
    test_eng(dataloader, model_root, save_root, model_name, net, args, cfg)
