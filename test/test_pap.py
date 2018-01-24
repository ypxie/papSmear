# -*- coding: utf-8 -*-

import os, sys
FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)

from papSmear.darknet import Darknet19
from papSmear.datasets.papsmear import testingData as Dataset
from papSmear.cfgs.config_pap import cfg
from papSmear.test_yolo import test_eng
from papSmear.proj_utils.local_utils import mkdirs

import argparse
import torch

def set_args():
    parser = argparse.ArgumentParser(description = 'Testing code for pap smear detection')
    parser.add_argument('--batch-size',      type=int, default=1)
    parser.add_argument('--device-id',       type=int, default=0)
    parser.add_argument('--resize-ratio',    type=list, default=[1.0])
    parser.add_argument('--load_from_epoch', type=str, default="00050")
    parser.add_argument('--model_name',      type=str, default='yolo_pap_best')
    args = parser.parse_args()

    return args

if  __name__ == '__main__':
    # Setting model, testing data and result path
    # DatasetDir = "/data/.data1/pingjun/Datasets/PapSmear"
    DatasetDir = "/home/pingjun/GitHub/papSmear"
    model_root = os.path.join(DatasetDir, 'models')
    data_root =  os.path.join(DatasetDir, 'data/EDF16')
    save_root =  os.path.join(DatasetDir, 'data/Results')
    mkdirs(save_root)

    # Config arguments
    args = set_args()

    # Dataloader setting
    dataloader = Dataset(data_root, args.batch_size, resize_ratio=args.resize_ratio, test_mode=True)
    # Set Darknet
    net = Darknet19(cfg)

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        net.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    print('>> START testing ')
    model_name ='{}'.format(args.model_name)
    test_eng(dataloader, model_root, save_root, model_name, net, args, cfg)
