# -*- coding: utf-8 -*-

import os, sys, pdb

HOME_PATH = os.path.expanduser('~')
FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)

import argparse
import torch
from papSmear.proj_utils.local_utils import mkdirs
from papSmear.datasets.papsmear import papSmearData as Dataset
from papSmear.cfgs.config_pap import cfg
from papSmear.darknet import Darknet19
from papSmear.train_yolo import train_eng


if  __name__ == '__main__':
    # Create model path
    model_root = os.path.join(PRJ_PATH, 'Model')
    mkdirs([model_root])
    # Set data_root and save_root
    data_root = os.path.join(HOME_PATH, 'DataSet/PapSmear/CellBoundingBox/TrainDataset')
    save_root = os.path.join(data_root,'rectangle')
    mkdirs(save_root)
    # Setting device_id
    device_id = 0

    # Arguments setting
    parser = argparse.ArgumentParser(description = 'Pap Smear Bounding Box Detection')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for training')
    parser.add_argument('--maxepoch', type=int, default=20000, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default = .0002, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay', type=float, default = 0.1, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay_epochs', default= [6000, 12000], help='decay the learning rate at this epoch')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--reuse_weights', action='store_true', default =  True, help='continue from last checkout point')
    parser.add_argument('--show_progress', action='store_false', default = True, help='show the training process using images')
    parser.add_argument('--save_freq', type=int, default= 20, metavar='N', help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default= 200, metavar='N', help='plot the results every {} batches')
    parser.add_argument('--verbose_per_iter', type=int, default= 50, help='print losses per iteration')
    parser.add_argument('--batch_size', type=int, default=6, metavar='N', help='batch size.')

    ## add more
    parser.add_argument('--device_id', type=int, default= device_id, help='which device')
    parser.add_argument('--img_size', type=int, default= [256, 320, 352], help='output image size')
    parser.add_argument('--epoch_decay', type=float, default=100, help='decay epoch image size')
    parser.add_argument('--load_from_epoch', type=int, default= 800, help='load from epoch')
    parser.add_argument('--model_name', type=str, default='yoloPap')

    # Parse args
    args = parser.parse_args()

    # Dataloader setting
    dataloader = Dataset(data_root, args.batch_size, img_shape = (256, 256))  #img_size is changable
    # dataloader.overlayImgs(save_root)

    # Set Darknet
    net = Darknet19(cfg)
    # print(net)

    # CUDA Settings
    args.cuda = torch.cuda.is_available()
    device_id = getattr(args, 'device_id', 0)
    if args.cuda:
        net.cuda(device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    print ('>> START training ')
    model_name ='{}'.format(args.model_name)
    train_eng(dataloader, model_root, model_name, net, args)
