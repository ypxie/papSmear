import os, sys
sys.path.insert(0, '..')

import argparse
import torch

from papSmear.darknet import Darknet19
from papSmear.datasets.papsmear import papSmearData as Dataset
from papSmear.cfgs.config_pap import cfg
from papSmear.train_yolo import train_eng
from papSmear.proj_utils.local_utils import mkdirs

proj_root = os.path.join('..')
model_root = os.path.join(proj_root, 'Model')
mkdirs([model_root])

home = os.path.expanduser('~')
dropbox = os.path.join(home, 'Dropbox')
data_root = os.path.join(home, 'DataSet','papSmear')

save_root = os.path.join(data_root,'rectangle')
mkdirs(save_root)
device_id = 1

if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Gans')    
    parser.add_argument('--weight_decay', type=float, default= 0,
                        help='weight decay for training')
    parser.add_argument('--maxepoch', type=int, default=20000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default = .0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay', type=float, default = 0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay_epochs', default= [6000, 12000], 
                        help='decay the learning rate at this epoch')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--reuse_weights', action='store_true', default = True,
                        help='continue from last checkout point')
    parser.add_argument('--show_progress', action='store_false', default = True,
                        help='show the training process using images')
    
    parser.add_argument('--save_freq', type=int, default= 20, metavar='N',
                        help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default= 200, metavar='N',
                        help='plot the results every {} batches')
    parser.add_argument('--verbose_per_iter', type=int, default= 50, 
                        help='print losses per iteration')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='batch size.')

    ## add more
    parser.add_argument('--device_id', type=int, default= device_id, 
                        help='which device')
    parser.add_argument('--imsize', type=int, default=256, 
                        help='output image size')
    parser.add_argument('--epoch_decay', type=float, default=100, 
                        help='decay epoch image size')
    parser.add_argument('--load_from_epoch', type=int, default= 40, 
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default='yoloPap')
    
    

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    
    dataloader = Dataset(data_root, args.batch_size, dst_size = cfg.inp_size)
    dataloader.overlayImgs(save_root)
    
    # net = Darknet19(cfg)
    
    # print(net)

    # device_id = getattr(args, 'device_id', 0)

    # if args.cuda:
    #     net.cuda(device_id)
    #     import torch.backends.cudnn as cudnn
    #     cudnn.benchmark = True

    # model_name ='{}'.format(args.model_name)
    # print ('>> START training ')
    # train_eng(dataloader, model_root, model_name, net, args)


