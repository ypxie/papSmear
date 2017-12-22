# -*- coding: utf-8 -*-

import os, sys, pdb
import cv2
import torch
import numpy as np
import datetime
from numba import jit

from torch.multiprocessing import Pool
import torch.nn as nn

from .proj_utils.plot_utils import plot_scalar
from .proj_utils.torch_utils import to_device, set_lr
from .proj_utils.local_utils import mkdirs, Indexflow, imshow
from .proj_utils.model_utils import resize_layer

#@jit
def get_pair_mask(gt_boxes, mask_list, featMaps, net, img_list):
    croped_feat_list=[]
    corresponding_mask_list = []

    for img_idx, this_bbox_list in enumerate(gt_boxes):
        
        this_img = img_list[img_idx]

        for bidx, bb in enumerate(this_bbox_list):
            x_min_, y_min_, x_max_, y_max_ = bb
            x_min_, y_min_, x_max_, y_max_ = int(x_min_),int(y_min_), int(x_max_), int(y_max_)
            this_mask      = mask_list[img_idx][bidx]
            try:
                this_featmap   = featMaps[img_idx:img_idx+1,:, y_min_:y_max_+1, x_min_:x_max_+1]
            except:
                print("invalid info:", img_idx,y_min_, y_max_, x_min_, x_max_, featMaps.size() )
            
            #this_patch     = this_img[1, y_min_:y_max_+1, x_min_:x_max_+1]
            #imshow(this_patch)
            #imshow(this_mask)

            dest_size      = list(this_featmap.size()[-2::])
            #print("this_mask shape: ", this_mask.shape)
            dest_size[0]   = this_mask.shape[0]
            dest_size[1]   = this_mask.shape[1]   
            #print("this_featmap shape: ", this_featmap.size() )
            resize_featmap = resize_layer(this_featmap, dest_size)
            
            croped_feat_list.append(resize_featmap)
            corresponding_mask_list.append(this_mask)
    if  len(croped_feat_list) > 0:
        croped_feat_nd = torch.cat(croped_feat_list, 0)
        np_corresponding_mask_nd = np.stack(corresponding_mask_list, 0)
        corresponding_mask_nd = to_device(np_corresponding_mask_nd, net.device_id, requires_grad=False)
        return croped_feat_nd, corresponding_mask_nd
    else:
        return None, None    
def batch_mask_forward(net, feat_nd, batch_size=128 ):
    mask_pred_list = []
    total_num = feat_nd.size()[0]
    for ind in Indexflow(total_num, batch_size, False):
        #th_ind    = to_device(torch.from_numpy(ind).long(), net.device_id, True)
        #print("feat_nd type: ", type(feat_nd))
        #this_feat = feat_nd.index_select( 0, th_ind)
        st, end = np.min(ind), np.max(ind)+1
        this_feat = feat_nd[st:end]
        #this_mask = torch.index_select(mask_nd, 0, th_ind)
        #print(this_feat.size())
        mask_pred = net(this_feat)
        mask_pred_list.append(mask_pred)
    total_pred = torch.cat(mask_pred_list, 0)
    return total_pred

def dice_coef_loss(y_pred, y_true):
    smooth = 1.
    y_pred_f = y_pred.view(-1)
    y_true_f = y_true.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1- (2. * intersection + smooth) / (torch.sum(y_true_f*y_true_f) + torch.sum(y_pred_f*y_pred_f) + smooth)

def train_eng(dataloader, model_root, model_name, net, args):
    net.train()
    lr = args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    loss_train_plot = plot_scalar(name = "loss_train", env= model_name, rate = args.display_freq)
    loss_bbox_plot = plot_scalar(name = "loss_bbox",   env= model_name, rate = args.display_freq)
    loss_iou_plot = plot_scalar(name = "loss_iou",     env= model_name, rate = args.display_freq)
    loss_cls_plot = plot_scalar(name = "loss_cls",     env= model_name, rate = args.display_freq)
    model_folder = os.path.join(model_root, model_name)
    mkdirs([model_folder])

    if args.reuse_weights:                        
        weightspath = os.path.join(model_folder, 'weights_epoch_{}.pth'.format(args.load_from_epoch))
        if os.path.exists(weightspath):
            weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(weightspath))
            #load_partial_state_dict(net, weights_dict)
            net.load_state_dict(weights_dict)
            start_epoch = args.load_from_epoch + 1
        else:
            print('WRANING!!! {} do not exist!!'.format(weightspath))
            start_epoch = 1
    else:
        start_epoch = 1

    thesh = 0.5
    batch_per_epoch = dataloader.batch_per_epoch
    bbox_loss, iou_loss, cls_loss = 0., 0., 0.
    train_loss = 0.0
    cnt = 0
    epoc_num = start_epoch

    print("Start training...")
    for step in range(start_epoch * batch_per_epoch, args.maxepoch * batch_per_epoch):
        if step > 0 and step%10 == 0:
            ridx = np.random.randint(0, len(args.img_size)-1 )
            dataloader.img_shape = [args.img_size[ridx], args.img_size[ridx]]

        batch           = dataloader.next_batch()
        im              = batch['images']
        gt_boxes        = batch['gt_boxes']
        gt_classes      = batch['gt_classes']
        dontcare        = batch['dontcare']
        orgin_im        = batch['origin_im']
        mask_list       = batch['mask_list']
        mask_bbox_list  = batch['mask_bbox_list']
        # forward
        im_data = to_device(im, net.device_id)
        bbox_pred, iou_pred, prob_pred, featMaps = net(im_data, gt_boxes, gt_classes, dontcare)
        loss = net.loss
        #-----------For calculating the segmentation-------------
        #print("featMap type: ", type(featMaps))
        
        cropped_feat, cropped_mask = get_pair_mask(mask_bbox_list, mask_list, featMaps, net, im)
        if cropped_feat is not None:
            total_pred = batch_mask_forward(net.seg_net, cropped_feat, batch_size=128 )
            mask_loss  =  dice_coef_loss(total_pred, cropped_mask)
            loss = loss + mask_loss
        #--------------------------------------------------------
        # backward
        
        bbox_loss_val  = net.bbox_loss.data.cpu().numpy().mean()
        iou_loss_val   = net.iou_loss.data.cpu().numpy().mean()
        cls_loss_val   = net.cls_loss.data.cpu().numpy().mean()
        train_loss_val = loss.data.cpu().numpy().mean()
        mask_loss_val  = mask_loss.data.cpu().numpy().mean()

        bbox_loss  += bbox_loss_val
        iou_loss   += iou_loss_val
        cls_loss   += cls_loss_val
        train_loss += train_loss_val
        loss_train_plot.plot(loss.data.cpu().numpy().mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1

        if step > 0 and step % args.display_freq == 0:
            train_loss /= cnt
            bbox_loss /= cnt
            iou_loss /= cnt
            cls_loss /= cnt
            print(('epoch {:>5}, total loss: {:.6f}, bbox_loss: {:.6f}, \
                    iou_loss: {:.6f}, cls_loss:{:.6f}, mask_loss:{:.6f}'. \
                    format(epoc_num, train_loss, bbox_loss, iou_loss, cls_loss, mask_loss_val)))
            train_loss, bbox_loss, iou_loss, cls_loss = 0, 0., 0., 0.
            cnt = 0

        if step > 0 and step % dataloader.batch_per_epoch == 0:
            if epoc_num in args.lr_decay_epochs: # adjust learing rate
                lr *= args.lr_decay
                optimizer = set_lr(optimizer, lr)
            epoc_num = start_epoch + dataloader.epoch - 1
            if dataloader.epoch > 0 and dataloader.epoch % args.save_freq == 0:
                weights_name = "weights_epoch_" + str(epoc_num) + '.pth'
                model_path = os.path.join(model_folder, weights_name)
                torch.save(net.state_dict(), model_path)
                print('save weights at {}'.format(model_path))

    dataloader.close()
