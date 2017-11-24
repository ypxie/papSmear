# -*- coding: utf-8 -*-

import os, sys, pdb
import cv2
import torch
import numpy as np
import datetime
from torch.multiprocessing import Pool

from .proj_utils.plot_utils import plot_scalar
from .proj_utils.torch_utils import to_device, set_lr
from .proj_utils.local_utils import mkdirs


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
        weightspath = os.path.join(model_folder, 'weights_epoch{}.pth'.format(args.load_from_epoch))
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

        batch = dataloader.next_batch()
        im = batch['images']
        gt_boxes = batch['gt_boxes']
        gt_classes = batch['gt_classes']
        dontcare = batch['dontcare']
        orgin_im = batch['origin_im']
        # forward
        im_data = to_device(im, net.device_id)
        _ = net(im_data, gt_boxes, gt_classes, dontcare)

        # backward
        loss = net.loss
        bbox_loss_val  = net.bbox_loss.data.cpu().numpy().mean()
        iou_loss_val   = net.iou_loss.data.cpu().numpy().mean()
        cls_loss_val   = net.cls_loss.data.cpu().numpy().mean()
        train_loss_val = loss.data.cpu().numpy().mean()

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
            print(('epoch {:>5}, total loss: {:.6f}, bbox_loss: {:.6f}, iou_loss: {:.6f}, cls_loss:{:.2f}'. \
                    format(epoc_num, train_loss, bbox_loss, iou_loss, cls_loss)))
            train_loss, bbox_loss, iou_loss, cls_loss = 0, 0., 0., 0.
            cnt = 0

        if step > 0 and step % dataloader.batch_per_epoch == 0:
            if epoc_num in args.lr_decay_epochs: # adjust learing rate
                lr *= args.lr_decay
                optimizer = set_lr(optimizer, lr)
            epoc_num = start_epoch + dataloader.epoch - 1
            if dataloader.epoch > 0 and dataloader.epoch % args.save_freq == 0:
                weights_name = "weights_epoch-" + str(epoc_num).zfill(5) + '.pth'
                model_path = os.path.join(model_folder, weights_name)
                torch.save(net.state_dict(), model_path)
                print('save weights at {}'.format(model_path))

    dataloader.close()
