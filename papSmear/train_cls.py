import os
import cv2
import torch
import numpy as np
import datetime, time
from random import shuffle
from torch.multiprocessing import Pool

from .proj_utils.plot_utils import plot_scalar, plot_img, save_images
from .proj_utils.torch_utils import *
from .proj_utils.local_utils import *
from .proj_utils.model_utils import resize_layer
from .proj_utils.data_augmentor import ImageDataGenerator

#@jit(nopython=True)
def make_contour_mask(images, contour_list):
    row, col = images.shape[0:2]
    mask = np.zeros((row, col))
    for idx, this_contour in enumerate(contour_list):
        #print("{}_th_contour shape: ".format(idx), this_contour.shape)
        this_contour = this_contour.astype(int)
        col, row = this_contour[:,0],this_contour[:,1]
        mask[row,col] = 1
    return mask

def mark_contours(images, contour_list):        
    #images = np.transpose(images, (1,2,0)).copy()
    mask = make_contour_mask(images, contour_list)
    
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    contour_mask = cv2.dilate(mask, se)
    
    contour_mask = contour_mask > 0
    
    images[contour_mask,0] = 165
    images[contour_mask,1] = 42
    images[contour_mask,2] = 42
    return images
    
def train_cls(dataloader, model_root, mode_name, net, args):
    net.train()
    lr = args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    loss_train_plot   = plot_scalar(name = "loss_cls",     env= mode_name, rate = args.display_freq)
    model_folder    = os.path.join(model_root, mode_name)
    mkdirs([model_folder])

    if args.reuse_weights :
        weightspath = os.path.join(model_folder, 'weights_epoch_{}.pth'.format(args.load_from_epoch))
        if os.path.exists(weightspath):
            weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(weightspath))
            #load_partial_state_dict(net, weights_dict)
            net.load_state_dict(weights_dict)
            start_epoch = args.load_from_epoch + 1
            #if os.path.exists(plot_save_path):
            #    plot_dict = torch.load(plot_save_path)
        else:
            print('WRANING!!! {} do not exist!!'.format(weightspath))
            start_epoch = 1
    else:
        start_epoch = 1
    
    batch_per_epoch = dataloader.batch_per_epoch
    train_loss = 0
    #bbox_loss, iou_loss, cls_loss = 0., 0., 0.
    cnt = 0
    step_cnt = 0
    epoc_num = start_epoch
    for step in range(start_epoch * batch_per_epoch, args.maxepoch * batch_per_epoch):
        if step >= 1 and step%10 == 0:
            ridx = random.randint(0, len(args.img_size)-1 )
            dataloader.img_shape = [args.img_size[ridx], args.img_size[ridx]]
        start_timer = time.time()
        
        get_mask = epoc_num >= args.start_seg
        
        batch           = dataloader.next_batch(get_mask)
        im              = batch['images']
        #gt_boxes        = batch['gt_boxes']
        gt_classes      = batch['gt_classes']
        res_mat         = batch['dontcare']
        orgin_im        = batch['origin_im']
        
        #print('this im  shape: ', im.shape)
        # forward
        im_data  = to_device(im, net.device_id)
        im_label = to_device(gt_classes, net.device_id, requires_grad=False).long()

        uniques_, counts = np.unique(gt_classes, return_counts=True)
        if len(uniques_) > 1:
            weights = np.zeros((net.num_classes), dtype=np.float32) + 1e-8
            #import pdb; pdb.set_trace()
            for idx, uniq in enumerate(uniques_):
                weights[uniq] = counts[idx]
            
            weights =  1.0/weights
            weights =  weights/np.sum(weights)
            cls_wgt =  to_device(weights, net.device_id, requires_grad=False)
            #print(uniques_, counts, weights)
            # for iidx, (this_img, bbox) in enumerate(zip(im_data, gt_boxes)):
            #     #print(bbox) 
            #     this_mask = mask_list[iidx]
            #     #imshow(this_mask[0])
            #     im_bbox = dataloader.overlay_bbox(this_img.cpu().data.numpy().transpose(1,2,0), bbox)
            #     imshow(im_bbox)
            #_ = net(im_data, gt_boxes, gt_classes, dontcare)

            probs = net(im_data)
            
            loss  = F.nll_loss(F.log_softmax(probs), im_label, weight=cls_wgt)
            num_sample  = im_data.size()[0]

            #--------------------------------------------------------
            # backward
            train_loss_val = loss.data.cpu().numpy().mean()

            train_loss  += train_loss_val


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
            step_cnt += 1
            
            if step % args.display_freq == 0:
                train_loss /= cnt
                
                end_timer = time.time()
                totoal_time = end_timer-start_timer
                print((' epoch {}[{}/{}], loss: {}, , #sample_{},  time:{:.2f}'. \
                        format(epoc_num, step_cnt, batch_per_epoch, train_loss, num_sample, totoal_time)))

                # print((' epoch {}[{}/{}], loss: {}, bbox_loss: {}, iou_loss: {}, cls_loss:{}, \
                #         mask_loss_val:{}, #sample_{}, #seg_sample_{}, time:{:.2f}'. \
                #         format(epoc_num, step_cnt, batch_per_epoch, train_loss, bbox_loss, 
                #                iou_loss, cls_loss, mask_loss_val, num_sample, num_seg, totoal_time)))

                start_timer = time.time()
                print(orgin_im[0].shape, im.shape)
                plot_img(X=im[0][0], win='cropped_img', env=mode_name)
                org_img_contours = mark_contours(orgin_im[0][:,:], res_mat[0])
                plot_img(X=org_img_contours, win='original image', env=mode_name)
                train_loss = 0
                bbox_loss, iou_loss, cls_loss = 0., 0., 0.
                cnt = 0
            
            loss_train_plot.plot(loss.data.cpu().numpy().mean())   

            if step > 0 and (step % dataloader.batch_per_epoch == 0):
                if epoc_num in args.lr_decay_epochs:
                    lr *= args.lr_decay
                    optimizer = set_lr(optimizer, lr)
                step_cnt = 0    

                epoc_num = start_epoch + dataloader.epoch - 1
                if dataloader.epoch>0 and dataloader.epoch % args.save_freq == 0:
                    torch.save(net.state_dict(), os.path.join(model_folder, 'weights_epoch_{}.pth'.format(epoc_num)))
                    print('save weights at {}'.format(model_folder))
                
    dataloader.close()
