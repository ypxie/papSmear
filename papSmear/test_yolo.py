# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np
import pickle
import torch
from torch.multiprocessing import Pool

import deepdish as dd
from .utils.timer import Timer
from .utils import yolo as yolo_utils
from .utils.cython_yolo import yolo_to_bbox
from .proj_utils.plot_utils  import plot_scalar
from .proj_utils.torch_utils import to_device
from .proj_utils.local_utils import writeImg, mkdirs, Indexflow, measure, imshow
from .proj_utils.model_utils import resize_layer
from torch.autograd import Variable

thresh = 0.5
def batch_forward(cls, BatchData, batch_size, **kwards):
    total_num = BatchData.shape[0]
    results = {'bbox':[],'iou':[], 'prob':[]}

    for ind in Indexflow(total_num, batch_size, False):
        data = BatchData[ind]
        
        data = to_device(data, cls.device_id, volatile=True)
        bbox_pred, iou_pred, prob_pred = cls.forward(data, **kwards)
        #print('data shape: ',bbox_pred.size(), iou_pred.size(), prob_pred.size())
        results['bbox'].append(bbox_pred.cpu().data.numpy())
        results['iou'].append(iou_pred.cpu().data.numpy())
        results['prob'].append(prob_pred.cpu().data.numpy())

    for k, v in results.items():
        results[k] = np.concatenate(v, 0)
    return results


def split_testing(cls, img,  batch_size = 4, windowsize=None, thresh= None, cfg=None):

    # since inputs is (B, T, C, Row, Col), we need to make (B*T*C, Row, Col)
    #windowsize = self.row_size
    board = 0
    adptive_batch_size=False # cause we dont need it for fixed windowsize.

    results = {'bbox':[],'iou':[], 'prob':[]}
    C, row_size, col_size = img.shape
    if windowsize is None:
        row_window = row_size
        col_window = col_size
    else:
        row_window = min(windowsize, row_size)
        col_window = min(windowsize, col_size)

    # print(row_window, col_window)
    num_row, num_col = math.ceil(row_size/row_window),  math.ceil(col_size/col_window) # lower int
    feat_map = None
    for row_idx in range(num_row):
        row_start = row_idx * row_window
        if row_start + row_window > row_size:
            row_start = row_size - row_window
        row_end   = row_start + row_window

        for col_idx in range(num_col):
            col_start = col_idx * col_window
            if col_start + col_window > col_size:
                col_start = col_size - col_window
            col_end   = col_start + col_window

            this_patch = img[:, row_start:row_end, col_start:col_end][None]

            #print('this_patch shape: ', this_patch.shape)
            batch_data = to_device(this_patch, cls.device_id, volatile=True)
            bbox_pred, iou_pred, prob_pred, large_map  = cls.forward(batch_data)

            bbox_pred = bbox_pred.cpu().data.numpy()
            iou_pred = iou_pred.cpu().data.numpy()
            prob_pred = prob_pred.cpu().data.numpy()
            large_map = large_map

            
            if feat_map is None:
                chnn = large_map.size()[1]
                feat_map = Variable(torch.zeros(1, chnn, row_size, col_size), volatile=True)

            feat_map[:,:, row_start:row_end, col_start:col_end] = large_map

            H, W = cls.out_size
            x_ratio, y_ratio = cls.x_ratio, cls.y_ratio

            bbox_pred = yolo_to_bbox(
                        np.ascontiguousarray(bbox_pred, dtype=np.float),
                        np.ascontiguousarray(cfg.anchors, dtype=np.float),
                        H, W,
                        x_ratio, y_ratio)

            np_start  = np.array([[col_start, row_start, col_start, row_start]]  )

            results['bbox'].append(bbox_pred +  np_start )
            results['iou'].append(iou_pred)
            results['prob'].append(prob_pred)
    
    results['feat_map'] = feat_map
    results['bbox'] = np.concatenate(results['bbox'], 1)
    results['iou']  = np.concatenate(results['iou'], 1)
    results['prob'] = np.concatenate(results['prob'], 1)
    
    return results

def get_feat_bbox(pred_boxes, featMaps, dest_size=[32,32], org_img=None):
    croped_feat_list=[]
    org_size, org_coord, patch_list = [], [], []
    for img_idx, this_bbox_list in enumerate(pred_boxes):
        for bidx, bb in enumerate(this_bbox_list):
            x_min_, y_min_, x_max_, y_max_ = bb
            x_min_, y_min_, x_max_, y_max_ = int(x_min_),int( y_min_), int(x_max_), int(y_max_)
            row_size = x_max_ - x_min_ + 1
            col_size = y_max_ - y_min_ + 1
            this_featmap   = featMaps[img_idx:img_idx+1,:, y_min_:y_max_+1, x_min_:x_max_+1]
            this_patch     = org_img[:, y_min_:y_max_+1, x_min_:x_max_+1]
            
            resize_featmap = resize_layer(this_featmap, dest_size)
            
            croped_feat_list.append(resize_featmap)
            org_size.append([row_size, col_size])
            org_coord.append([y_min_, x_min_])
            patch_list.append(this_patch)

    croped_feat_nd = torch.cat(croped_feat_list, 0)
    
    return croped_feat_nd, org_size, org_coord, patch_list

def batch_mask_forward(net, feat_nd, batch_size=128 ):
    mask_pred_list = []
    total_num = feat_nd.size()[0]
    for ind in Indexflow(total_num, batch_size, False):
        st, end = np.min(ind), np.max(ind)+1
        this_feat = feat_nd[st:end]
        mask_pred = net(this_feat)
        mask_pred_list.append(mask_pred)
    total_pred = torch.cat(mask_pred_list, 0)
    return total_pred

def mask2contour(mask_pred, org_size_list, org_coord_list, patch_list):
    total_num = mask_pred.shape[0]
    contour_list = []
    
    for idx in range(total_num):
        this_mask = mask_pred[idx][0] # row x col
        row, col  = this_mask.shape
        mask_boarder = np.zeros((row+2, col+2)).astype(np.int)
        this_mask = this_mask.astype(int)
        mask_boarder[1:-1, 1:-1] = this_mask
        
        contours = measure.find_contours(mask_boarder, 0)[0] # list of N*2
        contours = contours - np.array([1,1])

        row_ratio = float(org_size_list[idx][0])/ row
        col_ratio = float(org_size_list[idx][1])/ col
        contours[:,0] *= row_ratio
        contours[:,1] *= col_ratio

        rs, cs = org_coord_list[idx]
        #import pdb; pdb.set_trace()
        #imshow(patch_list[idx][0])
        #imshow(mask_boarder)
        thiscontour = (contours + np.array([rs, cs])).astype(int)
        contour_list.append(thiscontour)
    return contour_list

def mark_contours(images, contour_list):        
    for idx, this_contour in enumerate(contour_list):
        print("contour shape: ", this_contour.shape)
        this_contour = this_contour.astype(int)
        row, col = this_contour[:,0],this_contour[:,1]
        images[row,col,:] = 0
        #for idx in range(this_contour.shape[0]):
        #    x, y = this_contour[idx,0],this_contour[idx,1]
        #    images[x,y,:] = 0
    return images
    
def test_eng(dataloader, model_root, save_root, mode_name, net, args, cfg):
    save_folder = os.path.join(save_root, mode_name)
    save_masked_folder = os.path.join(save_root, mode_name, 'masked_contour')
    mkdirs([save_folder, save_masked_folder])

    net.eval()
    model_folder = os.path.join(model_root, mode_name)
    weightspath = os.path.join(model_folder, 'weights_epoch_{}.pth'.format(args.load_from_epoch))
    weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    print('reload weights from {}'.format(weightspath))
    net.load_state_dict(weights_dict)

    train_loss = 0
    bbox_loss, iou_loss, cls_loss = 0., 0., 0.
    num_images = dataloader.num_images # cause testing has to be of batch size 1
    # all_boxes = [[[] for _ in range(num_images)]
    #              for _ in range(1)]
    print('num_images: ', num_images)

    _t = {'im_detect': Timer(), 'misc': Timer()}
    for i in range(num_images): # batch size has to be 1.
        batch      = dataloader.next()
        ori_im     = batch['origin_im'][0]
        img_name   = batch['dontcare'][0]
        im_np      = batch['images'][0]
        windowsize = 512 # it has to be dividable by 32 to have no shifts.

        _t['im_detect'].tic()
        result_dict = split_testing(net, im_np,  batch_size=4, windowsize=windowsize, thresh=thresh, cfg=cfg)
        bbox_pred, iou_pred, prob_pred  = result_dict['bbox'], result_dict['iou'], result_dict['prob']
        feat_map   = result_dict["feat_map"]

        detect_time = _t['im_detect'].toc()
        _t['misc'].tic()

        bboxes, scores, cls_inds = yolo_utils.postprocess_bbox(bbox_pred, iou_pred, prob_pred,im_np.shape[1::], cfg, thresh)
        utils_time = _t['misc'].toc()

        # from bboxes to feat and to mask
        print(bboxes.shape)
        # bbox N*4, (xs, ys, xe, ye)
        cropped_feat, org_size_list, org_coord_list ,patch_list= get_feat_bbox(bboxes[None], feat_map, dest_size=[32,32], org_img= im_np)
        mask_pred    = batch_mask_forward(net.seg_net, cropped_feat, batch_size=128 )
        mask_pred    = mask_pred.data.cpu().numpy()
        binary_mask  = mask_pred > 0.4
        contour_list = mask2contour(binary_mask, org_size_list, org_coord_list, patch_list)

        
        naked_name = os.path.splitext(img_name)[0]
        resultsDict = {'bbox':bboxes, 'contour':contour_list}
        resultDictPath = os.path.join(save_folder, naked_name + '_res.h5')

        
        dd.io.save(resultDictPath, resultsDict, compression=None)    #compression='zlib'        

        print('{}/{} detection time {:.4f}, post_processing time {:.4f}'.format(i+1, num_images, detect_time, utils_time))
        ori_im = ori_im.transpose(1,2,0)

        bboxes = bboxes / args.resize_ratio[0]
        overlaid_img = dataloader.overlay_bbox(ori_im.copy(), bboxes, linewidth=6).astype(np.uint8)
        writeImg(overlaid_img, os.path.join(save_folder, img_name))

        # mark the contours on the images
        marked_images = mark_contours(ori_im.copy(), contour_list).astype(np.uint8)
        writeImg(marked_images, os.path.join(save_masked_folder, img_name))

    # bbox_file = os.path.join(save_folder, 'bbox_file.pkl')
    # with open(bbox_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

