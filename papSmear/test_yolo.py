import os
import cv2
import pickle
import torch
import numpy as np
import datetime
from torch.multiprocessing import Pool
from .utils.timer import Timer
from .proj_utils.plot_utils  import plot_scalar
from .proj_utils.torch_utils import *
from .proj_utils.local_utils import *

from .utils import yolo as yolo_utils
from .utils.cython_yolo import yolo_to_bbox

max_per_image = 10
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


def split_testing__notused(cls, img,  batch_size = 4, windowsize=None, anchors=None,  
                  W=None, H=None, x_ratio=None, y_ratio=None):

    # since inputs is (B, T, C, Row, Col), we need to make (B*T*C, Row, Col)
    #windowsize = self.row_size
    board = 0
    adptive_batch_size=False # cause we dont need it for fixed windowsize.
    
    results = {'bbox':[],'iou':[], 'prob':[]}

    PatchDict = split_img(img, windowsize = windowsize, board = board, fixed_window = True,step_size=None, tuple_slice=True)
    output = None
    all_keys = PatchDict.keys()
    for this_size in all_keys:
        BatchData, org_slice_list, extract_slice_list, slice_tuple_list = PatchDict[this_size]
        if adptive_batch_size == True:
            old_volume = batch_size * windowsize * windowsize
            new_bs = int(np.floor( 1.0*old_volume/np.prod(this_size)))
        else:
            new_bs = batch_size
        thisprediction = batch_forward(cls, BatchData, batch_size)

        bbox_pred, iou_pred, prob_pred = thisprediction['bbox'], thisprediction['iou'], thisprediction['prob']
        H, W = cls.out_size
        x_ratio, y_ratio = cls.x_ratio, cls.y_ratio
        print(H, W, x_ratio, y_ratio, BatchData.shape)
        
        bbox_pred = yolo_to_bbox(
                    np.ascontiguousarray(bbox_pred, dtype=np.float),
                    np.ascontiguousarray(anchors, dtype=np.float),
                    H, W,
                    x_ratio, y_ratio)

        for idx_, _ in enumerate(slice_tuple_list):
            
            org_slice = slice_tuple_list[idx_]
            row_start = org_slice[0]
            col_start = org_slice[1]
            np_start  = np.array([[col_start, row_start, col_start, row_start]]  )
            print('np start: ',np_start)
            this_bbox = bbox_pred[idx_:idx_+1]
            this_iou  = iou_pred[idx_:idx_+1]
            this_prob = prob_pred[idx_:idx_+1]
            
            #bboxes, scores, cls_inds = yolo_utils.postprocess_bbox(bbox_pred, iou_pred, prob_pred, im_np.shape[1::], cfg, thresh)
            
            results['bbox'].append(this_bbox +  np_start )
            results['iou'].append(this_iou)
            results['prob'].append(this_prob)
    
    results['bbox'] = np.concatenate(results['bbox'], 1)
    results['iou']  = np.concatenate(results['iou'], 1)
    results['prob'] = np.concatenate(results['prob'], 1)    
    return results

def split_testing_(cls, img,  batch_size = 4, windowsize=None, thresh= None, cfg=None):

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
    
    print(row_window, col_window)
    num_row, num_col = math.ceil(row_size/row_window),  math.ceil(col_size/col_window) # lower int
    
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

            batch_data = to_device(this_patch, cls.device_id, volatile=True)
            bbox_pred, iou_pred, prob_pred  = cls.forward(batch_data)
            
            bbox_pred = bbox_pred.cpu().data.numpy()
            iou_pred = iou_pred.cpu().data.numpy()
            prob_pred = prob_pred.cpu().data.numpy()
            
            H, W = cls.out_size
            x_ratio, y_ratio = cls.x_ratio, cls.y_ratio
            
            #bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred,
            #                            this_patch.shape[1::], cls.out_size , cfg, thresh)
            bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred,
                                       cls.inp_size, cls.out_size, cfg, thresh)                     
            # bbox_pred = yolo_to_bbox(
            #             np.ascontiguousarray(bbox_pred, dtype=np.float),
            #             np.ascontiguousarray(cfg.anchors, dtype=np.float),
            #             H, W,
            #             x_ratio, y_ratio)
            # bboxes, scores, cls_inds = yolo_utils.postprocess_bbox(bbox_pred, iou_pred, prob_pred, this_patch.shape[2::], cfg, 0.5)
            
            np_start  = np.array([[col_start, row_start, col_start, row_start]]  ) 
            
            results['bbox'].append(bboxes +  np_start )
            results['iou'].append(iou_pred)
            results['prob'].append(prob_pred)
    
    results['bbox'] = np.concatenate(results['bbox'], 0)
    results['iou'] = np.concatenate(results['iou'], 1)
    results['prob'] = np.concatenate(results['prob'], 1)

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
    
    print(row_window, col_window)
    num_row, num_col = math.ceil(row_size/row_window),  math.ceil(col_size/col_window) # lower int
    
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
            bbox_pred, iou_pred, prob_pred  = cls.forward(batch_data)
            
            bbox_pred = bbox_pred.cpu().data.numpy()
            iou_pred = iou_pred.cpu().data.numpy()
            prob_pred = prob_pred.cpu().data.numpy()
            
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
    
    results['bbox'] = np.concatenate(results['bbox'], 1)
    results['iou'] = np.concatenate(results['iou'], 1)
    results['prob'] = np.concatenate(results['prob'], 1)

    return results


def test_eng(dataloader, model_root,save_root, mode_name, net, args, cfg):
    net.eval()
    model_folder = os.path.join(model_root, mode_name)
    save_folder = os.path.join(save_root, mode_name)
    mkdirs(save_folder)
    bbox_file = os.path.join(save_folder, 'bbox_file.pkl')
    
    weightspath = os.path.join(model_folder, 'weights_epoch{}.pth'.format(args.load_from_epoch))
    weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    
    print('reload weights from {}'.format(weightspath))
    net.load_state_dict(weights_dict)

    train_loss = 0
    bbox_loss, iou_loss, cls_loss = 0., 0., 0.

    num_images = dataloader.num_images # cause testing has to be of batch size 1
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(1)]

    print('num_images: ', num_images)
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for i in range(num_images):
        
        batch = dataloader.next()
        ori_im   = batch['origin_im'][0]
        img_name = batch['dontcare'][0]
        im_np = batch['images'][0]
        windowsize = 512 # it has to be dividable by 32 to have no shifts.

        #bbox_pred, iou_pred, prob_pred = net(im_data)
        #import pdb; pdb.set_trace()
        #print(ori_im.dtype, ori_im.min(), ori_im.max(), im_np.shape)
        # im_data = to_device(im_np[None], net.device_id, volatile=True)
        # bbox_pred, iou_pred, prob_pred = net(im_data)
        # bbox_pred, iou_pred, prob_pred = bbox_pred.cpu().data.numpy(), \
        #                                 iou_pred.cpu().data.numpy(), \
        #                                 prob_pred.cpu().data.numpy()
        #bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred,im_np.shape[1::],net.out_size, cfg, thresh)

        _t['im_detect'].tic()
        result_dict = split_testing(net, im_np,  batch_size = 4, windowsize= windowsize, thresh=thresh, cfg=cfg)
        bbox_pred, iou_pred, prob_pred  = result_dict['bbox'], result_dict['iou'], result_dict['prob']
        #bboxes = bbox_pred
        #print(bbox_pred.shape, iou_pred.shape, prob_pred.shape)
        detect_time = _t['im_detect'].toc()
        _t['misc'].tic()

        bboxes, scores, cls_inds = yolo_utils.postprocess_bbox(bbox_pred, iou_pred, prob_pred,im_np.shape[1::], cfg, thresh)
        utils_time = _t['misc'].toc()
        
        print('detection time {}, post_processing time {}'.format(detect_time,utils_time)) 
        ori_im = ori_im.transpose(1,2,0)
        #import pdb; pdb.set_trace()    
        bboxes = bboxes / args.resize_ratio
        overlaid_img = dataloader.overlay_bbox(ori_im, bboxes, len=6)
        overlaid_img = overlaid_img.astype(np.uint8)
        
        #org_size_written = imresize(overlaid_img ,  1.0/args.resize_ratio)
        writeImg(overlaid_img, os.path.join(save_folder, img_name))
        #break
        # for bbox in bboxes:
        #     print(bbox)

        # if True:
        #     im2show = yolo_utils.draw_detection(ori_im, bboxes, scores, cls_inds, cfg, thr=0.1)
        #     if im2show.shape[0] > 1100:
        #         im2show = cv2.resize(im2show, (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))
        #     cv2.imshow('test', im2show)
        #     cv2.waitKey(0)
            
    with open(bbox_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    
