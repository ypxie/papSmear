# -*- coding: utf-8 -*-
import os, sys, pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pool
from functools import partial

from .proj_utils import network as net_utils
from .utils.cython_bbox import bbox_ious, anchor_intersections
from .utils.cython_yolo import yolo_to_bbox
from .proj_utils.model_utils import *

from .proj_utils.torch_utils import to_device
#from .proj_utils.local_utils import split_testing

class tinyFCN(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, inchan=3, activ= 'relu'):
        super(tinyFCN, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        
        self.in_tr_100     = InputTransition(inchan, 32, activ = activ)
        
        self.down_50       = DownTransition(32, 64, 1,   activ = activ)
        self.down_25       = DownTransition(64, 128, 1,  activ = activ)
        self.down_12       = DownTransition(128, 256, 1, dropout = 0.25, activ = activ)
        
        self.up_25         = UpConcat(256, 128, 128, 1,  catChans= 128, dropout=0.25, activ = activ)
        self.up_50         = UpConcat(128, 64,  64,  1,  catChans= 64,  dropout=0.25, activ = activ)
        self.up_100        = UpConv(64, 32, activ = activ)
        self.out_tr        = OutputTransition(32, 1, 32, activ = activ)

    def forward(self, inputs, testing=False, pred_step=None, stateful=False):
        # the input should be of shape (B, C, W, H)

        x       = to_device(inputs, self.device_id)
        out_100 = self.in_tr_100(x)
        out_50  = self.down_50(out_100)
        out_25  = self.down_25(out_50)
        out_12  = self.down_12(out_25)

        self.out_12_size = out_12.size()[2:]
        self.out_25_size = out_25.size()[2:]
        self.out_50_size = out_50.size()[2:]
        self.out_100_size = x.size()[2:]

        up_25  = self.up_25(out_12, out_25)
        up_50  = self.up_50(up_25, out_50)
        up_100 = self.up_100(up_50, self.out_100_size)
        out    = F.sigmoid(self.out_tr(up_100))
        if testing:
            out = out.data # to_device(out.cpu().data, self.device_id)
        return out
        
class MakeLayers(nn.Module):
    def __init__(self, in_channels, net_cfg, res_blocks=False):
        super(MakeLayers, self).__init__()
        pre_layers = []
        res_layers = []
        init_flag = True
        self.use_residual = False

        if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
            for sub_cfg in net_cfg:
                sub_layer   = MakeLayers(in_channels, sub_cfg)
                in_channels = sub_layer.out_chan
                pre_layers.append(sub_layer)
        else:
            for idx, item in enumerate(net_cfg):
                if item == 'M':
                    #pre_layers.append(padConv2d(in_channels,  in_channels, kernel_size = 3, stride=2, bias=False) )
                    #pre_layers.append(nn.LeakyReLU(0.1, inplace=True) )
                    pre_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    out_channels, ksize = item
                    if init_flag or (not res_blocks):
                        pre_layers.append(net_utils.Conv2d_BatchNorm(in_channels, out_channels, ksize, same_padding=True))
                        init_flag = False
                    else:
                        self.use_residual = True
                        use_relu = idx != (len(net_cfg) - 1)
                        res_layers.append(net_utils.Conv2d_BatchNorm(in_channels, out_channels, ksize, same_padding=True))

                    in_channels = out_channels
        self.pre_res  =   nn.Sequential(*pre_layers)   
        self.res_path =   nn.Sequential(*res_layers)    
        self.out_chan =   in_channels
        self.activ    =   nn.LeakyReLU(0.1, inplace=True)

    def forward(self, inputs):
        # TODO do we need to add activation? 
        # CycleGan regards this. I guess to prevent spase gradients
        pre_res = self.pre_res(inputs)
        if self.use_residual:
            res_path = self.res_path(pre_res)
            return self.activ(pre_res + res_path)
        else:
            return pre_res

def _process_batch(inputs, size_spec=None, cfg=None):
    inp_size, out_size = size_spec
    H, W = out_size

    x_ratio, y_ratio = float(inp_size[1])/W, float(inp_size[0])/H
    bbox_pred_np, gt_boxes, gt_classes, dontcares, iou_pred_np = inputs

    # net output
    hw, num_anchors, _ = bbox_pred_np.shape

    # gt
    _classes = np.zeros([hw, num_anchors, cfg.num_classes], dtype=np.float)
    _class_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)

    _ious = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _iou_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)

    _boxes = np.zeros([hw, num_anchors, 4], dtype=np.float)
    _boxes[:, :, 0:2] = 0.5
    _boxes[:, :, 2:4] = 1.0
    _box_mask = np.zeros([hw, num_anchors, 1], dtype=np.float) + 0.01
    
    # scale pred_bbox
    anchors = np.ascontiguousarray(cfg.anchors, dtype=np.float)
    bbox_pred_np = np.expand_dims(bbox_pred_np, 0)

    bbox_np = yolo_to_bbox(
        np.ascontiguousarray(bbox_pred_np, dtype=np.float),
        anchors,
        H, W,
        x_ratio, y_ratio)
    # for each prediction, calculate in 8*8 with all corresponsding anchors.
    bbox_np = bbox_np[0]  # bbox_np = (hw, num_anchors, (x1, y1, x2, y2))   range: 0 ~ 1

    # gt_boxes_b = np.asarray(gt_boxes[b], dtype=np.float)
    gt_boxes_b = np.asarray(gt_boxes, dtype=np.float)

    # for each cell, compare predicted_bbox and gt_bbox
    bbox_np_b = np.reshape(bbox_np, [-1, 4])
    ious = bbox_ious(
        np.ascontiguousarray(bbox_np_b,  dtype=np.float),
        np.ascontiguousarray(gt_boxes_b, dtype=np.float)
    )
    # for each assumed box, find the best-matched with ground-truth
    best_ious = np.max(ious, axis=1).reshape(_iou_mask.shape)

    iou_penalty = 0 - iou_pred_np[best_ious < cfg.iou_thresh]
    _iou_mask[best_ious <= cfg.iou_thresh] = cfg.noobject_scale * iou_penalty 
    
    # locate the cell of each gt_boxes_b
    cx = (gt_boxes_b[:, 0] + gt_boxes_b[:, 2]) * 0.5 / x_ratio
    cy = (gt_boxes_b[:, 1] + gt_boxes_b[:, 3]) * 0.5 / y_ratio
    cell_inds = np.floor(cy) * W + np.floor(cx)
    cell_inds = cell_inds.astype(np.int)

    # transfer ground-truth box to 8*8 format
    target_boxes = np.empty(gt_boxes_b.shape, dtype=np.float)
    target_boxes[:, 0] = cx - np.floor(cx)  # cx
    target_boxes[:, 1] = cy - np.floor(cy)  # cy
    target_boxes[:, 2] = (gt_boxes_b[:, 2] - gt_boxes_b[:, 0]) # / inp_size[0] * out_size[0]  # tw
    target_boxes[:, 3] = (gt_boxes_b[:, 3] - gt_boxes_b[:, 1]) # / inp_size[1] * out_size[1]  # th

    # for each gt boxes, match the best anchor and save the index
    gt_boxes_resize = np.copy(gt_boxes_b) # I don't need to resize
    anchor_ious = anchor_intersections(
        anchors,
        np.ascontiguousarray(gt_boxes_resize, dtype=np.float)
    )
    anchor_inds = np.argmax(anchor_ious, axis=0)

    ious_reshaped = np.reshape(ious, [hw, num_anchors, len(cell_inds)])
    for i, cell_ind in enumerate(cell_inds):
        if cell_ind >= hw or cell_ind < 0:
            print(cell_ind)
            continue
        anchor_ = anchor_inds[i]

        iou_pred_cell_anchor = iou_pred_np[cell_ind, anchor_, :]  # 0 ~ 1, should be close to 1
        _iou_mask[cell_ind, anchor_, :] = cfg.object_scale * (1 - iou_pred_cell_anchor)
        # _ious[cell_ind, anchor_, :] = anchor_ious[a, i]
        _ious[cell_ind, anchor_, :] = ious_reshaped[cell_ind, anchor_, i]

        _box_mask[cell_ind, anchor_, :] = cfg.coord_scale
        target_boxes[i, 2:4]  /= anchors[anchor_]
        _boxes[cell_ind, anchor_, :] = target_boxes[i]
        
        _class_mask[cell_ind, anchor_, :] = cfg.class_scale
        _classes[cell_ind, anchor_, gt_classes[i]] = 1.

    return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x, small_size):
        stride = self.stride

        x = match_tensor(x, (2*small_size[0], 2*small_size[1]))

        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = int(stride)
        hs = int(stride)
        x = x.contiguous().view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.contiguous().view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.contiguous().view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.contiguous().view(B, hs*ws*C, H//hs, W//ws)
        return x


class Darknet19(nn.Module):
    def __init__(self, cfg, seg_inchan = 35):
        super(Darknet19, self).__init__()
        self.cfg = cfg
        self.seg_net = tinyFCN(inchan=seg_inchan)
        self.register_buffer('device_id', torch.IntTensor(1))
        self.seg_inchan = seg_inchan

        net_cfgs = [
            # conv0s
            [(32, 3)],  # 0
            # conv1
            ['M', (64, 3)], # 1
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)], # 4
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)], # 5
            # ------------
            # conv3
            [(1024, 3), (1024, 3)],  # 6
            # conv4
            [(1024, 3)]  # 7
        ]

        # darknet
        self.conv0   = MakeLayers(3,  net_cfgs[0])
        self.conv1   = MakeLayers(self.conv0.out_chan,  net_cfgs[1:5])
        # --- 
        self.conv2   = MakeLayers(self.conv1.out_chan,  net_cfgs[5])
        self.conv3   = MakeLayers(self.conv2.out_chan,  net_cfgs[6])
        
        stride = 2
        self.reorg = Reorg(stride=2)   # stride*stride times the channels of conv1s
        # cat [conv1s, conv3]
        self.conv4 = MakeLayers((self.conv1.out_chan*(stride*stride) + self.conv3.out_chan), net_cfgs[7])
        
        #self.conv_shrink   = net_utils.Conv2d(c1+3, 16, 1)  
        # linear
        out_channels = cfg.num_anchors * (cfg.num_classes + 5)
        self.conv5   = net_utils.Conv2d(self.conv4.out_chan, out_channels, 1, 1, relu=False)
        #padConv2d(self.conv4.out_chan, out_channels, kernel_size = 1, stride=1, bias=True)
        

        # train
        self.bbox_loss = None
        self.iou_loss = None
        self.cls_loss = None
        self.pool = Pool(processes=4)
        
    @property
    def loss(self):
        return self.bbox_loss + self.iou_loss + self.cls_loss

    def forward(self, im_data, gt_boxes=None, gt_classes=None, dontcare=None):
        self.inp_size = im_data.size()[2:] # 256*256
        inp_size_tup  = (self.inp_size[0], self.inp_size[1])
        conv0   = self.conv0(im_data)
        conv1   = self.conv1(conv0)
        conv2   = self.conv2(conv1)

        conv3   = self.conv3(conv2)
        #up1s =  nn.UpsamplingBilinear2d(inp_size_tup)(conv1s)
        #up2s =  nn.UpsamplingBilinear2d(inp_size_tup)(conv2s)
        #up4s =  nn.UpsamplingBilinear2d(inp_size_tup)(conv4s)
        if self.seg_inchan == 35:
            cat_124   = torch.cat([im_data, conv0], 1)
            large_map  = cat_124
        else:
            large_map = im_data

        #conv2 = self.conv2(conv1s)
        conv1_reorg = self.reorg(conv1, conv3.size()[2::])
        
        cat_reorg = torch.cat([conv1_reorg, conv3], 1)
        conv4 = self.conv4(cat_reorg)
        conv5 = self.conv5(conv4)   # batch_size, out_channels, h, w

        self.out_size = conv4.size()[2::] # 8*8
        self.x_ratio = float(self.inp_size[1])/self.out_size[1] # 32
        self.y_ratio = float(self.inp_size[0])/self.out_size[0] # 32

        # for detection
        # bsize, c, h, w -> bsize, h, w, c -> bsize, h x w, num_anchors, 5+num_classes
        bsize, _, h, w = conv5.size()
        # assert bsize == 1, 'detection only support one image per batch'
        conv5_reshaped = conv5.permute(0, 2, 3, 1).contiguous().view(bsize, -1, self.cfg.num_anchors, self.cfg.num_classes + 5)
        
        #boundary_mask = conv5_reshaped.detach().clone().zero_()
        #boundary_mask[]

        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_pred = F.sigmoid(conv5_reshaped[:, :, :, 0:2])
        wh_pred = torch.exp(conv5_reshaped[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)
        iou_pred = F.sigmoid(conv5_reshaped[:, :, :, 4:5])
        score_pred = conv5_reshaped[:, :, :, 5:].contiguous()
        prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)

        # for training
        if self.training:
            bbox_pred_np = bbox_pred.data.cpu().numpy()
            iou_pred_np = iou_pred.data.cpu().numpy()
            
            _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = self._build_target(
                bbox_pred_np, gt_boxes, gt_classes, dontcare, iou_pred_np)

            _boxes     = to_device(_boxes, self.device_id, requires_grad=False)
            _ious      = to_device(_ious, self.device_id , requires_grad=False)
            _classes   = to_device(_classes, self.device_id, requires_grad=False)
            box_mask   = to_device(_box_mask, self.device_id, requires_grad=False)
            iou_mask   = to_device(_iou_mask, self.device_id, requires_grad=False)
            class_mask = to_device(_class_mask, self.device_id, requires_grad=False)


            num_boxes = sum((len(boxes) for boxes in gt_boxes))
            box_mask = box_mask.expand_as(_boxes)
            
            self.bbox_loss = nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes
            self.iou_loss  = nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes

            class_mask = class_mask.expand_as(prob_pred)
            self.cls_loss = nn.MSELoss(size_average=False)(prob_pred * class_mask, _classes * class_mask) / num_boxes

        return bbox_pred, iou_pred, prob_pred, large_map


    def _build_target(self, bbox_pred_np, gt_boxes, gt_classes, dontcare, iou_pred_np):
        """
        :param bbox_pred_np: shape: (bsize, h x w, num_anchors, 4) : (sig(tx), sig(ty), exp(tw), exp(th))
        """
        bsize = bbox_pred_np.shape[0]
        #print('bbox pred: ',bbox_pred_np.shape)
        _process_batch_func = partial(_process_batch, size_spec = (self.inp_size, self.out_size) ,cfg=self.cfg)
        targets = self.pool.map(_process_batch_func, ( (bbox_pred_np[b], gt_boxes[b], gt_classes[b], dontcare[b], iou_pred_np[b])for b in range(bsize)))
        #targets = []

        _boxes = np.stack(tuple((row[0] for row in targets)))
        _ious = np.stack(tuple((row[1] for row in targets)))
        _classes = np.stack(tuple((row[2] for row in targets)))
        _box_mask = np.stack(tuple((row[3] for row in targets)))
        _iou_mask = np.stack(tuple((row[4] for row in targets)))
        _class_mask = np.stack(tuple((row[5] for row in targets)))

        return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask


class BreastNet(nn.Module):
    def __init__(self, cfg, seg_inchan=35):
        super(BreastNet, self).__init__()
        self.cfg = cfg
        self.seg_net = tinyFCN(inchan=seg_inchan)
        self.register_buffer('device_id', torch.IntTensor(1))
        self.seg_inchan = seg_inchan

        net_cfgs = [
            # conv0s
            [(32, 3)],  # 0
            # conv1
            ['M', (64, 3)], # 1
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            # conv2
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)], # 4
            # ------------
            # conv3
            [(1024, 3), (1024, 3)],  # 6
            # conv4
            [(1024, 3)]  # 7
        ]

        # darknet
        self.conv0   = MakeLayers(3,  net_cfgs[0])
        self.conv1   = MakeLayers(self.conv0.out_chan,  net_cfgs[1:4])
        # --- 
        self.conv2   = MakeLayers(self.conv1.out_chan,  net_cfgs[4])
        self.conv3   = MakeLayers(self.conv2.out_chan,  net_cfgs[5])
        
        stride = 2
        self.reorg = Reorg(stride=2)   # stride*stride times the channels of conv1s
        # cat [conv1s, conv3]
        self.conv4 = MakeLayers((self.conv1.out_chan*(stride*stride) + self.conv3.out_chan), net_cfgs[6])
        
        #self.conv_shrink   = net_utils.Conv2d(c1+3, 16, 1)  
        # linear
        out_channels = cfg.num_anchors * (cfg.num_classes + 5)
        self.conv5   = net_utils.Conv2d(self.conv4.out_chan, out_channels, 1, 1, relu=False)
        #padConv2d(self.conv4.out_chan, out_channels, kernel_size = 1, stride=1, bias=True)

        # train
        self.bbox_loss = None
        self.iou_loss = None
        self.cls_loss = None
        self.pool = Pool(processes=4)

    @property
    def loss(self):
        return self.bbox_loss + self.iou_loss + self.cls_loss

    def forward(self, im_data, gt_boxes=None, gt_classes=None, dontcare=None):
        self.inp_size = im_data.size()[2:] # 256*256
        inp_size_tup  = (self.inp_size[0], self.inp_size[1])
        conv0   = self.conv0(im_data)
        conv1   = self.conv1(conv0)
        conv2   = self.conv2(conv1)
        conv3   = self.conv3(conv2)
        #up1s =  nn.UpsamplingBilinear2d(inp_size_tup)(conv1s)
        #up2s =  nn.UpsamplingBilinear2d(inp_size_tup)(conv2s)
        #up4s =  nn.UpsamplingBilinear2d(inp_size_tup)(conv4s)
        if self.seg_inchan == 35:
            cat_124   = torch.cat([im_data, conv0], 1)
            large_map  = cat_124
        else:
            large_map = im_data

        #conv2 = self.conv2(conv1s)
        conv1_reorg = self.reorg(conv1, conv3.size()[2::])
        cat_reorg   = torch.cat([conv1_reorg, conv3], 1)

        conv4 = self.conv4(cat_reorg)
        conv5 = self.conv5(conv4)   # batch_size, out_channels, h, w

        self.out_size = conv4.size()[2::] # 8*8
        self.x_ratio = float(self.inp_size[1])/self.out_size[1] # 32
        self.y_ratio = float(self.inp_size[0])/self.out_size[0] # 32

        # for detection
        # bsize, c, h, w -> bsize, h, w, c -> bsize, h x w, num_anchors, 5+num_classes
        bsize, _, h, w = conv5.size()
        # assert bsize == 1, 'detection only support one image per batch'
        conv5_reshaped = conv5.permute(0, 2, 3, 1).contiguous().view(bsize, -1, self.cfg.num_anchors, self.cfg.num_classes + 5)

        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_pred = F.sigmoid(conv5_reshaped[:, :, :, 0:2])
        wh_pred = torch.exp(conv5_reshaped[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)
        iou_pred = F.sigmoid(conv5_reshaped[:, :, :, 4:5])
        score_pred = conv5_reshaped[:, :, :, 5:].contiguous()
        prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)

        # for training
        if self.training:
            bbox_pred_np = bbox_pred.data.cpu().numpy()
            iou_pred_np = iou_pred.data.cpu().numpy()
            
            _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = self._build_target(
                bbox_pred_np, gt_boxes, gt_classes, dontcare, iou_pred_np)

            _boxes     = to_device(_boxes, self.device_id, requires_grad=False)
            _ious      = to_device(_ious, self.device_id , requires_grad=False)
            _classes   = to_device(_classes, self.device_id, requires_grad=False)
            box_mask   = to_device(_box_mask, self.device_id, requires_grad=False)
            iou_mask   = to_device(_iou_mask, self.device_id, requires_grad=False)
            class_mask = to_device(_class_mask, self.device_id, requires_grad=False)


            num_boxes = sum((len(boxes) for boxes in gt_boxes))
            box_mask = box_mask.expand_as(_boxes)

            self.bbox_loss = nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes
            self.iou_loss = nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes

            class_mask = class_mask.expand_as(prob_pred)
            self.cls_loss = nn.MSELoss(size_average=False)(prob_pred * class_mask, _classes * class_mask) / num_boxes

        return bbox_pred, iou_pred, prob_pred, large_map


    def _build_target(self, bbox_pred_np, gt_boxes, gt_classes, dontcare, iou_pred_np):
        """
        :param bbox_pred_np: shape: (bsize, h x w, num_anchors, 4) : (sig(tx), sig(ty), exp(tw), exp(th))
        """
        bsize = bbox_pred_np.shape[0]
        #print('bbox pred: ',bbox_pred_np.shape)
        _process_batch_func = partial(_process_batch, size_spec = (self.inp_size, self.out_size) ,cfg=self.cfg)
        targets = self.pool.map(_process_batch_func, ( (bbox_pred_np[b], gt_boxes[b], gt_classes[b], dontcare[b], iou_pred_np[b])for b in range(bsize)))
        #targets = []

        _boxes = np.stack(tuple((row[0] for row in targets)))
        _ious = np.stack(tuple((row[1] for row in targets)))
        _classes = np.stack(tuple((row[2] for row in targets)))
        _box_mask = np.stack(tuple((row[3] for row in targets)))
        _iou_mask = np.stack(tuple((row[4] for row in targets)))
        _class_mask = np.stack(tuple((row[5] for row in targets)))

        return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask
