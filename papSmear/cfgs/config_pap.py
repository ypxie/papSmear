# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
from os.path import dirname as opd

# def mkdir(path, max_depth=3):
#     parent, child = os.path.split(path)
#     if not os.path.exists(parent) and max_depth > 1:
#         mkdir(parent, max_depth-1)
#
#     if not os.path.exists(path):
#         os.mkdir(path)

# def _to_color(indx, base):
#     """ return (b, r, g) tuple"""
#     base2 = base * base
#     b = 2 - indx / base2
#     r = 2 - (indx % base2) / base
#     g = 2 - (indx % base2) % base
#     return b * 127, r * 127, g * 127


class config:
    def __init__(self):
        rand_seed = 1024
        use_tensorboard = True

        # pap_smear infor
        label_names = ['fake class']
        num_classes = len(label_names)

        # base = int(np.ceil(pow(num_classes, 1. / 3)))         # for display
        # colors = [_to_color(x, base) for x in range(num_classes)]

        anchors = np.asarray( [[  93.2755673,    76.6358055 ],
                                [ 171.9592885,   131.77937709],
                                [  86.79261832,  114.50667529],
                                [ 288.08578916,  274.38914068],
                                [  39.34243066,   38.42535943],
                                [  62.05614502,   64.58004901],
                                [ 127.33594888,  105.14651342],
                                [ 123.77266986,  148.97959267],
                                [ 175.22840536,  196.91588804]])
        num_anchors = len(anchors)

        object_scale = 5.
        noobject_scale = 1.
        class_scale = 1.
        coord_scale = 1.
        iou_thresh = 0.6

        self.__dict__.update(locals())

cfg = config()
