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

        # base = int(np.ceil(pow(num_classes, 1. / 3)))         # for display
        # colors = [_to_color(x, base) for x in range(num_classes)]

        # pap_smear infor
        label_names = ['fake class']
        num_classes = len(label_names)

        anchors = np.asarray( [[ 14.66110707 , 15.02742004],
                                [ 33.51202285,  26.13682675],
                                [ 31.24160307 , 38.53895268],
                                [ 23.85007885 , 30.81161072],
                                [ 18.073727   , 23.17147331],
                                [ 44.46996632 , 33.13864667],
                                [ 54.41738574 , 45.93044506],
                                [ 24.92695851 , 19.83214731],
                                [ 37.65221488 , 50.95304277]])
        num_anchors = len(anchors)

        iou_thresh = 0.6

        object_scale = 5.
        noobject_scale = 1.
        resize_ratio = [1, 0.8, 0.6]
        coord_scale = 1.
        class_scale = 1.

        self.__dict__.update(locals())

cfg = config()
