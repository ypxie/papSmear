import os, sys, pdb
import numpy as np


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


# # input and output size
# ############################
# inp_size = np.array([416, 416], dtype=np.int)   # w, h
# out_size = inp_size / 32


def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


class config:
    def __init__(self):
        # for display
        base = int(np.ceil(pow(num_classes, 1. / 3)))
        colors = [_to_color(x, base) for x in range(num_classes)]


        # dir config
        ############################
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        DATA_DIR = os.path.join(ROOT_DIR, 'data')
        MODEL_DIR = os.path.join(ROOT_DIR, 'models')
        TRAIN_DIR = os.path.join(MODEL_DIR, 'training')
        TEST_DIR = os.path.join(MODEL_DIR, 'testing')

        h5_fname = 'yolo-pap.weights.h5'
        trained_model = os.path.join(MODEL_DIR, h5_fname)
        pretrained_fname = 'darknet19.weights.npz'
        pretrained_model = os.path.join(MODEL_DIR, pretrained_fname)
        exp_name = 'darknet19_pap'
        train_output_dir = os.path.join(TRAIN_DIR, exp_name)
        mkdir(train_output_dir, max_depth=3)
        # pap_test = ""
        # test_output_dir = os.path.join(TEST_DIR, pap_test, h5_fname)
        # mkdir(test_output_dir, max_depth=4)
        batch_size = 1
        train_batch_size = 16


        # detection config
        ############################
        thresh = 0.3

        rand_seed = 1024
        use_tensorboard = True

        log_interval = 50
        disp_interval = 10
        display_freq = 100


        # pap_smear infor
        label_names = ['fake class']
        num_classes = len(label_names)
        anchors = np.asarray( [[  39.59209245 ,  38.94793038],
                                [  95.06052666 ,  76.82550108],
                                [ 172.93104188 , 132.40287078],
                                [  62.81947797,   65.13911612],
                                [ 175.75465577, 198.25737813],
                                [ 123.27760599,  149.23073386],
                                [ 292.34331914 , 281.81200181],
                                [  86.81196254 , 113.76905313],
                                [ 128.32483899 , 105.76417358]])
        num_anchors = len(anchors)


        # for training yolo2
        object_scale = 5.
        noobject_scale = 1.
        class_scale = 1.
        coord_scale = 1.
        iou_thresh = 0.6


        # training arguments
        start_step = 0
        lr_decay_epochs = [60, 90]
        lr_decay = 1./10
        max_epoch = 160
        weight_decay = 0.0005
        momentum = 0.9
        init_learning_rate = 1e-3

        self.__dict__.update(locals())

cfg = config()