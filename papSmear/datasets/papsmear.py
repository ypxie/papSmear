# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
import h5py
from scipy.io import loadmat
from multiprocessing import Pool
from numba import jit
from ..proj_utils.local_utils import getfileinfo, writeImg, imread, imresize

debug_mode = True

# Load annotated bounding box mat file
def load_mat(thismatfile, contourname_list=['Contours']):
    # First try load using h5py; then try using scipy.io.loadmat
    try:
        mat_file = h5py.File(thismatfile, 'r')
        for contourname in contourname_list:
            if contourname in list(mat_file.keys()):
                contour_mat = [np.transpose(mat_file[element[0]][:]) for element in mat_file[contourname]]
                break
        mat_file.close()
    except:
        loaded_mt = loadmat(thismatfile)
        for contourname in contourname_list:
            if contourname in loaded_mt.keys():
                contour_mat = []
                cnts = loaded_mt[contourname].tolist()
                if len(cnts) > 0:
                    contour_mat = cnts[0]
                break
    return contour_mat

# Resize bounding box to a certain ratio
def resize_mat(contour_mat, resize_ratio):
    numCell = len(contour_mat)
    res_contour = []
    for icontour in range(0, numCell):
        thiscontour = contour_mat[icontour]
        res_contour.append(thiscontour * resize_ratio)
    return res_contour


@jit
def get_bbox(contour_mat):
    numCell = len(contour_mat)
    bbox_list = []
    for icontour in range(0, numCell):
        thiscontour = contour_mat[icontour]
        xcontour = np.reshape(thiscontour[0,:], (1,-1) )
        ycontour = np.reshape(thiscontour[1,:], (1,-1) )

        x_min, x_max = np.min(xcontour), np.max(xcontour)
        y_min, y_max = np.min(ycontour), np.max(ycontour)
        bbox_list.append([x_min, y_min, x_max, y_max] )
    return bbox_list


@jit
def overlay_bbox(img, bbox,len=1):
    for bb in bbox:
        x_min_, y_min_, x_max_, y_max_ = bb
        x_min_, y_min_, x_max_, y_max_ = int(x_min_),int( y_min_), int(x_max_), int(y_max_)
        img[:,:,0] = change_val(img[:,:,0], 255, len, x_min_, y_min_, x_max_, y_max_)
        img[:,:,1] = change_val(img[:,:,1], 0, len,  x_min_, y_min_, x_max_, y_max_)
        img[:,:,2] = change_val(img[:,:,2], 0, len,  x_min_, y_min_, x_max_, y_max_)
    return img



@jit(nopython=True)
def change_val(img,val, len, x_min, y_min, x_max, y_max):
    left_len  = (len-1)//2
    right_len = (len-1) - left_len
    row_size, col_size = img.shape[0:2]
    for le in range(-left_len, right_len + 1):
        y_min_ = max(0, y_min + le )
        x_min_ = max(0, x_min + le )

        y_max_ = min(row_size, y_max - le )
        x_max_ = min(col_size, x_max - le )

        img[y_min_:y_max_, x_min_:x_min_+1] = val
        img[y_min_:y_min_+1, x_min_:x_max_] = val
        img[y_min_:y_max_, x_max_:x_max_+1] = val
        img[y_max_:y_max_+1, x_min_:x_max_] = val
    return img



def get_anchor(row_size, col_size, img_shape, boarder=0):
    dst_row, dst_col = img_shape
    br, bc = int(boarder*row_size), int(boarder*col_size)

    upleft_row  = np.random.randint(br, row_size - dst_row - br)
    upleft_col  = np.random.randint(bc, col_size - dst_col - bc)
    return (upleft_row, upleft_col)



def crop_bbox(bbox_list, r_min, c_min, r_max, c_max):
    numCell = len(bbox_list)
    new_bbox = []
    for idx in range(0, numCell):
        this_bbox = bbox_list[idx]
        x_min_, y_min_, x_max_, y_max_ = this_bbox

        x_min = x_min_  - c_min
        x_max = x_max_  - c_min
        y_min = y_min_  - r_min
        y_max = y_max_  - r_min

        row_size = r_max - r_min + 1
        col_size = c_max - c_min + 1

        if(x_min < 0 or y_min < 0 or x_max >=  row_size or y_max >= col_size):
            x_min = max(x_min, 3)
            x_max = min(x_max, col_size-3)
            y_min = max(y_min, 3)
            y_max = min(y_max, row_size-3)
            new_col_len = x_max - x_min + 1
            old_col_len = x_max_-x_min_ +1
            new_row_len = y_max - y_min + 1
            old_row_len = y_max_-y_min_ +1

            if (new_row_len > 0.6 *old_row_len ) and (new_col_len > 0.6 *old_col_len ):
                new_bbox.append([x_min, y_min, x_max, y_max] )
        else:
            new_bbox.append([x_min, y_min, x_max, y_max])
    return new_bbox




def _get_next(inputs):
    img_data, mat_data, img_path, mat_path, resize_ratio, img_shape, testing = inputs

    org_img = imread(img_path) if debug_mode else img_data #
    org_mat = load_mat(mat_path, contourname_list=['Contours']) if debug_mode else mat_data #

    try_box = 0
    while try_box <= 10:
        try_count = 0
        while True:
            chosen_ratio = resize_ratio[np.random.randint(0, len(resize_ratio)-1)]

            if try_count >= 10:
                chosen_ratio = 1.1 * max( float(dst_row)/float(row_size), float(dst_col)/float(col_size) )

            res_img = imresize(org_img, chosen_ratio)
            res_mat = resize_mat(org_mat, chosen_ratio)

            row_size, col_size, _ = res_img.shape

            if testing is True:
                img_shape = [row_size, col_size]
            dst_row, dst_col = img_shape

            if row_size >= dst_row and col_size >= dst_col:
                up_r, up_c = get_anchor(row_size, col_size, img_shape, boarder=0)
                break
            else:
                try_count += 1
                continue

        bbox =  get_bbox(res_mat)
        r_min, c_min, r_max, c_max = up_r, up_c, up_r+dst_row, up_c+dst_col
        this_patch = res_img[r_min:r_max, c_min:c_max, :]

        this_patch = this_patch.transpose(2, 0, 1)
        this_bbox = crop_bbox(bbox, r_min, c_min, r_max, c_max)
        num_bbox  = len(this_bbox)
        classes = np.zeros((num_bbox), dtype=np.int32)
        if len(this_bbox) != 0:
           break
        else:
            try_box += 1

    return (this_patch, this_bbox, classes, img_path, res_img)


class papSmearData:
    def __init__(self, data_dir, batch_size=8, processes=8, img_shape=(256, 256),
                 resize_ratio=[0.3, 0.5, 0.6], testing=False):
        self.__dict__.update(locals())

        all_dict_list  = getfileinfo(self.data_dir, ['_gt'], ['.png'], '.mat')
        self.img_list_ = [this_dict['thisfile']    for this_dict in all_dict_list]
        self.mat_list_ = [this_dict['thismatfile'] for this_dict in all_dict_list]

        self.img_list  = self.img_list_ if debug_mode else [imread(img_path) for img_path in self.img_list_]
        self.mat_list  = self.mat_list_ if debug_mode else [load_mat(mat_path, contourname_list=['Contours']) for mat_path in self.mat_list_]
        self.img_num      = len(all_dict_list)
        self.img_shape    = img_shape
        self._img_shape   = img_shape

        self.resize_ratio = resize_ratio
        self.testing = testing
        self.overlay_bbox = overlay_bbox
        self._classes = ['fake class']
        self._epoch = 0
        self.count = 0
        self.batch_count = 0

        self._pool_processes = processes
        self.pool = Pool(self._pool_processes)
        self._shuffle = True

        self.indices = list(range(self.img_num))
        self.start = 0

    def get_all_bbox(self):
        bbox_all = []
        for idx in range(self.img_num):
            # org_mat = self.mat_list[idx]
            if debug_mode:
                org_mat = load_mat(self.mat_list_[idx], contourname_list=['Contours'])
            else:
                org_mat = self.mat_list[idx]

            for chosen_ratio in self.resize_ratio:
                res_mat = resize_mat(org_mat, chosen_ratio)
                bbox = get_bbox(res_mat)
                bbox_all.extend(bbox)
        return np.asarray(bbox_all)


    def overlayImgs(self, save_path):
        num_img = len(self.img_list)
        for idx in range(num_img):
            img_path, mat_path = self.img_list_[idx], self.mat_list_[idx]
            _, img_name = os.path.split(img_path)
            org_img, org_mat = self.img_list[idx], self.mat_list[idx]
            bbox    = get_bbox(org_mat)
            overlayed_img = overlay_bbox(org_img, bbox).astype(np.uint8)
            writeImg(overlayed_img, os.path.join(save_path, img_name))

    def next_batch(self):
        batch = {'images': [], 'gt_boxes': [], 'gt_classes': [], 'dontcare': [], 'origin_im': []}
        this_num = min(self.img_num - self.batch_size*self.batch_count, self.batch_size)
        diff = self.batch_size-this_num
        start = self.start - diff

        this_batch_indices = self.indices[start: self.start+this_num]
        targets = self.pool.imap(_get_next,
                     ( (self.img_list[i], self.mat_list[i], self.img_list_[i], self.mat_list_[i],
                        self.resize_ratio,self.img_shape, self.testing) for i in this_batch_indices) )

        #print('len of targets and batch_size: ',start, self.start, this_num, len(this_batch_indices), self.batch_size)
        self.start += this_num
        self.batch_count += 1
        i = 0
        while i < self.batch_size:
            images, gt_boxes, classes, dontcare, origin_im = targets.__next__()
            if len(gt_boxes) > 0:
                batch['images'].append(images)
                batch['gt_boxes'].append(gt_boxes)
                batch['gt_classes'].append(classes)
                batch['dontcare'].append(dontcare)
                batch['origin_im'].append(origin_im)
            i += 1

        img_array = np.stack(batch['images'], 0)
        batch['images'] = img_array * (2. / 255) - 1.

        if self.start >= self.img_num:
            if self._shuffle:
                np.random.shuffle(self.indices)
            self._epoch += 1
            # print(('epoch {} start...'.format(self._epoch)))
            self.start = 0
            self.batch_count = 0
        return batch

    def close(self):
        pass

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes
    @property
    def epoch(self):
        return self._epoch

    @property
    def image_names(self):
        return self.img_list

    @property
    def num_images(self):
        return self.img_num

    @property
    def batch_per_epoch(self):
        return (self.img_num + self.batch_size -1) // self.batch_size



class testingData:
    def __init__(self, data_dir, batch_size, resize_ratio=[0.6], test_mode=True):
        self.__dict__.update(locals())

        all_dict_list  = getfileinfo(self.data_dir, ['_gt'], ['.png'], '.mat', test_mode=True)
        self.img_list_ = [this_dict['thisfile'] for this_dict in all_dict_list]
        self.img_list  = self.img_list_
        self.img_num  = len(all_dict_list)

        self.resize_ratio = resize_ratio
        self.test_mode = test_mode
        self.overlay_bbox = overlay_bbox
        self._classes = ['fake class']
        self._shuffle = False
        self.indices = list(range(self.img_num))

        self._epoch = 0
        self.count = 0
        self.start = 0


    def next(self):
        batch = {'images': [], 'gt_boxes': [], 'gt_classes': [], 'dontcare': [], 'origin_im': []}
        img_path = self.img_list[self.count]
        self.count += 1

        org_img  = imread(img_path)
        chosen_ratio = self.resize_ratio[0]
        res_img = imresize(org_img, chosen_ratio)
        res_img = res_img.transpose(2, 0, 1)
        ret_img =  res_img * (2. / 255) - 1.

        # batch['images'] = ret_img[None]
        batch['images'] = np.expand_dims(ret_img, 0)
        batch['origin_im'].append(org_img.transpose(2, 0, 1))
        batch['dontcare'].append(os.path.basename(img_path))

        return batch

    def close(self):
        pass

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def epoch(self):
        return self._epoch

    @property
    def image_names(self):
        return self.img_list_

    @property
    def num_images(self):
        return self.img_num

    @property
    def batch_per_epoch(self):
        return (self.img_num + self.batch_size -1) // self.batch_size
