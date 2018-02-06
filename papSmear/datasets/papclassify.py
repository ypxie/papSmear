import pickle
import os, sys
import uuid
import cv2, openslide
import xml.etree.ElementTree as ET

import h5py, time, copy

import numpy as np
import scipy.sparse
import scipy.ndimage as ndi

from scipy.io import loadmat
from ..proj_utils.local_utils import *

from .papsmear import *
# from functools import partial

#from torch.multiprocessing import Pool
from multiprocessing import Pool
#from .voc_eval import voc_eval
# from utils.yolo import preprocess_train
debug_mode = False
fill_val = np.pi * 1e-8

binary_map_dict = \
{
 'Normal-Cell':0,
 'Superficial-Cell': 0,
 'Intermediate-Cell': 0,
 'Basal-Cell': 0,
 'Endocervical-Epithelium-Cell': 0,
 'Endometrial-Cell': 0,
 'Repair-Cell': 0,
  
 'Trichomonas': 1,
 'Mycosis:Candida': 1,
 'Virus-Infection': 1,
 'Actinomyces': 1,
 'Clue-Cells': 1,
 'Other-Inflammations': 1,

 'Koilocytes': 1,
 'Binucleated-Cell': 1,
 'ASC-US':1,
 'LSIL': 1,
 'ASC-H': 1,
 'HSIL': 1,
 'SCC': 1,

 'Gland-Abnormal': 1
}

def load_anno(thismatfile, contourname_list=['Contours']):
    # First try load using h5py; then try using scipy.io.loadmat
    #import pdb;  pdb.set_trace()
    try:
        loaded_mt = loadmat(thismatfile)
        for contourname in contourname_list:
            if contourname in loaded_mt.keys():
                contour_mat = None
                cnts = loaded_mt[contourname].tolist()
                if len(cnts) > 0:
                    contour_mat = cnts[0]
                break
                if not contour_mat:
                    contour_mat = loaded_mt.values()[0].tolist()[0]
                    print('check the mat keys, we use the first one default key: ' + loaded_mt.keys()[0])
        label_mat = None
    except:
        mat_file = h5py.File(thismatfile, 'r')
        for contourname in contourname_list:
            if contourname in list(mat_file.keys()):
                contour_mat = [np.transpose(mat_file[element[0]][:]) for element in mat_file[contourname]]
                break

        #labels_element = mat_file['Labels']
        #import pdb;  pdb.set_trace()
        labels = []
        for label_ele in mat_file['Labels']:
            num_array  = mat_file[label_ele[0]][:]
            char_array = ''.join([chr(ii) for ii in num_array])
            print(char_array, thismatfile)
            label_mat  = binary_map_dict[char_array]
            labels.append(label_mat)
        mat_file.close()
    return contour_mat, labels

@jit
def get_bbox(contour_mat):
    numCell = len(contour_mat)
    #print('number of cells: ',numCell)
    #print('contour_mat shape', len(contour_mat), contour_mat[0].shape)
    bbox_list = [] 
    for icontour in range(0, numCell):
        thiscontour = contour_mat[icontour]
        xcontour = np.reshape(thiscontour[0,:], (1,-1) )
        ycontour = np.reshape(thiscontour[1,:], (1,-1) )
         
        x_min, x_max = np.min(xcontour), np.max(xcontour)
        y_min, y_max = np.min(ycontour), np.max(ycontour)
        bbox_list.append([x_min, y_min, x_max, y_max] )
    return bbox_list

def safe_boarder(xcontour, ycontour, row, col):
    '''
    board_seed: N*2 represent row and col for 0 and 1 axis.
    '''
    xcontour[xcontour[:,0] < 0] = 0
    xcontour[xcontour[:,0] >= col] = col-1

    ycontour[ycontour[:,0] < 0] = 0
    ycontour[ycontour[:,0] >= row] = row-1
    
    return xcontour, ycontour

def rotate_triple(inputs):
    org_img, mat_input, theta, prob = inputs
    contour_mat, labels = mat_input
    img_row, img_col = float(org_img.shape[0]) , float(org_img.shape[1])
    
    if random.random() > prob:
      return org_img, (contour_mat, labels)

    theta = theta%(2*np.pi)
    theta = theta if theta <= np.pi else theta-2*np.pi
    rtheta = np.pi + theta if theta < 0 else theta

    if rtheta <= np.pi/2:    
        width, height = img_col, img_row

        new_col = width * np.cos(rtheta) + height * np.sin(rtheta)
        new_row = width * np.sin(rtheta) + height * np.cos(rtheta)
    else:
        #width, height = img_row, img_col
        width, height = img_col, img_row
        ntheta =  np.pi - rtheta
        new_col = width * np.cos(ntheta) + height * np.sin(ntheta)
        new_row = width * np.sin(ntheta) + height * np.cos(ntheta)
    new_row, new_col = int(new_row), int(new_col)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    cont_theta = np.pi -  theta                    
    cont_rotation_matrix = np.array([[np.cos(cont_theta), -np.sin(cont_theta), 0],
                                    [np.sin(cont_theta), np.cos(cont_theta), 0],
                                    [0, 0, 1]
                                    ] )
    #transform_matrix = np.array([[np.cos(theta), -np.sin(theta)],
    #                            [np.sin(theta), np.cos(theta)]])
    #offset= 0.5*np.array([img_row, img_col])- (0.5*np.array([new_row, new_col])).dot(transform_matrix)
    #print( [img_row, img_col], (new_row, new_col))
    #rot_img = apply_transform(org_img, transform_matrix, offset, output_shape = (new_row, new_col))
    transform_matrix = transform_matrix_offset_center(rotation_matrix,  img_row,img_col, new_row,  new_col,)
    rot_img = apply_transform(org_img, transform_matrix, output_shape = (new_row, new_col))
    
    numCell = len(contour_mat)
    rot_contour = []
    for icontour in range(0, numCell):
        #thiscontour = np.transpose(contour_mat[icontour], (1,0)) # Nx2
        thiscontour = contour_mat[icontour] # 2*N
        img_coor   = np.array([[-float(img_col)/2, float(img_row)/2]]).T # 2*1
        new_icoor  = np.array([[-float(new_col)/2, float(new_row)/2]]).T # 2*1

        cont_coor  = img_coor + thiscontour*np.array([[1,-1]]).T
        #thiscontour_aug = np.ones((3, cont_coor.shape[1]))
        #thiscontour_aug[0:2,:] = cont_coor
        thiscontour_aug = cont_rotation_matrix[:2,:2].dot(cont_coor)
        thiscontour = thiscontour_aug[0:2,:]
        # this is sooo fucking wierd!
        thiscontour   = thiscontour*np.array([[-1,1]]).T + (new_icoor*np.array([[-1,1]]).T )
        #thiscontour = thiscontour[np.array([1,0])]
        #thiscontour = np.transpose(thiscontour, (1,0)).astype(int)
        rot_contour.append(thiscontour) 

    #rot_contour = rotate_mat(contour_mat, cont_rotation_matrix)
    #imshow(rot_img)
    return rot_img, (rot_contour, labels)


def get_anchor(bbox, img, boarder=0, shift_ratio = 0.12):
    # shift a random position around the corner of bounding box
    
    cmin, rmin, cmax, rmax = bbox
    row_len = rmax - rmin + 1
    col_len = cmax - cmin + 1
    shift_row = int(row_len * shift_ratio)
    shift_col = int(col_len * shift_ratio)
    
    random_row = np.random.uniform(0.5, 1) * row_len
    random_col = np.random.uniform(0.5, 1) * col_len

    img_row, img_col = img.shape[0:2]
    
    draw_rmin, draw_cmin = int(max(0, rmin-shift_row*0.25)), int(max(0, cmin-shift_col*0.25))
    #draw_rmin, draw_cmin = int(max(0, rmin)), int(max(0, cmin))
    draw_rmax, draw_cmax = int(min(img_row, rmin+shift_row)), int(min(img_col, cmin+shift_col))

    upleft_row  = random.randint(draw_rmin, draw_rmax)
    upleft_col  = random.randint(draw_cmin, draw_cmax)

    downright_row, downright_col = int(min(img_row, upleft_row + random_row)), int(min(img_col, upleft_col + random_col))

    return (upleft_row, upleft_col, downright_row, downright_col)
    
def _get_next(inputs):
    #for each image, we collect each boundingbox

    img_data,  mat_data, img_path, mat_path, resize_ratio, \
    img_shape, testing, mask_size,  get_mask, board_ratio = inputs
                        
    org_img = imread(img_path).astype(float) if debug_mode else img_data #
    org_mat, labels = load_anno(mat_path, contourname_list=['Contours']) if debug_mode else mat_data #
    # here we apply rotation transformation
    #if random.random() > 0:
    #    import pdb; pdb.set_trace()
    #    theta = np.pi / 180 * np.random.uniform(0, 360)
    #    org_img, org_mat =  rotate_pair(  (org_img, org_mat, theta, 0.99) )
    res_img = org_img #imresize(org_img, chosen_ratio)
    res_mat = org_mat #resize_mat(org_mat, chosen_ratio)

    bboxes =  get_bbox(res_mat)
    patch_list = []
    label_list = []
    for this_bbox, this_label in zip(bboxes, labels):
        r_min, c_min, r_max, c_max = get_anchor(this_bbox, img=org_img, boarder=0, shift_ratio = 0.25)

        this_patch = res_img[r_min:r_max, c_min:c_max, :]
        this_patch = imresize_shape(this_patch, img_shape)

        this_patch = this_patch.transpose(2, 0, 1)
        patch_list.append(this_patch)
        label_list.append(this_label)
         
    #return (this_patch, this_bbox, classes, img_path, res_img)
    return (patch_list, label_list, res_img, res_mat)

class papClassify:
    def __init__(self, data_dir, batch_size, img_shape=None, processes= 4, 
                 testing= False, resize_ratio=[0.3, 0.5, 0.6], aug_rate = 1):
        self.__dict__.update(locals())
        
        all_dict_list  = getfileinfo(self.data_dir, ['_gt', '_withcontour'], ['.png', '.tif'], '.mat')
        self.img_list_ = [this_dict['thisfile']    for this_dict in all_dict_list] 
        self.mat_list_ = [this_dict['thismatfile'] for this_dict in all_dict_list]
        
        self.img_list  = self.img_list_ if debug_mode else [imread(img_path) for img_path in self.img_list_]
        self.mat_list  = self.mat_list_ if debug_mode else [load_anno(mat_path, contourname_list=['Contours']) for mat_path in self.mat_list_]
        
        # for idx, this_mat in enumerate(self.mat_list):
        #     if this_mat is None:
        #         print(self.mat_list_[idx])

        self.aug_img_list  = [copy.deepcopy(this_img)   for this_img in self.img_list] 
        self.aug_mat_list  = [copy.deepcopy(this_mat)   for this_mat in self.mat_list]
             
        self.img_num      = len(all_dict_list)
        self.img_shape    = img_shape
        self._img_shape   = img_shape
        self.mask_size    = [64,64]
        self.board_ratio  = 0.1
        self.resize_ratio = resize_ratio
        self.testing = testing
        self.overlay_bbox = overlay_bbox
        self._classes = ['fake class']
        self._epoch = 0
        self.count = 0
        self.batch_count = 0
        self.aug_rate = aug_rate  # how many epochs before you rotate each image
        self._pool_processes = processes
        self.pool = Pool(self._pool_processes)
        self._shuffle = True
        
        self.indices = list(range(self.img_num))
        self.start = 0
        
    def next_batch(self, get_mask=True):
        batch = {'images': [], 'gt_boxes': [], 'gt_classes': [], 'dontcare': [], 
                 'origin_im': [], "mask_list":[], 'mask_bbox_list':[]}
        this_num = min(self.img_num - self.batch_size*self.batch_count, self.batch_size)
        #if this_num < self.batch_size:
        diff = self.batch_size-this_num
        start = self.start - diff

        this_batch_indices = self.indices[start: self.start+this_num]
        if debug_mode is False:
            targets = self.pool.imap(_get_next,
                      ((self.aug_img_list[i], self.aug_mat_list[i], self.img_list_[i], self.mat_list_[i],
                        self.resize_ratio,self.img_shape, self.testing, self.mask_size, get_mask, self.board_ratio) for i in this_batch_indices) )                
        #print('len of targets and batch_size: ',start, self.start, this_num, len(this_batch_indices), self.batch_size)
        self.start += this_num
        self.batch_count += 1
        i = 0
        while i < this_num:
            if debug_mode is False:
                images, labels, origin_im, dontcare = targets.__next__()
            else:
                ii = this_batch_indices[i]
                images, labels, origin_im, dontcare = _get_next(
                    (self.aug_img_list[ii], self.aug_mat_list[ii], self.img_list_[ii], self.mat_list_[ii],
                    self.resize_ratio,self.img_shape, self.testing, self.mask_size, get_mask, self.board_ratio))
                    
            if len(labels) > 0:
                #print(images[0].shape, type(images[0]), len(images))
                batch['images'].extend(images)
                #batch['gt_boxes'].append(gt_boxes)
                batch['gt_classes'].extend(labels)
                batch['dontcare'].append(dontcare)
                batch['origin_im'].append(origin_im)
                #batch['mask_list'].append(mask_list)
                #batch['mask_bbox_list'].append(mask_bbox_list)
            i += 1

        img_array   =  np.stack(batch['images'], axis=0).astype(np.float32)
        gt_classes  =  np.stack(batch['gt_classes'], axis=0).astype(np.int32)
        
        #print(img_array.shape)
        batch['images'] = img_array * (2. / 255) - 1.
        batch['gt_classes'] = gt_classes

        if self.start >= self.img_num:
            if self._shuffle:
                random.shuffle(self.indices)
            self._epoch += 1

            print(('epoch {} start...'.format(self._epoch)))
            self.start = 0
            self.batch_count = 0

            #it's buggy, you cannot just rotate it, the size will roll up.
            if self._epoch!= 0 and (self._epoch % self.aug_rate == 0):
                # rotate all the images at each epoch
                start_timer = time.time()
                rotation_pool = [90, 180, 270]
                num_imgs = len(self.mat_list)
                #roates_pool = self.pool.imap(rotate_imgs,
                #      ((self.img_list[i], self.mat_list[i], np.pi / 180 * random.choice(rotation_pool), 0) for i in range(num_imgs)) ) 
                roates_pool = self.pool.imap(rotate_triple,
                      ((self.img_list[i], self.mat_list[i], np.pi / 180 * np.random.uniform(0, 360), 1) for i in range(num_imgs)) ) 

                for iidx in range(num_imgs ):
                    aug_img, aug_mat = roates_pool.__next__()
                    self.aug_img_list[iidx] = aug_img
                    self.aug_mat_list[iidx] = aug_mat
                print('data augmentation takes time {:.2f}.'.format(time.time()-start_timer))

        return batch

    def overlayImgs(self, save_path):
        num_img = len(self.img_list)
        for idx in range(num_img):
            img_path, mat_path = self.img_list_[idx], self.mat_list_[idx]
            _, img_name = os.path.split(img_path)
            org_mat = load_mat(mat_path, contourname_list=['Contours'])
            org_img = imread(img_path)
            
            if org_mat is None:
                print('None mat: ',mat_path)
            else:
                #org_img, org_mat = self.img_list[idx], self.mat_list[idx]
                #org_img = imread(img_path)
                #org_mat = load_mat(mat_path, contourname_list=['Contours'])
                #imshow(org_img)
                bbox    = get_bbox(org_mat)
                overlayed_img = overlay_bbox(org_img, bbox).astype(np.uint8)    
                writeImg(overlayed_img, os.path.join(save_path, img_name) )  
                
    def get_all_bbox(self):
        bbox_all = []
        for idx in range(self.img_num):
            img_path, mat_path = self.img_list[idx], self.mat_list[idx]
            org_mat = self.load_mat(mat_path, contourname_list=['Contours'])
            for chosen_ratio in self.resize_ratio:
                res_mat = self.resize_mat(org_mat, chosen_ratio)
                bbox =  get_bbox(res_mat)
                bbox_all.extend(bbox)
        return np.asarray(bbox_all)
         
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

    