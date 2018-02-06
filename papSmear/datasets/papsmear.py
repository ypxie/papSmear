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
# from functools import partial

#from torch.multiprocessing import Pool
from multiprocessing import Pool
#from .voc_eval import voc_eval
# from utils.yolo import preprocess_train
debug_mode = False
fill_val = np.pi * 1e-8

def crop_svs(slide_path, location=[0,0], level=0, size=None):
    #size (col, row) wise, same for location
    # SlideWidth, SlideHeight = slide_img.level_dimensions[0]
    slide_img  = openslide.open_slide(slide_path)
    SlideWidth,  SlideHeight = slide_img.level_dimensions[0]
    size = [SlideWidth, SlideHeight] if size is None else size
    # read_region(location, level, size)
    # 	location (tuple) – (x, y) tuple giving the top left pixel in the level 0 reference frame
    # 	level (int) – the level number
    # 	size (tuple) – (width, height) tuple giving the region size
    cur_patch = slide_img.read_region(location, level, size)

    return cur_patch


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

def safe_boarder(xcontour, ycontour, row, col, rs=0, cs=0):
    '''
    board_seed: N*2 represent row and col for 0 and 1 axis.
    '''
    xcontour[xcontour < cs] = cs
    xcontour[xcontour >= col] = col-1

    ycontour[ycontour < rs] = rs
    ycontour[ycontour >= row] = row-1
    
    return xcontour, ycontour

@jit  
def overlay_bbox(img, bbox,linewidth=1):
    for bb in bbox:
        x_min_, y_min_, x_max_, y_max_ = bb
        x_min_, y_min_, x_max_, y_max_ = int(x_min_),int( y_min_), int(x_max_), int(y_max_)
        img[:,:,0] = change_val(img[:,:,0], 255, linewidth, x_min_, y_min_, x_max_, y_max_)
        img[:,:,1] = change_val(img[:,:,1], 0, linewidth,  x_min_, y_min_, x_max_, y_max_)
        img[:,:,2] = change_val(img[:,:,2], 0, linewidth,  x_min_, y_min_, x_max_, y_max_)
    return img

def get_single_mask(thiscontour, mask_size, mask_bbox, left_board_ratio=0.1, right_board_ratio=0.1):
    #row_size, col_size = mask_size
    #boarder_row = int(board_ratio * row_size)
    #boarder_col = int(board_ratio * col_size)
    xmin, ymin, xmax, ymax = mask_bbox

    bbox_row, bbox_col = int(ymax-ymin+1), int(xmax-xmin+1)
    #left_boarder_row, right_boarder_row = int(left_board_ratio * bbox_row),  int(right_board_ratio * bbox_row)
    #left_boarder_col, right_boarder_col = int(left_board_ratio * bbox_col),  int(right_board_ratio * bbox_col)

    xcontour = np.reshape(thiscontour[0,:], (1,-1) ).copy()
    ycontour = np.reshape(thiscontour[1,:], (1,-1) ).copy()

    #x_min, x_max = np.min(xcontour), np.max(xcontour)
    #y_min, y_max = np.min(ycontour), np.max(ycontour)
    #print('1: ', xmin, ymin, xmax, ymax  )
    #print('2: ', x_min, y_min, x_max, y_max  )

    norm_xcont = (xcontour - xmin)
    norm_ycont = (ycontour - ymin)

    #print(bbox_row, bbox_col, (y_max - y_min + 1, x_max-x_min + 1) )
    #print(left_board_ratio, right_board_ratio)
    #print('xcontour: ', xcontour[0,0::5])
    #print('ycontour: ', ycontour[0,0::5])
    #import pdb; pdb.set_trace()
    norm_xcont, norm_ycont = safe_boarder(norm_xcont.copy(), norm_ycont.copy(), bbox_row,  bbox_col)
    
    tempmask = roipoly(bbox_row, bbox_col, norm_xcont, norm_ycont)
    
    large_mask = imresize_shape(tempmask, mask_size)
    large_mask = (large_mask>100).astype(np.float32)
    
    #imshow(large_mask)
    #import pdb; pdb.set_trace()
    return large_mask

def _get_single_mask(thiscontour, mask_size, left_board_ratio=0.1, right_board_ratio=0.1):
    row_size, col_size = mask_size
    #boarder_row = int(board_ratio * row_size)
    #boarder_col = int(board_ratio * col_size)

    left_boarder_row, right_boarder_row = int(left_board_ratio * row_size),  int(right_board_ratio * row_size)
    left_boarder_col, right_boarder_col = int(left_board_ratio * col_size),  int(right_board_ratio * col_size)

    xcontour = np.reshape(thiscontour[0,:], (1,-1) )
    ycontour = np.reshape(thiscontour[1,:], (1,-1) )
    x_min, x_max = np.min(xcontour), np.max(xcontour)
    y_min, y_max = np.min(ycontour), np.max(ycontour)
    
    temprow = y_max - y_min + 1
    tempcol = x_max - x_min + 1
    

    choped_row = row_size -  left_boarder_row - right_boarder_row
    choped_col = col_size -  left_boarder_col - right_boarder_col
    
    x_ratio = float(choped_col )/tempcol
    y_ratio = float(choped_row )/temprow

    norm_xcont = (xcontour - x_min)*x_ratio
    norm_ycont = (ycontour - y_min)*y_ratio

    real_chop_row, real_chop_col = min(row_size, choped_row), min(col_size, choped_col)
    norm_xcont, norm_ycont = safe_boarder(norm_xcont, norm_ycont, real_chop_row, real_chop_col)

    tempmask = roipoly(real_chop_row, real_chop_col, norm_xcont, norm_ycont)
    large_mask = np.zeros((row_size, col_size), dtype=np.float32)
    
    rs, cs = max(0, left_boarder_row), max(0, left_boarder_col)
    large_mask[rs:rs+real_chop_row, cs:cs+real_chop_col ] = tempmask

    return large_mask

def get_anchor(row_size, col_size, img_shape, img, boarder=0):
    dst_row, dst_col = img_shape
    br, bc = int(boarder*row_size), int(boarder*col_size)
    idx = 0
    while idx <= 5:
        upleft_row  = random.randint(br, row_size - dst_row - br)
        upleft_col  = random.randint(bc, col_size - dst_col - bc)
        if img[upleft_row, upleft_col, 0] != fill_val:
            break
    return (upleft_row, upleft_col)
         
def resize_mat(contour_mat, resize_ratio):
    numCell = len(contour_mat)
    res_contour = []
    for icontour in range(0, numCell):
        thiscontour = contour_mat[icontour]
        res_contour.append(thiscontour * resize_ratio) 
    return res_contour

def crop_bbox(bbox_list, contour_mat, r_min, c_min, r_max, c_max, mask_size, get_mask=True, board_ratio=0.1):
    numCell = len(bbox_list)
    new_bbox, mask_bbox_list, mask_list = [], [], []
    for idx in range(0, numCell):
        this_bbox   = bbox_list[idx]
        thiscontour = contour_mat[idx]
        x_min_, y_min_, x_max_, y_max_ = this_bbox

        x_min = x_min_  - c_min
        x_max = x_max_  - c_min
        y_min = y_min_  - r_min
        y_max = y_max_  - r_min
        
        thiscontour = thiscontour - np.array([[c_min],[r_min]]  )

        row_size = r_max - r_min + 1
        col_size = c_max - c_min + 1
        
        old_col_len = x_max_-x_min_ +1
        old_row_len = y_max_-y_min_ +1
            
        if(x_min < 0 or y_min < 0 or x_max >=  row_size or y_max >= col_size):
            x_min = max(x_min, 3)
            x_max = min(x_max, col_size-3)
            y_min = max(y_min, 3)
            y_max = min(y_max, row_size-3)
            new_col_len = x_max - x_min + 1
            #old_col_len = x_max_-x_min_ +1
            new_row_len = y_max - y_min + 1
            #old_row_len = y_max_-y_min_ +1
            
            if (new_row_len > 0.8 *old_row_len ) and (new_col_len > 0.8 *old_col_len ):
                new_bbox.append([x_min, y_min, x_max, y_max] )    
        else:
            new_bbox.append([x_min, y_min, x_max, y_max])

            left_board_ratio, right_board_ratio = np.random.uniform(-0.06, 0.15), np.random.uniform(-0.06, 0.15)

            left_boarder_row, right_boarder_row = int(left_board_ratio * old_row_len),  int(right_board_ratio * old_row_len)
            left_boarder_col, right_boarder_col = int(left_board_ratio * old_col_len),  int(right_board_ratio * old_col_len)

            #boarder_row = int(board_ratio * old_row_len)
            #boarder_col = int(board_ratio * old_col_len)
            
            xmin, ymin = x_min - left_boarder_col,  y_min - left_boarder_row
            xmax, ymax = x_max + right_boarder_col, y_max + right_boarder_row

            #xmin, ymin = x_min - 1,  y_min - 1
            #xmax, ymax = x_max + 1,  y_max + 1

            if not (xmin < 0 or ymin < 0 or xmax >=  row_size or ymax >= col_size):
                #import pdb; pdb.set_trace()
                mask_bbox = [xmin, ymin, xmax, ymax]
                mask_bbox_list.append(mask_bbox)
                if get_mask:
                    this_mask = get_single_mask(thiscontour, mask_size,  mask_bbox,
                                left_board_ratio=left_board_ratio, right_board_ratio=right_board_ratio)
                else:
                    this_mask = None
                mask_list.append(this_mask)
    #import pdb; pdb.set_trace()
    #returned_dict = {"bbox":new_bbox, "mask_list":mask_list, "mask_bbox_list":mask_bbox_list }    
    return new_bbox, mask_list, mask_bbox_list

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
                contour_mat = None
                cnts = loaded_mt[contourname].tolist()
                if len(cnts) > 0:
                    contour_mat = cnts[0]
                break
                if not contour_mat:
                    contour_mat = loaded_mt.values()[0].tolist()[0]
                    print('check the mat keys, we use the first one default key: ' + loaded_mt.keys()[0])
    return contour_mat

def transform_matrix_offset_center(matrix, x, y, nx, ny):
    o_x = float(x) / 2 #+ 0.5
    o_y = float(y) / 2 #+ 0.5

    n_x = float(nx) / 2 #+ 0.5
    n_y = float(ny) / 2 #+ 0.5
    
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -n_x], [0, 1, -n_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform_(x, transform_matrix, offset,  fill_mode='constant', cval=0., output_shape=None):
    #final_affine_matrix = transform_matrix[:2, :2]
    #final_offset   = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x[:,:,cidx], transform_matrix.T, 
                      offset, order=2, mode=fill_mode, 
                      cval=cval, output_shape= output_shape, ) for cidx in range(x.shape[2])]
    x = np.stack(channel_images, axis=2)
    
    return x

def apply_transform(x, transform_matrix,  fill_mode='constant', cval = fill_val, output_shape=None):
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset   = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x[:,:,cidx], final_affine_matrix, 
                      final_offset, order=2, mode=fill_mode, 
                      cval=cval, output_shape= output_shape, ) for cidx in range(x.shape[2])]
    
    x = np.stack(channel_images, axis=2)
    return x

def rotate_pair(inputs):
    org_img, contour_mat, theta, prob = inputs
    img_row, img_col = float(org_img.shape[0]) , float(org_img.shape[1])
    
    if random.random() > prob:
      return org_img, contour_mat

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
    return rot_img, rot_contour

def _get_next(inputs):
    img_data, mat_data, img_path, mat_path, resize_ratio, \
    img_shape, testing, mask_size,  get_mask, board_ratio = inputs
                        
    org_img = imread(img_path).astype(float) if debug_mode else img_data #
    org_mat = load_mat(mat_path, contourname_list=['Contours']) if debug_mode else mat_data #
    # here we apply rotation transformation
    #if random.random() > 0:
    #    import pdb; pdb.set_trace()
    #    theta = np.pi / 180 * np.random.uniform(0, 360)
    #    org_img, org_mat =  rotate_pair(  (org_img, org_mat, theta, 0.99) )
    org_row, org_col, _  = org_img.shape

    try_box = 0
    while try_box <= 5:
        try_count = 0 
        mask_list = []
        dst_row, dst_col = img_shape
        
        while True:
            chosen_ratio = resize_ratio[random.randint(0, len(resize_ratio)-1)]
            if try_count >= 5:
                chosen_ratio = 1.1 * max( float(dst_row)/float(org_row), 
                                          float(dst_col)/float(org_col))
                #print('chosen ratio: ', chosen_ratio, org_img.shape, img_shape, try_count)

            if testing is True:
                dst_row = int(org_row*chosen_ratio)
                dst_col = int(col_size*chosen_ratio)

            if int(org_row*chosen_ratio) >= dst_row and int(org_col*chosen_ratio) >= dst_col:
                res_img = imresize(org_img, chosen_ratio)
                res_mat = resize_mat(org_mat, chosen_ratio)
                row_size, col_size, _ = res_img.shape
                up_r, up_c = get_anchor(row_size, col_size, img_shape, img=res_img, boarder=0)
                #print('suc',up_r, up_c, chosen_ratio, res_img.shape, org_img.shape, img_shape, try_count)
                break
            else:
                #print('fail', chosen_ratio,  org_img.shape, img_shape, try_count)
                try_count += 1
                continue   

            

        bbox =  get_bbox(res_mat)
        r_min, c_min, r_max, c_max = up_r, up_c, up_r+dst_row, up_c+dst_col
        this_patch = res_img[r_min:r_max, c_min:c_max, :]
        
        this_patch = this_patch.transpose(2, 0, 1)
        this_bbox, mask_list, mask_bbox_list = crop_bbox(bbox, res_mat, r_min, c_min, 
                                                        r_max, c_max, mask_size, get_mask, board_ratio=board_ratio)

        num_bbox  = len(this_bbox)
        classes = np.zeros((num_bbox), dtype=np.int32)
        if len(this_bbox) != 0:
            break
        else:
            try_box += 1
    #return (this_patch, this_bbox, classes, img_path, res_img)
    return (this_patch, this_bbox, classes, img_path, res_img, mask_list, mask_bbox_list)

class papSmearData:
    def __init__(self, data_dir, batch_size=16, img_shape=None, processes= 4, 
                 testing= False, resize_ratio=[0.3, 0.5, 0.6], aug_rate = 1):
        self.__dict__.update(locals())
        
        all_dict_list  = getfileinfo(self.data_dir, ['_gt', '_withcontour'], ['.png', '.tif'], '.mat')
        self.img_list_ = [this_dict['thisfile']    for this_dict in all_dict_list] 
        self.mat_list_ = [this_dict['thismatfile'] for this_dict in all_dict_list]
        
        self.img_list  = self.img_list_ if debug_mode else [imread(img_path) for img_path in self.img_list_]
        self.mat_list  = self.mat_list_ if debug_mode else [load_mat(mat_path, contourname_list=['Contours']) for mat_path in self.mat_list_]
        
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
                images, gt_boxes, classes, dontcare, origin_im, mask_list, mask_bbox_list = targets.__next__()
            else:
                ii = this_batch_indices[i]
                images, gt_boxes, classes, dontcare, origin_im, mask_list, mask_bbox_list = _get_next(
                    (self.aug_img_list[ii], self.aug_mat_list[ii], self.img_list_[ii], self.mat_list_[ii],
                    self.resize_ratio,self.img_shape, self.testing, self.mask_size, get_mask, self.board_ratio))
                    
            if gt_boxes is not None and len(gt_boxes) > 0:
                batch['images'].append(images)
                batch['gt_boxes'].append(gt_boxes)
                batch['gt_classes'].append(classes)
                batch['dontcare'].append(dontcare)
                batch['origin_im'].append(origin_im)
                batch['mask_list'].append(mask_list)
                batch['mask_bbox_list'].append(mask_bbox_list)
            i += 1
            
        img_array = np.stack(batch['images'], 0).astype(np.float32)
        batch['images'] = img_array * (2. / 255) - 1.
        
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
                #roates_pool = self.pool.imap(rotate_pair,
                #      ((self.img_list[i], self.mat_list[i], np.pi / 180 * random.choice(rotation_pool), 0) for i in range(num_imgs)) ) 
                roates_pool = self.pool.imap(rotate_pair,
                      ((self.img_list[i], self.mat_list[i], np.pi / 180 * np.random.uniform(0, 360), 0.5) for i in range(num_imgs)) ) 

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
            img_path, mat_path = self.img_list_[idx], self.mat_list_[idx]
            org_mat = load_mat(mat_path, contourname_list=['Contours'])
            for chosen_ratio in self.resize_ratio:
                res_mat = resize_mat(org_mat, chosen_ratio)
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



class testingData:
    def __init__(self, data_dir, batch_size, resize_ratio=[0.5], test_mode=True, read_cvs=False):
        self.__dict__.update(locals())

        all_dict_list  = getfileinfo(self.data_dir, ['_gt'], ['.png', '.tif', '.svs'], '.mat', test_mode=True)
        self.img_list_ = [this_dict['thisfile']    for this_dict in all_dict_list] 
        
        self.img_list  = self.img_list_
        
        self.img_num      = len(all_dict_list)

        self.resize_ratio = resize_ratio
        self.test_mode    = test_mode
        self.overlay_bbox = overlay_bbox
        self._classes = ['fake class']
        self._epoch = 0
        self.count = 0
        self.read_cvs = read_cvs

        self._shuffle = True

        self.indices = list(range(self.img_num))
        self.start = 0
    
    def next(self):
        batch = {'images': [], 'gt_boxes': [], 'gt_classes': [], 'dontcare': [], 'origin_im': []}
        img_path = self.img_list[self.count]
        self.count += 1
        if not self.read_cvs: 
            org_img  = imread(img_path) 
        else:
            org_img  = crop_svs(img_path, location=[0,0], level=0, size=None)

        chosen_ratio = self.resize_ratio[0]  
        res_img = imresize(org_img, chosen_ratio)
        res_img = res_img.transpose(2, 0, 1)
        ret_img =  res_img * (2. / 255) - 1.
        
        batch['images'] = ret_img[None]
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
        return self.img_list

    @property
    def num_images(self):
        return self.img_num

    @property
    def batch_per_epoch(self):
        return (self.img_num + self.batch_size -1) // self.batch_size


