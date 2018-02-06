# -*- coding: utf-8 -*-

import os, sys, pdb
sys.path.insert(0, '..')

import numpy as np
from sklearn.cluster import KMeans
from papSmear.datasets.papsmear import papSmearData as Dataset

home = os.path.expanduser('~')
dropbox = os.path.join(home, 'Dropbox')

if __name__ == '__main__':
    # Load dataset
    if 0: # pap smear
        data_root = "/data/.data1/pingjun/Datasets/PapSmear/data/training/"
        resize_ratio = [0.3, 0.5, 0.6]
    if 1: # breast cancer
        data_root = os.path.join(home, 'DataSet','yoloSeg', 'train','breast')
        resize_ratio = [1, 0.8, 0.6]


    dataloader = Dataset(data_dir=data_root, resize_ratio=resize_ratio)

    # Fetch all bbox dims
    all_bbox = dataloader.get_all_bbox()
    col_size_list = all_bbox[:,2] - all_bbox[:,0]
    row_size_list = all_bbox[:,3] - all_bbox[:,1]
    dim_array = np.stack([col_size_list,row_size_list], axis=1)

    #pdb.set_trace()
    
    # Clustered centered image dims
    kmeans = KMeans(n_clusters=9, random_state=0).fit(dim_array)
    centers = kmeans.cluster_centers_
    print("Clustered centers are:")
    print(centers)
