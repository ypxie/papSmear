# -*- coding: utf-8 -*-

import os, sys, pdb

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)

from papSmear.darknet import Darknet19
from papSmear.datasets.papsmear import papSmearData as Dataset

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering


if __name__ == '__main__':
    home_dir = os.path.expanduser('~')
    # data_root = os.path.join(home_dir, 'DataSet/PapSmear/CellBoundingBox/TrainDataset')
    data_root = os.path.join(home_dir, 'DataSet/PapSmear/CellBoundingBox/ValDataset')
    dataloader = Dataset(data_root, 64, (256,256))

    all_bbox = dataloader.get_all_bbox()

    col_size_list = all_bbox[:,2] - all_bbox[:,0]
    row_size_list = all_bbox[:,3] - all_bbox[:,1]

    pdb.set_trace()

    size_array = np.stack([col_size_list,row_size_list], 1)

    kmeans = KMeans(n_clusters= 9, random_state=0).fit(size_array)
    centers = kmeans.cluster_centers_
    print(centers)
