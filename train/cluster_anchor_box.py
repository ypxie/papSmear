# -*- coding: utf-8 -*-
import os, sys, pdb
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering

FILE_PATH = os.path.abspath(__file__)
PRJ_PATH = os.path.dirname(os.path.dirname(FILE_PATH))
sys.path.append(PRJ_PATH)
from papSmear.darknet import Darknet19
from papSmear.datasets.papsmear import papSmearData as Dataset

if __name__ == '__main__':
    # Load dataset
    data_root = os.path.join(os.path.expanduser('~'),
                             'DataSet/PapSmear/CellBoundingBox/TrainDataset')
    resize_ratio = [0.3, 0.5, 0.6]
    dataloader = Dataset(data_dir=data_root, resize_ratio=resize_ratio)

    # Fetch all bbox dims
    all_bbox = dataloader.get_all_bbox()
    col_size_list = all_bbox[:,2] - all_bbox[:,0]
    row_size_list = all_bbox[:,3] - all_bbox[:,1]
    dim_array = np.stack([col_size_list,row_size_list], axis=1)

    # Clustered centered image dims
    kmeans = KMeans(n_clusters=9, random_state=0).fit(dim_array)
    centers = kmeans.cluster_centers_
    print("Clustered centers are:")
    print(centers)