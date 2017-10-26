import os, sys
sys.path.insert(0, '..')

import argparse
import torch
import numpy as np
from papSmear.darknet import Darknet19
from papSmear.datasets.papsmear import papSmearData as Dataset
from papSmear.cfgs.config_pap import cfg
from papSmear.train_yolo import train_eng
from papSmear.proj_utils.local_utils import mkdirs
from papSmear.utils.cython_bbox import bbox_ious

from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering

proj_root = os.path.join('..')
model_root = os.path.join(proj_root, 'Model')
mkdirs([model_root])

home = os.path.expanduser('~')
dropbox = os.path.join(home, 'Dropbox')
data_root = os.path.join(home, 'DataSet','papSmear')

dataloader = Dataset(data_root, 64, (256,256))
all_bbox = dataloader.get_all_bbox()

col_size_list = all_bbox[:,2] - all_bbox[:,0]
row_size_list = all_bbox[:,3] - all_bbox[:,1]

size_array = np.stack([col_size_list,row_size_list], 1)

kmeans = KMeans(n_clusters= 9, random_state=0).fit(size_array)
centers = kmeans.cluster_centers_
print(centers)
# ious = bbox_ious(all_bbox, all_bbox)
# ious = (ious + ious.transpose())/2
# print(ious)
# affinity = ious #np.exp(ious**2)

# labels = spectral_clustering(ious, n_clusters= 9, eigen_solver='arpack')

# unique_labels = np.unique(labels)
# shape_list = []

# print(labels)

# for this_label in unique_labels:
#     this_center  = np.mean(size_array[labels==this_label], 0 )
#     shape_list.append(this_center)
# print(shape_list)


    