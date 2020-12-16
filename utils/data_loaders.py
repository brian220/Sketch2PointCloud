# -*- coding: utf-8 -*-
#
# Developed by Chaoyu Huang 
# email: b608390.cs04@nctu.edu.tw
# Lot's of codes are borrowed from Pix2Vox: 
# https://github.com/hzxie/Pix2Vox

import cv2
import json
import numpy as np
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset

from datetime import datetime as dt
from enum import Enum, unique

from pyntcloud import PyntCloud


@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


# //////////////////////////////// = End of DatasetType Class Definition = ///////////////////////////////// #


class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, point_cloud, ground_truth_view = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return taxonomy_name, sample_name, rendering_images, point_cloud, ground_truth_view

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        point_cloud_path = self.file_list[idx]['point_cloud']
        views = self.file_list[idx]['views']

        # get data of rendering images (sample 1 image from paths)
        if self.dataset_type == DatasetType.TRAIN:
            rand_id = random.randint(0, len(rendering_image_paths) - 1)
            selected_rendering_image_path = rendering_image_paths[rand_id]
            selected_view = views[rand_id]
        else:
        # test, valid with the first image
            selected_rendering_image_path = rendering_image_paths[1]
            selected_view = views[1]
        
        # read the test, train image
        # print(selected_rendering_image_path)
        rendering_images = []
        rendering_image =  cv2.imread(selected_rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if len(rendering_image.shape) < 3:
            print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                     (dt.now(), image_path))
            sys.exit(2)
        rendering_images.append(rendering_image)
    
        # get data of point cloud
        _, suffix = os.path.splitext(point_cloud_path)

        if suffix == '.ply':
            point_cloud = PyntCloud.from_file(point_cloud_path)
            point_cloud = np.array(point_cloud.points)

        return taxonomy_name, sample_name, np.asarray(rendering_images), point_cloud, np.asarray(selected_view)


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #


class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH
        self.point_cloud_path_template = cfg.DATASETS.SHAPENET.POINT_CLOUD_PATH
        self.view_path_template = cfg.DATASETS.SHAPENET.VIEW_PATH
        self.class_name = cfg.DATASET.CLASS

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())
        
        # Get the class data (in dict) from taxonomy
        self.dataset_class_data_taxonomy = self.dataset_taxonomy[self.class_name]

    def get_dataset(self, dataset_type, transforms=None):
        taxonomy_folder_name = self.dataset_class_data_taxonomy['taxonomy_id']
        print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s]' %
                (dt.now(), self.dataset_class_data_taxonomy['taxonomy_id'], self.class_name))
        
        samples = []
        if dataset_type == DatasetType.TRAIN:
            samples = self.dataset_class_data_taxonomy['train']
        elif dataset_type == DatasetType.TEST:
            samples = self.dataset_class_data_taxonomy['test']
        elif dataset_type == DatasetType.VAL:
            samples = self.dataset_class_data_taxonomy['val']

        files = self.get_files_of_taxonomy(taxonomy_folder_name, samples)

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
        return ShapeNetDataset(dataset_type, files, transforms)
        
    def get_files_of_taxonomy(self, taxonomy_folder_name, samples):
        files_of_taxonomy = []
        for sample_idx, sample_name in enumerate(samples):
            # Get file path of pointcloud
            point_cloud_file_path = self.point_cloud_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(point_cloud_file_path):
                print('[WARN] %s Ignore sample %s/%s since point cloud file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Get file list of rendering images
            img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
            img_folder = os.path.dirname(img_file_path)
            total_views = len(os.listdir(img_folder))
            rendering_image_indexes = range(total_views)
            rendering_images_file_paths = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(img_file_path):
                    continue

                rendering_images_file_paths.append(img_file_path)

            if len(rendering_images_file_paths) == 0:
                print('[WARN] %s Ignore sample %s/%s since image files not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue
             
            # Get views of objects (azimuth, elevation)
            view_path = self.view_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(view_path):
                print('[WARN] %s Ignore sample %s/%s since view file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            lines = []
            with open(view_path) as f:
                lines = f.readlines()
            
            views = []
            for view in lines:
                view = view.split()
                # get first two (azimuth, elevation)
                view = view[:2]
                view = [round(float(item)) if float(item) < 359.5 else 0 for item in view]
                views.append(view)

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'rendering_images': rendering_images_file_paths,
                'point_cloud': point_cloud_file_path,
                'views': views
            })

            # Report the progress of reading dataset
            # if sample_idx % 500 == 499 or sample_idx == n_samples - 1:
            #     print('[INFO] %s Collecting %d of %d' % (dt.now(), sample_idx + 1, n_samples))
            
        return files_of_taxonomy


# /////////////////////////////// = End of ShapeNetDataLoader Class Definition = /////////////////////////////// #


DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader
    # 'Pascal3D': Pascal3dDataLoader, # not implemented
    # 'Pix3D': Pix3dDataLoader # not implemented
} 

