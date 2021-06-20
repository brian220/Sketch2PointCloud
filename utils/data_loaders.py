# -*- coding: utf-8 -*-
#
# Developed by Chaoyu Huang 
# email: b608390.cs04@nctu.edu.tw
# Lot's of code reference to Pix2Vox: 
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


def init_pointcloud_loader(num_points):
    Z = np.random.rand(num_points) + 1.
    h = np.random.uniform(10., 214., size=(num_points,))
    w = np.random.uniform(10., 214., size=(num_points,))
    X = (w - 111.5) / 248. * -Z
    Y = (h - 111.5) / 248. * Z
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')

"""
def init_pointcloud_loader(num_points):
    Y = np.random.rand(num_points) + 2.
    h = np.random.uniform(10., 214., size=(num_points,))
    w = np.random.uniform(10., 214., size=(num_points,))
    Z = (w - 111.5) / 420. * -Y
    X = (h - 111.5) / 420. * Y
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')
"""


def sample_spherical(n_points):
    vec = np.random.rand(n_points, 3) * 2. - 1.
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    pc = vec * .3 + np.array([[6.462339e-04,  9.615256e-04, -7.909229e-01]])
    return pc.astype('float32')


class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, cfg, dataset_type, file_list, init_num_points, transforms=None):
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.init_num_points = init_num_points
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, \
        model_azi, model_ele, \
        init_point_cloud, ground_truth_point_cloud = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return (taxonomy_name, sample_name, rendering_images,
                model_azi, model_ele,
                init_point_cloud, ground_truth_point_cloud)

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']

        rendering_image_paths = self.file_list[idx]['rendering_images']
        
        rec_radian_azi = self.file_list[idx]['rec_radian_azi']
        rec_radian_ele = self.file_list[idx]['rec_radian_ele']

        ground_truth_point_cloud_path = self.file_list[idx]['point_cloud']
        
        rec_id = 0
        # get data of rendering images (sample 1 image from paths)
        if self.dataset_type == DatasetType.TRAIN:
            rand_id = random.randint(0, len(rendering_image_paths) - 1)
            selected_rendering_image_path = rendering_image_paths[rand_id]
            # update_id is equal to image_id in single-view model
            rec_id = rand_id 
        else:
        # test, valid with the first image
            selected_rendering_image_path = rendering_image_paths[1]
            rec_id = 1

        # read the test, train image
        rendering_images = []
        rendering_image = cv2.imread(selected_rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if self.cfg.DATASET.TRAIN_DATASET == 'ShapeNetSketch':
            rendering_image = cv2.cvtColor(rendering_image, cv2.COLOR_GRAY2RGB)

        if len(rendering_image.shape) < 3:
            print('[FATAL] %s It seems that there is something wrong with the rendering image file %s' %
                     (dt.now(), selected_rendering_image_path))
            sys.exit(2)
        rendering_images.append(rendering_image)
        
        # get model_azi, model_ele
        model_azi = rec_radian_azi[rec_id]
        model_ele = rec_radian_ele[rec_id]

        # get data of point cloud
        _, suffix = os.path.splitext(ground_truth_point_cloud_path)

        if suffix == '.ply':
            ground_truth_point_cloud = PyntCloud.from_file(ground_truth_point_cloud_path)
            ground_truth_point_cloud = np.array(ground_truth_point_cloud.points).astype(np.float32)
        
        # convert to np array
        rendering_images = np.array(rendering_images).astype(np.float32)
        model_azi = np.array(model_azi).astype(np.float32)
        model_ele = np.array(model_ele).astype(np.float32)
        
        return (taxonomy_name, sample_name, rendering_images,
                model_azi, model_ele,
                init_pointcloud_loader(self.init_num_points), ground_truth_point_cloud)
# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #

class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_taxonomy = None

        # rec
        self.render_views = cfg.DATASET.RENDER_VIEWS
        self.rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH
        self.rec_view_path_template = cfg.DATASETS.SHAPENET.VIEW_PATH
        self.point_cloud_path_template = cfg.DATASETS.SHAPENET.POINT_CLOUD_PATH
        
        self.class_name = cfg.DATASET.CLASS
        self.init_num_points = cfg.GRAPHX.NUM_INIT_POINTS
        
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
        return ShapeNetDataset(self.cfg, dataset_type, files, self.init_num_points, transforms)
        
    def get_files_of_taxonomy(self, taxonomy_folder_name, samples):
        files_of_taxonomy = []
        for sample_idx, sample_name in enumerate(samples):
            # check samples
            sample_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
            sample_folder = os.path.dirname(sample_path)
            if not os.path.exists(sample_folder):
                print('[WARN] %s Ignore sample %s/%s since sample file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Get file paths of pointcloud
            point_cloud_file_path = self.point_cloud_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(point_cloud_file_path):
                print('[WARN] %s Ignore sample %s/%s since point cloud file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Get file paths of rendering images
            rendering_image_indexes = range(self.render_views)
            rendering_image_file_paths = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(img_file_path):
                    continue
                rendering_image_file_paths.append(img_file_path)
            
            if len(rendering_image_file_paths) == 0:
                print('[WARN] %s Ignore sample %s/%s since rendering image files not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Get views for rec model (stage 1) (azimuth, elevation)
            rec_view_path = self.rec_view_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(rec_view_path):
                print('[WARN] %s Ignore sample %s/%s since view file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            rec_angles = []
            with open(rec_view_path) as f:
                rec_angles = [item.split('\n')[0] for item in f.readlines()]
            
            rec_radian_azi = [] # azi
            rec_radian_ele = [] # ele
            for rec_angle in rec_angles:
                rec_angle_azi = float(rec_angle.split(' ')[0])
                rec_angle_ele = float(rec_angle.split(' ')[1])
                # convert angles to radians
                rec_radian_azi.append(rec_angle_azi*np.pi/180.)
                rec_radian_ele.append((rec_angle_ele - 90.)*np.pi/180.)

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'rendering_images': rendering_image_file_paths,
                'rec_radian_azi' : rec_radian_azi,
                'rec_radian_ele' : rec_radian_ele,
                'point_cloud': point_cloud_file_path,
            })

        return files_of_taxonomy


# /////////////////////////////// = End of ShapeNetDataLoader Class Definition = /////////////////////////////// #

class ShapeNetRefineDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetRefineDataset class used for PyTorch DataLoader"""
    def __init__(self, cfg, dataset_type, file_list, init_num_points, transforms=None):
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.init_num_points = init_num_points
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, update_images, \
        model_azi, model_ele, \
        init_point_cloud, ground_truth_point_cloud = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)
            update_images = self.transforms(update_images)

        return (taxonomy_name, sample_name, rendering_images, update_images,
                model_azi, model_ele,
                init_point_cloud, ground_truth_point_cloud)

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']

        rendering_image_paths = self.file_list[idx]['rendering_images']
        update_image_paths = self.file_list[idx]['update_images']
        
        rec_radian_azi = self.file_list[idx]['rec_radian_azi']
        rec_radian_ele = self.file_list[idx]['rec_radian_ele']

        ground_truth_point_cloud_path = self.file_list[idx]['point_cloud']
        
        rec_id = 0
        # get data of rendering images (sample 1 image from paths)
        if self.dataset_type == DatasetType.TRAIN:
            rand_id = random.randint(0, len(rendering_image_paths) - 1)
            selected_rendering_image_path = rendering_image_paths[rand_id]
            selected_update_image_path = update_image_paths[rand_id]
            # update_id is equal to image_id in single-view model
            rec_id = rand_id 
        else:
        # test, valid with the first image
            selected_rendering_image_path = rendering_image_paths[1]
            selected_update_image_path = update_image_paths[1]
            rec_id = 1

        # read the test, train image
        rendering_images = []
        rendering_image = cv2.imread(selected_rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if self.cfg.DATASET.TRAIN_DATASET == 'ShapeNetSketch':
            rendering_image = cv2.cvtColor(rendering_image, cv2.COLOR_GRAY2RGB)

        if len(rendering_image.shape) < 3:
            print('[FATAL] %s It seems that there is something wrong with the rendering image file %s' %
                     (dt.now(), selected_rendering_image_path))
            sys.exit(2)
        rendering_images.append(rendering_image)

        # read the refine image
        update_images = []
        update_image = cv2.imread(selected_update_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if len(update_image.shape) < 3:
            print('[FATAL] %s It seems that there is something wrong with the update image file %s' %
                     (dt.now(), selected_update_image_path))
            sys.exit(2)
        update_images.append(update_image)

        # get model_azi, model_ele
        model_azi = rec_radian_azi[rec_id]
        model_ele = rec_radian_ele[rec_id]

        # get data of point cloud
        _, suffix = os.path.splitext(ground_truth_point_cloud_path)

        if suffix == '.ply':
            ground_truth_point_cloud = PyntCloud.from_file(ground_truth_point_cloud_path)
            ground_truth_point_cloud = np.array(ground_truth_point_cloud.points).astype(np.float32)
        
        # convert to np array
        rendering_images = np.array(rendering_images).astype(np.float32)
        model_azi = np.array(model_azi).astype(np.float32)
        model_ele = np.array(model_ele).astype(np.float32)
        
        return (taxonomy_name, sample_name, rendering_images, update_images,
                model_azi, model_ele,
                init_pointcloud_loader(self.init_num_points), ground_truth_point_cloud)

# //////////////////////////////// = End of ShapeNetRefineDataset Class Definition = ///////////////////////////////// #

class ShapeNetRefineDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_taxonomy = None

        # rec
        self.render_views = cfg.DATASET.RENDER_VIEWS
        self.rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH
        self.update_image_path_template = cfg.DATASETS.SHAPENET.UPDATE_PATH
        self.rec_view_path_template = cfg.DATASETS.SHAPENET.VIEW_PATH
        self.point_cloud_path_template = cfg.DATASETS.SHAPENET.POINT_CLOUD_PATH
        
        self.class_name = cfg.DATASET.CLASS
        self.init_num_points = cfg.GRAPHX.NUM_INIT_POINTS
        
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
        return ShapeNetRefineDataset(self.cfg, dataset_type, files, self.init_num_points, transforms)
        
    def get_files_of_taxonomy(self, taxonomy_folder_name, samples):
        files_of_taxonomy = []
        for sample_idx, sample_name in enumerate(samples):
            # check samples
            sample_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
            sample_folder = os.path.dirname(sample_path)
            if not os.path.exists(sample_folder):
                print('[WARN] %s Ignore sample %s/%s since sample file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Get file paths of pointcloud
            point_cloud_file_path = self.point_cloud_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(point_cloud_file_path):
                print('[WARN] %s Ignore sample %s/%s since point cloud file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Get file paths of rendering images
            rendering_image_indexes = range(self.render_views)
            rendering_image_file_paths = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(img_file_path):
                    continue
                rendering_image_file_paths.append(img_file_path)
            
            if len(rendering_image_file_paths) == 0:
                print('[WARN] %s Ignore sample %s/%s since rendering image files not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Get file paths of update images
            update_image_indexes = range(self.render_views)
            update_image_file_paths = []
            for image_idx in update_image_indexes:
                image_file_path = self.update_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(image_file_path):
                    continue
                update_image_file_paths.append(image_file_path)
            
            if len(update_image_file_paths) == 0:
                print('[WARN] %s Ignore sample %s/%s since update image files not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue
            
            # Get views for rec model (stage 1) (azimuth, elevation)
            rec_view_path = self.rec_view_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(rec_view_path):
                print('[WARN] %s Ignore sample %s/%s since view file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            rec_angles = []
            with open(rec_view_path) as f:
                rec_angles = [item.split('\n')[0] for item in f.readlines()]
            
            rec_radian_azi = [] # azi
            rec_radian_ele = [] # ele
            for rec_angle in rec_angles:
                rec_angle_azi = float(rec_angle.split(' ')[0])
                rec_angle_ele = float(rec_angle.split(' ')[1])
                # convert angles to radians
                rec_radian_azi.append(rec_angle_azi*np.pi/180.)
                rec_radian_ele.append((rec_angle_ele - 90.)*np.pi/180.)

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name'   : taxonomy_folder_name,
                'sample_name'     : sample_name,
                'rendering_images': rendering_image_file_paths,
                'update_images'   : update_image_file_paths,
                'rec_radian_azi'  : rec_radian_azi,
                'rec_radian_ele'  : rec_radian_ele,
                'point_cloud'     : point_cloud_file_path,
            })

        return files_of_taxonomy


DATASET_LOADER_MAPPING = {
    'ShapeNetSketch'              : ShapeNetDataLoader,
    'ShapeNetFix'                 : ShapeNetDataLoader,
    'ShapeNetColor'               : ShapeNetDataLoader,
    'ShapeNetEmptyAnnSketch'      : ShapeNetDataLoader,
    'ShapeNetEmptyBorderAnnSketch': ShapeNetDataLoader,
    'ShapeNetFixRefine'           : ShapeNetRefineDataLoader,
    # 'Pascal3D': Pascal3dDataLoader, # not implemented
    # 'Pix3D': Pix3dDataLoader # not implemented
} 

