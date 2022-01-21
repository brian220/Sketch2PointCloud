import os, sys
import numpy as np
import math
import trimesh

import shapely
from shapely import geometry as shapely_geometry
shapely.speedups.disable()

import sketch_3d_ui.geometry.geometry_utils as geometry_utils
from sketch_3d_ui.geometry.transform import Transform
from sketch_3d_ui.geometry.plane import Plane
from sketch_3d_ui.geometry.point_cloud import PointCloud

from utils.point_cloud_utils import output_point_cloud_ply

class WorkPlane(Plane):
    def __init__(self):
        super(WorkPlane, self).__init__()
                
        # sketch lines
        self.lines_3d = []
       
        self.generate_meshes = []
        self.generate_point_clouds = []
        self.align_point_clouds = []
        
        # for transform
        self.transform = Transform()

    def get_center(self):
        return (self.bounding_rec_3d[0] + self.bounding_rec_3d[2]) / 2.

    def get_transform(self):
        return self.transform

    def init_transform(self):
        self.transform.set_center(self.get_center())
        self.transform.set_vector_from_bounding_rec(self.bounding_rec_3d)
        self.transform.set_end_points()

    def update_bounding_rec_2d_from_lines(self, current_2d_line):
        line_2d = np.array(current_2d_line)
        line_2d_x = line_2d[:,0]
        line_2d_y = line_2d[:,1]

        # compute bounding rectangle
        min_x = min(line_2d_x)
        max_x = max(line_2d_x)
        min_y = min(line_2d_y)
        max_y = max(line_2d_y)

        self.bounding_rec_2d = [[min_x - 20, min_y - 20], 
                                [min_x - 20, max_y + 20],
                                [max_x + 20, max_y + 20],
                                [max_x + 20, min_y - 20]]

    def sample_point_cloud_from_line(self, width):
        radius = 0.001*float(width)
        points_num = 24
        radian_list = [ (i*(360./24.))*math.pi/180. for i in range(points_num) ]
        point_list = [  shapely_geometry.Point(radius*math.cos(radian), radius*math.sin(radian)) for radian in radian_list ]
        polygon = shapely_geometry.Polygon([[p.x, p.y] for p in point_list])

        current_line_id = len(self.lines_3d) - 1
        line = self.lines_3d[current_line_id]
        
        path = tuple(line)
        generate_mesh = trimesh.creation.sweep_polygon(polygon, path)
        pc, _ = trimesh.sample.sample_surface(generate_mesh, 256)
        
        point_cloud = PointCloud(np.array(pc))
        point_cloud.set_color_according_camera_pos()
        self.generate_point_clouds.append(point_cloud)

    def rotate(self, r_mat):
        self.bounding_rec_3d = \
            geometry_utils.rotate_according_to_origin(self.bounding_rec_3d, 
                                                      self.transform.center, r_mat)

        self.transform.set_vector_from_bounding_rec(self.bounding_rec_3d)
        self.transform.set_end_points()
        
        # update line
        for i in range(len(self.lines_3d)):
            if self.lines_3d[i] != []:
                self.lines_3d[i] = \
                    geometry_utils.rotate_according_to_origin(self.lines_3d[i],
                                                              self.transform.center, r_mat)

        # update point clouds
        for i in range(len(self.generate_point_clouds)):
            if self.generate_point_clouds[i] != []:
                self.generate_point_clouds[i].positions = \
                    geometry_utils.rotate_according_to_origin(self.generate_point_clouds[i].positions, 
                                                              self.transform.center, r_mat)
                self.generate_point_clouds[i].set_color_according_camera_pos()

        # update plane
        self.point=self.transform.center
        self.normal=self.transform.vector_z

    def translate(self, vector):
        self.bounding_rec_3d += vector
        
        self.transform.set_center(self.get_center())
        self.transform.set_vector_from_bounding_rec(self.bounding_rec_3d)
        self.transform.set_end_points()

        # update line
        for i in range(len(self.lines_3d)):
            if self.lines_3d[i] != []:
                self.lines_3d[i] += vector

        # update point clouds
        for i in range(len(self.generate_point_clouds)):
            if self.generate_point_clouds[i] != []:
                self.generate_point_clouds[i].positions += vector
        
       # update plane
        self.point=self.transform.center
        self.normal=self.transform.vector_z
