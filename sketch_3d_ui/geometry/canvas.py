import os, sys
import numpy as np
import math
import trimesh

import shapely
from shapely import geometry as shapely_geometry

import sketch_3d_ui.geometry.geometry_utils as geometry_utils
from sketch_3d_ui.geometry.transform import Transform
from sketch_3d_ui.geometry.plane import Plane
from sketch_3d_ui.geometry.point_cloud import PointCloud

from utils.point_cloud_utils import output_point_cloud_ply

class Canvas(Plane):
    def __init__(self):
        super(Canvas, self).__init__()
                
        # sketch lines
        self.lines_2d = []
        self.lines_2d.append({})
        self.lines_3d = []
        self.lines_3d.append({})

    def get_center(self):
        return (self.bounding_rec_3d[0] + self.bounding_rec_3d[2]) / 2.

    def get_transform(self):
        return self.transform

    def init_transform(self):
        self.transform.set_center(self.get_center())
        self.transform.set_vector_from_bounding_rec(self.bounding_rec_3d)
        self.transform.set_end_points()

    def update_bounding_rec_2d_from_lines(self):
        line_2d = np.array(self.lines_2d[0])
        line_2d_x = line_2d[:,0]
        line_2d_y = line_2d[:,1]

        # compute bounding rectangle
        min_x = min(line_2d_x)
        max_x = max(line_2d_x)
        min_y = min(line_2d_y)
        max_y = max(line_2d_y)

        self.bounding_rec_2d = [[min_x - 5, min_y - 5], 
                                [min_x - 5, max_y + 5],
                                [max_x + 5, max_y + 5],
                                [max_x + 5, min_y - 5]]

    def update_lines(self):
        self.lines_2d.append({})
        self.lines_3d.append({})
    
    def init_line(self, color): 
        current_line_id = len(self.lines_2d) - 1
        self.lines_2d[current_line_id]['color'] = color
        self.lines_3d[current_line_id]['color'] = color
        self.lines_2d[current_line_id]['points'] = []
        self.lines_3d[current_line_id]['points'] = []
        
    def add_point_to_lines(self, point_2d, point_3d):
        current_line_id = len(self.lines_2d) - 1
        if 'points' in self.lines_2d[current_line_id].keys():
            self.lines_2d[current_line_id]['points'].append(point_2d)
            self.lines_3d[current_line_id]['points'].append(point_3d)
    
    def clean_lines(self):
        self.lines = []
        self.lines.append({})
