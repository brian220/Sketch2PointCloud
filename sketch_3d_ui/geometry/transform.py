import sketch_3d_ui.geometry.geometry_utils as geometry_utils
import numpy as np

class Transform:
    def __init__(self):
        # transform
        self.axis_length = 0.15

        self.center = []
        
        self.vector_x = []
        self.vector_y = []
        self.vector_z = []

        self.start_x = []
        self.end_x = []
        self.start_y = []
        self.end_y = []
        self.start_z = []
        self.end_z = []
    
    def set_center(self, point):
        self.center = point

    def set_vector(self, vector_x, vector_y, vector_z):
        self.vector_x = vector_x
        self.vector_y = vector_y
        self.vector_z = vector_z

    def set_vector_from_bounding_rec(self, bounding_rec_3d):
        self.vector_x = geometry_utils.normalized_vector(bounding_rec_3d[1] - bounding_rec_3d[0])
        self.vector_y = geometry_utils.normalized_vector(bounding_rec_3d[2] - bounding_rec_3d[1])
        self.vector_z = np.cross(self.vector_x, self.vector_y)
        
    def set_end_points(self):
        self.start_x = self.center - self.vector_x*self.axis_length
        self.end_x = self.center + self.vector_x*self.axis_length
        self.start_y = self.center - self.vector_y*self.axis_length
        self.end_y = self.center + self.vector_y*self.axis_length
        self.start_z = self.center - self.vector_z*self.axis_length
        self.end_z = self.center + self.vector_z*self.axis_length

    def get_center(self):
        return self.center

        