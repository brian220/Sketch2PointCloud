# update all the position of control points
# store the mouse movement for current deformation
# store the lattice info of current deformation
import numpy as np
import sketch_3d_ui.geometry.geometry_utils as geometry_utils

from sketch_3d_ui.geometry.transform import Transform

class LATTICE:
    def __init__(self):
        # current select control point
        self.current_ids = []
        self.selected_center = None

        # original points
        self.original_point_cloud = []
        
        # control points
        self.n_control_points = [2, 3, 2]
        self.control_point_positions = []

        # transform for current select control points
        self.transform = None
        
        # bbox
        self.box_length = np.array([1., 1., 1.])
        self.box_origin = np.array([0., 0., 0.])
        self.rot_angle = np.array([0., 0., 0.])

        # movement of control points
        self.array_mu_x = np.zeros(self.n_control_points)
        self.array_mu_y = np.zeros(self.n_control_points)
        self.array_mu_z = np.zeros(self.n_control_points)

    def get_selected_center(self):
        return self.selected_center

    def get_transform(self):
        return self.transform

    def set_current_ids(self, ids):
        self.current_ids = ids

    def set_n_control_points(self, n_control_points):
        self.n_control_points = n_control_points

    def set_control_points(self, control_point_positions):
        self.control_point_positions = control_point_positions

    def set_selected_center(self):
        self.selected_center = \
            np.mean(np.array([self.control_point_positions[i] for i in self.current_ids]), axis=0)
    
    def set_transforms(self):
        self.transform.set_center(self.selected_center)
        self.transform.set_end_points()

    def init_from_bbox(self, box_points, original_point_cloud):
        # origin of box
        self.box_origin = box_points[0]
        self.box_x_axis = geometry_utils.normalized_vector(box_points[1] - box_points[0])
        self.box_y_axis = geometry_utils.normalized_vector(box_points[2] - box_points[0])
        self.box_z_axis = geometry_utils.normalized_vector(box_points[3] - box_points[0])

        # length of bbox
        box_length_x = geometry_utils.two_points_distances(box_points[0], box_points[1])
        box_length_y = geometry_utils.two_points_distances(box_points[0], box_points[2])
        box_length_z = geometry_utils.two_points_distances(box_points[0], box_points[3])
        self.box_length = np.array([box_length_x, box_length_y, box_length_z])
        
        # rotate angles
        self.rot_angle = np.array([0., 0., 0.])

        # movement of control points
        self.array_mu_x = np.zeros(self.n_control_points)
        self.array_mu_y = np.zeros(self.n_control_points)
        self.array_mu_z = np.zeros(self.n_control_points)

        self.original_point_cloud = original_point_cloud
    
    def init_transform(self):
        transform = Transform()
        transform.set_center(self.selected_center)
        transform.set_vector(vector_x=np.array([1., 0., 0.]), 
                             vector_y=np.array([0., 1., 0.]),
                             vector_z=np.array([0., 0., 1.]))
        transform.set_end_points()
        
        self.transform = transform
    
    def map_id_3d_to_id(self, id_3d):
        return id_3d[2] + \
               id_3d[1]*self.n_control_points[2] + \
               id_3d[0]*self.n_control_points[1]*self.n_control_points[2]

    def map_id_1d_to_3d(self, id_1d):
        id_3d = [0, 0, 0]

        for i in range(len(self.n_control_points)):
            id_3d[-1-i] = int(id_1d) % int(self.n_control_points[-1-i])
            if i == len(self.n_control_points):
                break

            if id_1d != 0:
                id_1d /= int(self.n_control_points[-1-i])
        
        return tuple(id_3d)

    def update_move(self, vector, axis):
        for id in self.current_ids:
            control_point_id_3d = self.map_id_1d_to_3d(id)
    
            # check for x
            if axis ==  'X':
                length = vector[0]
                self.array_mu_x[control_point_id_3d] += length/self.box_length[0]
            
            # check for y
            elif axis == 'Y':
                length = vector[1]
                self.array_mu_y[control_point_id_3d] += length/self.box_length[1]
            
            # check for z
            elif axis == 'Z':
                length = vector[2]
                self.array_mu_z[control_point_id_3d] += length/self.box_length[2]
    
            else:
                pass

    def form_connection(self):
        self.connection_list = []
        for i, pos in enumerate(self.control_point_positions):
            id_3d = self.map_id_1d_to_3d(i)
            
            x_neighbor = [id_3d[0] + 1, id_3d[1], id_3d[2]]
            if x_neighbor[0] < self.n_control_points[0]:
                id_1d = self.map_id_3d_to_id(x_neighbor)
                self.connection_list.append([i, id_1d])

            y_neighbor = [id_3d[0], id_3d[1] + 1, id_3d[2]]
            if y_neighbor[1] < self.n_control_points[1]:
                id_1d = self.map_id_3d_to_id(y_neighbor)
                self.connection_list.append([i, id_1d])
            
            z_neighbor = [id_3d[0], id_3d[1], id_3d[2] + 1]
            if z_neighbor[2] < self.n_control_points[2]:
                id_1d = self.map_id_3d_to_id(z_neighbor)
                self.connection_list.append([i, id_1d])

        
