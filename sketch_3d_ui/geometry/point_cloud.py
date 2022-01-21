import numpy as np
from pyntcloud import PyntCloud
from pyntcloud.geometry.models.plane import Plane
from pyntcloud.utils.array import PCA

class PointCloud:
    def __init__(self, positions):
        self.positions = positions
        self.colors = []
        self.near_point_ids = []
        self.attach_near_point_ids = []
        
        self.clicked_id = None
        self.is_clicked = False
    
    def set_color(self, color):
        self.colors = [color for i in range(len(self.positions))]

    def set_color_according_camera_pos(self, camera_pos=[1.5, 1.5, 0.0]):
        light_pos_distance = [np.dot((camera_pos - position), (camera_pos - position)) - 1.0 for position in self.positions ]
        max_d = max(light_pos_distance)
        light_pos_distance_normalise = [(d / max_d) for d in light_pos_distance]
        self.colors = [ [1.0-1.0*d, 1.0-1.0*d, 1.0-1.0*d] for d in light_pos_distance_normalise]

    def get_near_points_from_point_cloud(self, point_pos):
        # find the 10 nearest point according to pos
        point_distances = []
        for pos in self.positions:
            point_distances.append(np.dot((point_pos - pos), (point_pos - pos)))

        sorted_id = np.argpartition(point_distances, 300)
        self.near_point_ids = sorted_id[:300]
        
        # fit a plane for the nearest points
        near_point_cloud = [self.positions[i] for i in self.near_point_ids]
        
        return near_point_cloud

    def fit_plane_according_to_pred_and_camera(self, pred_pos, camera_pos):
        # Find the nearsest point
        # plane normal = nearest_vec x (nearest_vec x camera_vec)

        pred_point_distances = []
        for pos in self.positions:
            pred_point_distances.append(np.dot((pred_pos - pos), (pred_pos - pos)))

        sorted_id = np.argpartition(pred_point_distances, 1)
        self.near_point_ids = sorted_id[:1]

        nearest_pos = self.positions[self.near_point_ids[0]]
        nearest_vec = nearest_pos - pred_pos
        nearest_vec = nearest_vec / np.sqrt(np.sum(nearest_vec**2))

        camera_vec = camera_pos - pred_pos
        camera_vec = camera_vec / np.sqrt(np.sum(camera_vec**2))
        
        nearest_camera_vec = np.cross(nearest_vec, camera_vec)
        nearest_camera_vec = nearest_camera_vec / np.sqrt(np.sum(nearest_camera_vec**2))
        normal = np.cross(nearest_vec, nearest_camera_vec)
        normal = normal / np.sqrt(np.sum(normal**2))
        
        plane = Plane(point=pred_pos, normal=normal)

        return plane

    def find_attach_points_from_lines(self, lines):
        mean_start_point = []
        mean_end_point = []

        for line in lines:
            if len(line) <= 1:
                continue

            start_point = line[0]
            end_point = line[len(line) - 1]

            start_point_distances = []
            for pos in self.positions:
                start_point_distances.append(np.dot((start_point - pos), (start_point - pos)))
            
            sorted_id = np.argpartition(start_point_distances, 10)
            start_point_ids = sorted_id[:10]
            start_points = self.positions[start_point_ids]
            self.start_points = start_points
            mean_start_point = np.mean(start_points, axis=0)

            end_point_distances = []
            for pos in self.positions:
                end_point_distances.append(np.dot((end_point - pos), (end_point - pos)))

            sorted_id = np.argpartition(end_point_distances, 10)
            end_point_ids = sorted_id[:10]
            end_points = self.positions[end_point_ids]
            self.end_points = end_points
            mean_end_point = np.mean(end_points, axis=0)
        
        print('mean start point')
        print(mean_start_point)

        print('mean end point')
        print(mean_end_point)

        return mean_start_point, mean_end_point
