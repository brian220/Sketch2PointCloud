'''
The manager can solve the selection operations:
When user click on:

    1. The base_point_cloud
    
    2. Point clouds in a work plane
    
The point clouds will be selected.
'''

import numpy as np
from sketch_3d_ui.manager.geometry_manager import GeometryManager as GM
from sketch_3d_ui.counter import COUNTER

class PointCloudSelectManager(GM):
    def __init__(self):
        self.select_base_model = False
        self.select_work_plane_id = None
        self.select_line_id = None
    
    def init_manager(self):
        COUNTER.count_point_cloud_selection += 1

        self.init_state()
        self.reset_point_cloud_color(reset_base=True)

        self.select_base_model = False
        self.select_work_plane_id = None
        self.select_line_id = None
        
        GM.select_point_cloud = False
        GM.current_point_cloud_select_mode = 'click'

        GM.current_point_cloud_data['select_base_model'] = self.select_base_model
        GM.current_point_cloud_data['work_plane_id'] = self.select_work_plane_id
        GM.current_point_cloud_data['line_id'] = self.select_line_id

        GM.current_point_cloud_comp_data = []
        
    def mouse_pressed(self):
        self.select_point_cloud_by_click()
    
    def select_point_cloud_by_click(self):
        hit_work_plane, work_plane_id, line_id = \
        self.mouse_ray_and_work_plane_point_cloud_hit_detection(mouse_x=self.mouse_x, mouse_y=self.mouse_y,
                                                                work_planes=self.work_planes)
        
        if hit_work_plane:
            self.reset_point_cloud_color(reset_base=True)
                
            self.select_base_model = False
            self.select_work_plane_id = work_plane_id
            self.select_line_id = line_id

            GM.select_point_cloud = True
            GM.work_planes[work_plane_id].generate_point_clouds[line_id].set_color([0.0, 0.0, 1.0])

        hit_base, _ = self.mouse_ray_and_point_cloud_hit_detection(mouse_x=self.mouse_x, mouse_y=self.mouse_y,
                                                                   point_cloud=self.base_point_cloud.positions)
        
        if (not hit_work_plane) and hit_base:
            self.reset_point_cloud_color(reset_base=False)

            self.select_base_model = True
            self.select_work_plane_id = None
            self.select_line_id = None

            GM.select_point_cloud = True
            GM.base_point_cloud.set_color([0.0, 0.0, 1.0])

        GM.current_point_cloud_data['select_base_model'] = self.select_base_model
        GM.current_point_cloud_data['work_plane_id'] = self.select_work_plane_id
        GM.current_point_cloud_data['line_id'] = self.select_line_id

        