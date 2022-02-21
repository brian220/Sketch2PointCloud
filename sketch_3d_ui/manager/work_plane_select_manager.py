'''
The manager can solve the work plane selection,
When user clicks on the work plane, user can
select or delete work plane.
'''

import numpy as np
import trimesh
import math

import shapely
from shapely import geometry as shapely_geometry

import sketch_3d_ui.geometry.geometry_utils as geometry_utils
from sketch_3d_ui.geometry.point_cloud import PointCloud
from sketch_3d_ui.geometry.bill_board import BillBoard

from sketch_3d_ui.manager.geometry_manager import GeometryManager as GM

from sketch_3d_ui.counter import COUNTER

class WorkPlaneSelectManager(GM):
    def __init__(self):
        super(WorkPlaneSelectManager, self).__init__()

        self.__select_id = None

        # bill board
        self.confirm_board = BillBoard()
        self.delete_board = BillBoard()
    
    def get_selected_work_plane(self):
        if self.__select_id != None:
            return GM.work_planes[self.__select_id]
        else:
            return None

    def init_manager(self):
        self.init_state()
        self.__select_id = None
        GM.current_id = None
    
    def update_state(self):
        if self.state == 'UN_SELECTED':
            self.state = 'CONFIRM'
        elif self.state == 'CONFIRM':
            if GM.current_id == None:
                self.state = 'UN_SELECTED'
            else:
                self.state = 'SELECTED'
        elif self.state == 'SELECTED':
            self.state = 'UN_SELECTED'
        else:
            pass

    def solve_mouse_press(self):
        if self.state == 'UN_SELECTED':
            self.check_click_all_work_plane()

        elif self.state == 'CONFIRM':
            self.check_click_bill_board()

        elif self.state == 'SELECTED':
            self.check_click_current_work_plane()

        else:
            pass

    def check_click_all_work_plane(self):
        hit, hit_point, hit_id = self.mouse_ray_and_planes_hit_detection(mouse_x=self.mouse_x, mouse_y=self.mouse_y, 
                                                                         planes=GM.work_planes, boundary=True)
        if hit:
            self.__select_id = hit_id
            self.init_bill_board_list(left_top=GM.work_planes[self.__select_id].bounding_rec_3d[3],
                                      bill_boards=['confirm_board', 'delete_board'])
            self.update_state()
    
    def check_click_bill_board(self):
        hit, hit_point, hit_id = self.mouse_ray_and_planes_hit_detection(mouse_x=self.mouse_x, mouse_y=self.mouse_y,
                                                                         planes=[self.confirm_board, self.delete_board], boundary=True)
        if hit:
            if hit_id == 0:
                GM.current_id = self.__select_id
                COUNTER.count_plane_selection += 1
           
            elif hit_id == 1:
                GM.current_id = None
                del GM.work_planes[self.__select_id]
                COUNTER.count_plane_deletion += 1
           
            self.update_state()
    
    def check_click_current_work_plane(self):
        hit, _, _ = self.mouse_ray_and_planes_hit_detection(mouse_x=self.mouse_x, mouse_y=self.mouse_y, 
                                                            planes=[GM.work_planes[GM.current_id]], boundary=True)
        if not hit:
            self.__select_id = None
            GM.current_id = None
            self.update_state()
    
    


