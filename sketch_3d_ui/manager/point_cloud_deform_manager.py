import numpy as np
import open3d

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QColor, QPen

import sketch_3d_ui.geometry.geometry_utils as geometry_utils

from sketch_3d_ui.manager.geometry_manager import GeometryManager as GM
from sketch_3d_ui.geometry.point_cloud import PointCloud

from sketch_3d_ui.ffd.ffd import FFD
from sketch_3d_ui.ffd.lattice import LATTICE

from sketch_3d_ui.counter import COUNTER

class PointCloudDeformManager(GM):
    def __init__(self):
        self.__deform_point_cloud = None
        
        self.canvas = QImage(896, 896, QImage.Format_ARGB32)
        self.canvas.fill(Qt.transparent)

        self.lattice = LATTICE()
        self.ffd = FFD()

        self.last_hit_point = []
        self.hit_axis = None

        self.leave_dragging = False
    
    def get_deform_point_cloud(self):
        return self.__deform_point_cloud

    def init_manager(self):
        COUNTER.count_point_cloud_deformation += 1

        self.init_state()
        self.init_deform()
        self.__deform_point_cloud = self.get_current_point_cloud()

    def init_state(self):
        self.state = 'SELECT_CONTROL_POINTS'
    
    def update_state(self):
        if self.state == 'SELECT_CONTROL_POINTS':
            self.state = 'DRAG_CONTROL_POINTS'

        elif self.state == 'DRAG_CONTROL_POINTS':
            self.state = 'SELECT_CONTROL_POINTS'

        else:
            pass
    
    def solve_mouse_event(self, event):
        if self.state == 'SELECT_CONTROL_POINTS':
            self.select_control_points(event)
        elif self.state == 'DRAG_CONTROL_POINTS':
            self.drag_control_points(event)
        else:
            pass

    def select_control_points(self, event):
        if event == 'press':
            self.record_select_rec_start_pos()

        elif event == 'move':
            self.clear_canvas()
            self.draw_on_canvas()

        elif event == 'release':
            self.clear_canvas()
            self.record_select_rec_end_pos()
            self.check_control_points_in_rec()

            self.select_rec_start_pos = []
            self.select_rec_end_pos = []

        else:
            pass
    
    def drag_control_points(self, event):
        if event == 'press':
            self.leave_dragging = False
            hit, hit_point, hit_axis = self.check_click_control_axis()
            if hit:
                self.last_hit_point = hit_point
                self.hit_axis = hit_axis

        elif event == 'double_click':
            self.lattice.set_current_ids([])
            self.__deform_point_cloud.set_color_according_camera_pos()
            self.set_current_point_cloud(self.__deform_point_cloud)
            self.leave_dragging = True

        elif event == 'move':
            hit, hit_point, hit_axis = self.check_click_control_axis()
            if hit:
                self.lattice.update_move(hit_point - self.last_hit_point, self.hit_axis)  
                self.lattice.set_transforms()
                self.last_hit_point = hit_point
                self.deform()
         
        elif event == 'release':
            if self.leave_dragging:
                self.update_state()
        
        else:
            pass
    
    def init_deform(self):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(self.get_current_point_cloud().positions)
        bbox = pcd.get_axis_aligned_bounding_box()
        box_points = np.array(bbox.get_box_points())
        
        # initialize the lattice according to bbox
        self.lattice.init_from_bbox(box_points, self.get_current_point_cloud().positions)
        self.lattice.set_current_ids([])

        self.ffd.init_parameters(n_control_points=self.lattice.n_control_points,
                                 box_length=self.lattice.box_length,
                                 box_origin=self.lattice.box_origin,
                                 rot_angle=self.lattice.rot_angle,
                                 array_mu_x=self.lattice.array_mu_x, 
                                 array_mu_y=self.lattice.array_mu_y,
                                 array_mu_z=self.lattice.array_mu_z)
        
        self.lattice.set_control_points(self.ffd.control_points(deformed=False))

    def record_select_rec_start_pos(self):
        self.select_rec_start_pos = (self.mouse_x, self.mouse_y)

    def record_select_rec_end_pos(self):
        self.select_rec_end_pos = (self.mouse_x, self.mouse_y)

    def draw_on_canvas(self):
        painter = QPainter(self.canvas)
        painter.setPen(QPen(QColor(Qt.green),
                            5,
                            Qt.SolidLine,
                            Qt.RoundCap,
                            Qt.RoundJoin))
        
        # draw select rectangle
        painter.save()
        
        # find the left top corner
        self.select_min_x = min(self.select_rec_start_pos[0], self.mouse_x)
        self.select_max_x = max(self.select_rec_start_pos[0], self.mouse_x)
        self.select_min_y = min(self.select_rec_start_pos[1], self.mouse_y)
        self.select_max_y = max(self.select_rec_start_pos[1], self.mouse_y)

        painter.drawRect(self.select_min_x,
                         self.select_min_y,
                         geometry_utils.two_points_distances(self.select_min_x, self.select_max_x),
                         geometry_utils.two_points_distances(self.select_min_y, self.select_max_y))

        painter.restore()
        painter.end()

    def clear_canvas(self):
        self.canvas = QImage(896, 896, QImage.Format_ARGB32)
        self.canvas.fill(Qt.transparent)
    
    def check_control_points_in_rec(self):
        select = False
        for i, pos in enumerate(self.lattice.control_point_positions):
            screen_pos = geometry_utils.world_pos_to_screen_pos(worldPos=pos,
                                                                screenWidth=self.current_view_port.screen_width,
                                                                screenHeight=self.current_view_port.screen_height,
                                                                ProjectionMatrix=self.current_view_port.projection_matrix,
	                                                            ViewMatrix=self.current_view_port.model_view_matrix)
            # check if the point is in the boundry
            if np.all(np.logical_and(np.array([screen_pos]) >= np.array((self.select_min_x, self.select_min_y)),
                                     np.array([screen_pos]) <= np.array((self.select_max_x, self.select_max_y))),
                      axis=1):
                select = True
                self.lattice.current_ids.append(i)

        if select:
            self.lattice.set_selected_center()
            self.lattice.init_transform()
            self.update_state()
    
    def check_click_control_axis(self, update_axis=False):
        hit, hit_point, hit_axis = self.ray_translate_hit_detection(self.lattice.get_transform())
        
        return hit, hit_point, hit_axis        

    def deform(self):
        self.ffd.update_move(array_mu_x=self.lattice.array_mu_x, 
                             array_mu_y=self.lattice.array_mu_y,
                             array_mu_z=self.lattice.array_mu_z)
        d_points = self.ffd(self.lattice.original_point_cloud)
        dpc = PointCloud(d_points)
        dpc.set_color([0.0, 1.0, 0.0])
        self.__deform_point_cloud = dpc
        self.lattice.set_control_points(self.ffd.control_points())
        self.lattice.set_selected_center()
    