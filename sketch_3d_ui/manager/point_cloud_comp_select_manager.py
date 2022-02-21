'''
The manager can solve the component selection operations
When user draw a contour on the screen, the point clouds
inside the contour will be selected.
'''

import numpy as np
import cv2

import sketch_3d_ui.geometry.geometry_utils as geometry_utils
from PyQt5.QtCore import QSize, Qt, QRect, QPoint
from PyQt5.QtGui import QImage, QPainter, QColor, QPen
from sketch_3d_ui.manager.geometry_manager import GeometryManager as GM
from sketch_3d_ui.counter import COUNTER

class PointCloudCompSelectManager(GM):
    def __init__(self):
        self.contour = []

        self.canvas = QImage(896, 896, QImage.Format_ARGB32)
        self.canvas.fill(Qt.transparent)

    def init_manager(self):
        COUNTER.count_point_cloud_selection += 1

        self.reset_point_cloud_color(reset_base=True)
        
        GM.select_point_cloud = False
        GM.current_point_cloud_select_mode = 'comp'
        GM.current_point_cloud_comp_data = []

    def solve_mouse_event(self, event):
        if event == 'press':
            self.last_pos =QPoint(self.mouse_x, self.mouse_y)
            self.sketch_on_canvas()
        elif event == 'move':
            self.sketch_on_canvas()
        elif event == 'release':
            self.check_control_points_in_contour()
            self.clear_canvas()
            self.contour = []
        else:
            pass

    def sketch_on_canvas(self):
        painter = QPainter(self.canvas)
        painter.setPen(QPen(QColor(Qt.green),
                            5,
                            Qt.SolidLine,
                            Qt.RoundCap,
                            Qt.RoundJoin))
        current_pos = QPoint(self.mouse_x, self.mouse_y)
        self.contour.append([self.mouse_x, self.mouse_y])
        
        # draw select rectangle
        painter.save()
        
        painter.drawLine(self.last_pos, current_pos)

        painter.restore()
        painter.end()

        self.last_pos = QPoint(self.mouse_x, self.mouse_y)
    
    def clear_canvas(self):
        self.canvas = QImage(896, 896, QImage.Format_ARGB32)
        self.canvas.fill(Qt.transparent)

    def check_control_points_in_contour(self):
        for i, pos in enumerate(GM.base_point_cloud.positions):
            screen_pos = geometry_utils.world_pos_to_screen_pos(worldPos=pos,
                                                                screenWidth=self.current_view_port.screen_width,
                                                                screenHeight=self.current_view_port.screen_height,
                                                                ProjectionMatrix=self.current_view_port.projection_matrix,
	                                                            ViewMatrix=self.current_view_port.model_view_matrix)
            
            # check if the point is in the contour
            if self.check_contour((int(screen_pos[0]), int(screen_pos[1]))):
               data = {}
               data['type'] = 'base'
               data['id'] = i
               data['work_plane_id'] = None
               data['line_id'] = None
               
               GM.select_point_cloud = True
               GM.current_point_cloud_comp_data.append(data)
               GM.base_point_cloud.colors[i] = [0., 0., 1.]
        
        for work_plane_id, work_plane in enumerate(GM.work_planes):
            for line_id, point_cloud in enumerate(work_plane.generate_point_clouds):
                for i, pos in enumerate(point_cloud.positions):
                    screen_pos = geometry_utils.world_pos_to_screen_pos(worldPos=pos,
                                                                        screenWidth=self.current_view_port.screen_width,
                                                                        screenHeight=self.current_view_port.screen_height,
                                                                        ProjectionMatrix=self.current_view_port.projection_matrix,
	                                                                    ViewMatrix=self.current_view_port.model_view_matrix)
                    
                    # check if the point is in the contour
                    if self.check_contour((int(screen_pos[0]), int(screen_pos[1]))):
                       data = {}
                       data['type'] = 'work_plane'
                       data['id'] = i
                       data['work_plane_id'] = work_plane_id
                       data['line_id'] = line_id
                       
                       GM.select_point_cloud = True
                       GM.current_point_cloud_comp_data.append(data)
                       GM.work_planes[work_plane_id].generate_point_clouds[line_id].colors[i] = [0., 0., 1.]
                       
    def check_contour(self, pos):
        contour_np = np.array(self.contour)
        contour_np = contour_np.reshape((-1,1,2))
        dist = cv2.pointPolygonTest(contour_np, pos, False)
    
        return dist == 1.0
             