import numpy as np
import trimesh
import math
from pyntcloud.utils.array import PCA

from configs.config_ui import cfg

import sketch_3d_ui.geometry.geometry_utils as geometry_utils
from sketch_3d_ui.geometry.bill_board import BillBoard
from sketch_3d_ui.geometry.work_plane import WorkPlane
from sketch_3d_ui.manager.geometry_manager import GeometryManager as GM

from sketch_3d_ui.counter import COUNTER

from utils.point_cloud_utils import output_point_cloud_ply

from PyQt5.QtCore import QSize, Qt, QRect, QPoint
from PyQt5.QtGui import QColor, QIcon, QPixmap, QScreen, QPainter, QPen, QImage
from PyQt5.QtWidgets import QOpenGLWidget


class WorkPlaneSketchManager(GM):
    def __init__(self):
        super(WorkPlaneSketchManager, self).__init__()
        
        self.line_mode = 'free'
        self.__candidate_work_plane = None

        # bill board
        self.confirm_board = BillBoard()
        self.candidate_board = BillBoard()

        self.canvas = QImage(896, 896, QImage.Format_ARGB32)
        self.canvas.fill(Qt.transparent)

        self.tmp_canvas = QImage(896, 896, QImage.Format_ARGB32)
        self.tmp_canvas.fill(Qt.transparent)

        self.sketching = False
        self.current_sketch_width = 5
        
        self.current_2d_line = []
    
    def get_candidate_work_plane(self):
        return self.__candidate_work_plane

    def init_manager(self):
        self.init_state()

    def init_state(self):
        if GM.current_id == None:
            self.state = 'UN_SELECTED'
        else:
            self.state = 'SELECTED'
    
    def update_state(self):
        if self.state == 'UN_SELECTED':
            if self.__candidate_work_plane:
                self.state = 'CONFIRM'
            else:
                self.state = 'SELECTED'

        elif self.state == 'CONFIRM':
            self.state = 'SELECTED'

        elif self.state == 'SELECTED':
            self.state = 'UN_SELECTED'

        else:
            pass
    
    def solve_mouse_event(self, event):
        if self.state == 'UN_SELECTED':
            self.solve_unselected(event)
        elif self.state == 'CONFIRM':
            self.solve_confirm(event)
        elif self.state == 'SELECTED':
            self.solve_selected(event)
    
    def solve_unselected(self, event):
        if event == 'press':
            self.create_work_plane_by_near_point_cloud_to_camera()

            self.start_pos = QPoint(self.mouse_x, self.mouse_y)
            self.last_pos = QPoint(self.mouse_x, self.mouse_y)
            self.draw_on_canvas()

        elif event == 'move':
            self.draw_on_canvas()

        elif event == 'release':
            print('current_2d_line')
            print(self.current_2d_line)

            if len(self.current_2d_line) > 1 and GM.current_id != None:
                self.add_current_3d_line(boundary=False)
                self.create_work_plane()
                self.finish_sketch_create_plane()
                self.update_state()

            else:
                # if only mouse only click once, remove the created work plane
                GM.work_planes = GM.work_planes[:-1]
                GM.current_id = None

            self.clear_canvas()
            self.current_2d_line = []
            
        else:
            pass
    
    def solve_confirm(self, event):
        if event == 'press':
            pass
            
        elif event == 'move':
            pass

        elif event == 'release':
            hit = self.check_click_bill_board()
            if hit:
                self.update_state()

        else:
            pass
       
    def solve_selected(self, event):
        if event == 'press':
            # on canvas, for visualize
            self.start_pos = QPoint(self.mouse_x, self.mouse_y)
            self.last_pos = QPoint(self.mouse_x, self.mouse_y)
            self.draw_on_canvas()
            
        elif event == 'move':
            # on canvas, for visualize
            self.draw_on_canvas()

        elif event == 'release':
            print('current_2d_line')
            print(self.current_2d_line)
            
            if len(self.current_2d_line) > 1 and GM.current_id != None:
                self.add_current_3d_line(boundary=True)
                self.finish_sketch_current_plane()
            
            self.clear_canvas()
            self.current_2d_line = []
        
        else:
            pass
    
    def create_work_plane_by_near_point_cloud_to_camera(self):
        # find the 100 nearest point from the camera and fit a work_plane
        near_point_cloud = GM.base_point_cloud.get_near_points_from_point_cloud(self.current_view_port.camera_pos)
        work_plane = WorkPlane()
        work_plane.init_from_point_cloud(near_point_cloud)

        # set the current work work plane to be the newest work_plane
        GM.work_planes.append(work_plane)
        GM.current_id = len(GM.work_planes) - 1
    
    def draw_on_canvas(self):
        current_pos = QPoint(self.mouse_x, self.mouse_y)

        if self.line_mode == 'straight':
            self.canvas = self.tmp_canvas.copy()
        
        painter = QPainter(self.canvas)
        painter.setPen(QPen(QColor(Qt.green),
                            self.current_sketch_width*2,
                            Qt.SolidLine,
                            Qt.RoundCap,
                            Qt.RoundJoin))

        # avoid the duplicate input points, which may happen when using the ipad and apple pencil input.
        if self.line_mode == 'free' and current_pos != self.last_pos:
            painter.drawLine(self.last_pos, current_pos)
            self.current_2d_line.append([self.mouse_x, self.mouse_y])
            
        elif self.line_mode == 'straight' and current_pos != self.start_pos:
            painter.drawLine(self.start_pos, current_pos)
            self.current_2d_line = [[self.start_pos.x(), self.start_pos.y()], 
                                    [current_pos.x(), current_pos.y()]]
        
        else:
            pass

        painter.end()
        self.last_pos = QPoint(self.mouse_x, self.mouse_y)

    def clear_canvas(self):
        self.canvas.fill(Qt.transparent)

    def check_click_bill_board(self):
        hit, hit_point, hit_id = self.mouse_ray_and_planes_hit_detection(mouse_x=self.mouse_x, mouse_y=self.mouse_y,
                                                                         planes=[self.confirm_board, 
                                                                         self.candidate_board])
        if hit:
            # delete candidate when click confirm
            if hit_id == 0:        
                self.__candidate_work_plane = None
           
            # add candidate when click candidate
            elif hit_id == 1:
                GM.work_planes.append(self.__candidate_work_plane)
                self.__candidate_work_plane = None
               
        return hit
    
    def add_current_3d_line(self, boundary=False):
        self.sketching = False
        line_3d = []
        for point_2d in self.current_2d_line:
            hit, hit_point, _ = \
                self.mouse_ray_and_planes_hit_detection(mouse_x=point_2d[0], mouse_y=point_2d[1],
                                                        planes=[GM.work_planes[GM.current_id]], 
                                                        boundary=boundary)
            if hit:
                self.sketching = True
                line_3d.append(hit_point)

        GM.work_planes[GM.current_id].lines_3d.append(line_3d)
    
    def create_work_plane(self):
        COUNTER.count_plane_creation += 1
        
        self.create_current_plane()
        self.create_candidate_plane()

    # For rule 1: near plane
    # adjust the work_plane size according to the sketch
    def create_current_plane(self):    
        GM.work_planes[GM.current_id].update_bounding_rec_2d_from_lines(self.current_2d_line)
        bounding_rec_3d = []
        for point in GM.work_planes[GM.current_id].bounding_rec_2d:
            hit, hit_point, _ = self.mouse_ray_and_planes_hit_detection(mouse_x=point[0], mouse_y=point[1], 
                                                                        planes=[GM.work_planes[GM.current_id]], boundary=False)
            if hit:
                bounding_rec_3d.append(hit_point)

        GM.work_planes[GM.current_id].bounding_rec_3d = bounding_rec_3d
    
    # For rule 2: candidate plane
    # detect if 2 endpoints of the sketch is attached on the point cloud
    # if true, suggest that work_plane too
    def create_candidate_plane(self):
        attach, attach_start_point, attach_end_point = self.check_end_points_attach()
        if attach:
            candidate_center = (attach_start_point + attach_end_point) / 2
            candidate_vector = attach_end_point - attach_start_point
            near_point_cloud = self.base_point_cloud.get_near_points_from_point_cloud(candidate_center)
            self.create_candidate_work_plane(candidate_center, near_point_cloud)
            self.update_candidate_plane(candidate_center, candidate_vector)
            self.init_bill_board_list(left_top=GM.work_planes[GM.current_id].bounding_rec_3d[3],
                                      bill_boards=['confirm_board', 'candidate_board'])
    
    def check_end_points_attach(self):
        attach = False
        attach_start_point = None
        attach_end_point = None

        start_point_2d = self.current_2d_line[0]
        end_point_2d = self.current_2d_line[len(self.current_2d_line) - 1]
       
        hit_start, hit_start_id = self.mouse_ray_and_point_cloud_hit_detection(mouse_x=start_point_2d[0], 
                                                                               mouse_y=start_point_2d[1],
                                                                               point_cloud=self.base_point_cloud.positions)
        hit_end, hit_end_id = self.mouse_ray_and_point_cloud_hit_detection(mouse_x=end_point_2d[0],
                                                                           mouse_y=end_point_2d[1],
                                                                           point_cloud=self.base_point_cloud.positions)
        
        if hit_start and hit_end:
            attach = True
            attach_start_point = self.base_point_cloud.positions[hit_start_id[0]]
            attach_end_point = self.base_point_cloud.positions[hit_end_id[0]]

        return attach, attach_start_point, attach_end_point

    def create_candidate_work_plane(self, candidate_center, near_point_cloud):
        w, v = PCA(near_point_cloud)
        
        # if not span a flat plane
        max_id = 0
        if not (w[0] >= w[2]*2 and w[1] >= w[2]*2 and w[0] <= w[1]*2):
            view_vector = (self.current_view_port.camera_pos - candidate_center)
            unit_view_vector = view_vector / np.linalg.norm(view_vector)
            
            angles = []
            for i in range(3):
                angle = geometry_utils.compute_angle(unit_view_vector, v[:, i])
                angles.append(angle)
            
            max_id = np.argmax(angles)
            candidate_plane_normal = v[:, max_id]
            
            self.__candidate_work_plane = WorkPlane()
            self.__candidate_work_plane.init_from_point_normal(point=candidate_center, 
                                                               normal=candidate_plane_normal)
        
        else:
            self.__candidate_work_plane = WorkPlane()
            self.__candidate_work_plane.init_from_point_cloud(near_point_cloud)
    
    def update_candidate_plane(self, candidate_center, candidate_vector):
        # user draw line which spans the plane
        line = GM.work_planes[GM.current_id].lines_3d[0]
        
        # check if current normal is face to the camera 
        current_normal = GM.work_planes[GM.current_id].normal
        veiw_vector = self.current_view_port.camera_pos - GM.work_planes[GM.current_id].get_center()
        if np.dot(current_normal, veiw_vector) < 0:
            current_normal = -current_normal

        # align bounding rectangle to x,y plane
        z_vector = np.array([0., 0., 1.])
        x_vector = np.array([1., 0., 0.])
        origin = np.array([0., 0., 0.])
        line, rec, r_mat_to_plane, r_mat_to_vector = geometry_utils.align_points_to_plane(line=GM.work_planes[GM.current_id].lines_3d[0],
                                                                                          rec=GM.work_planes[GM.current_id].bounding_rec_3d,
                                                                                          ori_normal=current_normal,
                                                                                          des_normal=z_vector,
                                                                                          align_end_points_vector=x_vector,
                                                                                          align_end_points_center=origin)

        # scale points
        candidate_vector_length = np.sqrt(np.sum(candidate_vector**2))
        line_length = np.sqrt(np.sum((line[len(line) - 1] - line[0])**2))
        x_factor = candidate_vector_length/line_length
        
        line = geometry_utils.scale_points(line, x_factor=x_factor, y_factor=1., z_factor=1.)
        rec = geometry_utils.scale_points(rec, x_factor=x_factor, y_factor=1., z_factor=1.)
        
        line = np.dot(line, np.transpose(np.linalg.inv(r_mat_to_vector)))
        line = np.dot(line, np.transpose(np.linalg.inv(r_mat_to_plane)))

        rec = np.dot(rec, np.transpose(np.linalg.inv(r_mat_to_vector)))
        rec = np.dot(rec, np.transpose(np.linalg.inv(r_mat_to_plane)))

        candidate_normal = self.__candidate_work_plane.normal
        veiw_vector = self.current_view_port.camera_pos - self.__candidate_work_plane.point
        if np.dot(candidate_normal, veiw_vector) < 0:
            candidate_normal = -candidate_normal

        # align bounding rectangle to candidate plane
        line, rec, _, _ = geometry_utils.align_points_to_plane(line=line,
                                                               rec=rec,
                                                               ori_normal=current_normal,
                                                               des_normal=candidate_normal,
                                                               align_end_points_vector=candidate_vector,
                                                               align_end_points_center=candidate_center)
        
        self.__candidate_work_plane.lines_3d.append(line)
        self.__candidate_work_plane.bounding_rec_3d = rec
    
    def finish_sketch_create_plane(self):
        if self.sketching:
            if self.__candidate_work_plane:
                self.__candidate_work_plane.sample_point_cloud_from_line(self.current_sketch_width)
            
            GM.work_planes[GM.current_id].sample_point_cloud_from_line(self.current_sketch_width)
        self.sketching = False

    def finish_sketch_current_plane(self):
        if self.sketching:
            GM.work_planes[GM.current_id].sample_point_cloud_from_line(self.current_sketch_width)
        self.sketching = False
    
    def save_work_plane_generate_point_clouds(self, save_pc_path):
        for work_plane_id, work_plane in enumerate(GM.work_planes):
            for pc_id, pc in enumerate(work_plane.generate_point_clouds):
                output_point_cloud_ply(np.array([pc.positions]), ['part' + str(work_plane_id) + '_' + str(pc_id)], save_pc_path)
    
    def save_merge_point_clouds(self, save_pc_path):
        merge_pcs = list(GM.base_point_cloud.positions)
        for work_plane_id, work_plane in enumerate(GM.work_planes):
            for pc_id, pc in enumerate(work_plane.generate_point_clouds):
                merge_pcs.append(list(pc.positions))
        
        print(np.array(merge_pcs).shape)
        output_point_cloud_ply(np.array([merge_pcs]), ['final_merge'], save_pc_path)
        
                
