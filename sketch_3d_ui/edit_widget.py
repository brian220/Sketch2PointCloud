import numpy as np
from PIL import Image
import math
import json
import os, sys
import cv2
from shutil import copyfile
from distutils.dir_util import copy_tree

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from pyntcloud import PyntCloud

# For DL Model
from configs.config_ui import cfg
from core.inference import inference_net
from core.refine import refine_net

# For UI
import sketch_3d_ui.geometry.geometry_utils as geometry_utils
from sketch_3d_ui.geometry.point_cloud import PointCloud

from sketch_3d_ui.base_opengl_widget import BaseOpenGLWidget as BASEOPENGL
from sketch_3d_ui.manager.geometry_manager import GeometryManager as GM
from sketch_3d_ui.manager.canvas_manager import CanvasManager
from sketch_3d_ui.manager.work_plane_select_manager import WorkPlaneSelectManager
from sketch_3d_ui.manager.work_plane_sketch_manager import WorkPlaneSketchManager
from sketch_3d_ui.manager.work_plane_transform_manager import WorkPlaneTransformManager
from sketch_3d_ui.manager.point_cloud_select_manager import PointCloudSelectManager
from sketch_3d_ui.manager.point_cloud_comp_select_manager import PointCloudCompSelectManager
from sketch_3d_ui.manager.point_cloud_deform_manager import PointCloudDeformManager
from sketch_3d_ui.view.camera import Camera_Z_UP

from sketch_3d_ui.counter import COUNTER, reset_counter

from utils.point_cloud_utils import output_point_cloud_ply

class CheckBoxWidget(QWidget):
    def __init__(self, parent=None):
        super(CheckBoxWidget, self).__init__(parent)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.count_work_plane_label = QLabel('Work plane : 0')
        self.count_3d_line_label = QLabel('3D line : 0')

        self.count_plane_selection_label = QLabel('Plane confirmation : 0')
        self.count_plane_deletion_label = QLabel('Plane deletion : 0')
        self.count_plane_creation_label = QLabel('Plane creation : 0')
        self.count_plane_transformation_label = QLabel('Plane transformation : 0')
        
        self.count_point_cloud_selection_label = QLabel('Point cloud selection : 0')
        self.count_point_cloud_deformation_label = QLabel('Point cloud deformation : 0')
        
        layout.addWidget(self.count_work_plane_label)
        layout.addWidget(self.count_3d_line_label)

        layout.addWidget(self.count_plane_selection_label)
        layout.addWidget(self.count_plane_deletion_label)
        layout.addWidget(self.count_plane_creation_label)
        layout.addWidget(self.count_plane_transformation_label)

        layout.addWidget(self.count_point_cloud_selection_label)
        layout.addWidget(self.count_point_cloud_deformation_label)
    
    def update_label(self):
        count_work_plane = len(GM.work_planes)
        count_3d_line = 0
        for work_plane in GM.work_planes:
            count_3d_line += len(work_plane.lines_3d)
        self.count_work_plane_label.setText("Work plane : " + str(count_work_plane))
        self.count_3d_line_label.setText("3D line : " + str(count_3d_line))

        self.count_plane_selection_label.setText("Plane selection : " \
                                                + str(COUNTER.count_plane_selection))
        self.count_plane_deletion_label.setText("Plane deletion : " \
                                                + str(COUNTER.count_plane_deletion))
        self.count_plane_creation_label.setText("Plane creation : " \
                                                + str(COUNTER.count_plane_creation))
        self.count_plane_transformation_label.setText("Plane transformation : " \
                                                      + str(COUNTER.count_plane_transformation))
        
        self.count_point_cloud_selection_label.setText("Point cloud selection : "\
                                                + str(COUNTER.count_point_cloud_selection))
        self.count_point_cloud_deformation_label.setText("Point cloud deformation : " \
                                                + str(COUNTER.count_point_cloud_deformation))
        
class EditWidget(BASEOPENGL):
    def __init__(self, parent=None):
        super(EditWidget, self).__init__(parent)
        
        # show counter
        self.show_counter_widget = CheckBoxWidget(self)
        self.show_counter_widget.setMinimumSize(300, 100)
        self.show_counter_widget.setGeometry(0, 0, 150, 150)

        self.width = 896
        self.height = 896
        self.sub_width = 448
        self.sub_height = 448

        # mode
        self.mode = 'view'

        # eye
        self.azi = 0.
        self.ele = 0.
        
        """
        pc_path = "/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/03001627/27c00ec2b6ec279958e80128fd34c2b1.ply"
        pc = PyntCloud.from_file(pc_path)
        pc_points = np.array(pc.points).astype(np.float32)
        GM.base_point_cloud = PointCloud(pc_points)
        GM.base_point_cloud.set_color_according_camera_pos(camera_pos=[1.5, 1.5, 0.0])
        """

        self.camera = Camera_Z_UP(theta=self.azi*math.pi/180., \
                                  phi= (90. - self.ele)*math.pi/180., \
                                  distance=2.0)
        
        self.mouse_state = None
    
    ########################################################################
    #    Settings and reference models                                     #
    ########################################################################
    def save_result(self, save_path):
        save_img_path = os.path.join(save_path, 'imgs/')
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path) 

        save_pc_path = os.path.join(save_path, 'pcs/')
        if not os.path.exists(save_pc_path):
            os.makedirs(save_pc_path) 

        # copy the imgs from cache
        copy_tree(cfg.INFERENCE.CACHE_IMAGE_PATH, save_img_path)
        # copy the pcs from cache
        copy_tree(cfg.INFERENCE.CACHE_POINT_CLOUD_PATH, save_pc_path)

        # save edited base point clouds
        output_point_cloud_ply(np.array([GM.base_point_cloud.positions]), ['final_base'], save_pc_path)
        # save edited work plane point clouds
        self.work_plane_sketch_manager.save_work_plane_generate_point_clouds(save_pc_path)
        
        # save counter info for the user study
        self.save_counter(save_path)
        reset_counter()

        # reset geometry manager
        self.reset_manager()

        self.update()
    
    def load_point_cloud(self, pc_path):
        # load and merge all .ply file in the directory
        pc_list = []
        for dir in os.listdir(pc_path):
            if dir == 'final_base.ply' or dir[:4] == 'part':
                pc = PyntCloud.from_file(os.path.join(pc_path, dir))
                pc_points = np.array(pc.points).astype(np.float32)
                pc_list.append(pc_points)
       
        merge_pc = np.concatenate(pc_list)
        GM.base_point_cloud = PointCloud(merge_pc)
        GM.base_point_cloud.set_color_according_camera_pos(camera_pos=[1.5, 1.5, 0.0])
        
        self.update()

    def save_counter(self, save_path):
        f = open(os.path.join(save_path, 'counter.txt'), 'w')

        f.write(self.show_counter_widget.count_work_plane_label.text() + '\n')
        f.write(self.show_counter_widget.count_3d_line_label.text() + '\n')
        
        f.write(self.show_counter_widget.count_plane_selection_label.text() + '\n')
        f.write(self.show_counter_widget.count_plane_deletion_label.text() + '\n')
        f.write(self.show_counter_widget.count_plane_creation_label.text() + '\n')
        f.write(self.show_counter_widget.count_plane_transformation_label.text() + '\n')

        f.write(self.show_counter_widget.count_point_cloud_selection_label.text() + '\n')
        f.write(self.show_counter_widget.count_point_cloud_deformation_label.text() + '\n')

        f.close()
      
    def reset_manager(self):
        GM.rendering_mask = []

        # point cloud
        GM.base_point_cloud = None
        GM.reference_point_cloud = None
        GM.current_point_cloud = None
        GM.current_point_cloud_select_mode = None
        GM.current_point_cloud_data = {}
        GM.current_point_cloud_comp_data = {}

        # work plane
        GM.work_planes = []
        GM.current_id = None
        
        # manager
        # sketch
        BASEOPENGL.canvas_manager = CanvasManager()
        # work plane
        BASEOPENGL.work_plane_select_manager = WorkPlaneSelectManager()
        BASEOPENGL.work_plane_sketch_manager = WorkPlaneSketchManager()
        BASEOPENGL.work_plane_transform_manager = WorkPlaneTransformManager()
        # point cloud
        BASEOPENGL.point_cloud_select_manager = PointCloudSelectManager()
        BASEOPENGL.point_cloud_comp_select_manager = PointCloudCompSelectManager()
        BASEOPENGL.point_cloud_deform_manager = PointCloudDeformManager()
    
    def setReferenceModel(self, reference_model_path):
        pc = PyntCloud.from_file(reference_model_path)
        pc_points = np.array(pc.points).astype(np.float32)
        GM.base_point_cloud = PointCloud(pc_points)
        GM.base_point_cloud.set_color_according_camera_pos(camera_pos=[1.5, 1.5, 0.0])
        self.update()
    
    def set_draw_line(self, line_mode):
        self.canvas_manager.line_mode = line_mode
        self.work_plane_sketch_manager.line_mode = line_mode

    def set_lattice_dimension(self, n_control_points):
        BASEOPENGL.point_cloud_deform_manager.lattice.set_n_control_points(n_control_points)
        if self.mode == 'pointCloudDeform':
            BASEOPENGL.point_cloud_deform_manager.init_deform()
            self.update()

    def set_work_plane_sketch_width(self, width):
        self.work_plane_sketch_manager.current_sketch_width = width
        self.changeCurser()

    ########################################################################
    #    Mode                                                              #
    ########################################################################
    def init_mode(self):
        if self.mode == 'inputSketch':
            BASEOPENGL.canvas_manager.init_manager(self.mode)
            self.changeCurser()

        elif self.mode == 'inputRulerSketch':
            BASEOPENGL.canvas_manager.init_manager(self.mode)
            self.changeCurser()
            
        elif self.mode == 'inputErase':
            BASEOPENGL.canvas_manager.init_manager(self.mode)
            self.changeCurser()
            
        elif self.mode == 'inputDetail':
            BASEOPENGL.canvas_manager.init_manager(self.mode)
            self.changeCurser()

        elif self.mode == 'workPlaneSelect':
            BASEOPENGL.work_plane_select_manager.init_manager()
            self.changeCurser()

        elif self.mode == 'workPlaneSketch':
            BASEOPENGL.work_plane_sketch_manager.init_manager()
            self.changeCurser()

        elif self.mode == 'workPlaneTransform':
            BASEOPENGL.work_plane_transform_manager.init_manager()
            self.changeCurser()
            
        elif self.mode == 'pointCloudSelect':
            BASEOPENGL.point_cloud_select_manager.init_manager()
            self.changeCurser()

        elif self.mode == 'pointCloudCompSelect':
            BASEOPENGL.point_cloud_comp_select_manager.init_manager()
            self.changeCurser()

        elif self.mode == 'pointCloudDeform':
            BASEOPENGL.point_cloud_deform_manager.init_manager()
            self.changeCurser()

        else:
            pass

        self.update()
    
    ########################################################################
    #    Mouse Event                                                       #
    ########################################################################
    def mousePressEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            self.mouse_state ='LEFT'
        elif e.buttons() == Qt.MidButton:
            self.mouse_state = 'MID'
        else:
            return

        if self.mode[:5] == 'input':
            return self.generalCanvas_mousePressEvent(e)

        else:
            fn = getattr(self, "%s_mousePressEvent" % self.mode, None)
            if fn:
                return fn(e)

    def mouseMoveEvent(self, e):
        if self.mode[:5] == 'input':
            return self.generalCanvas_mouseMoveEvent(e)

        else:
            fn = getattr(self, "%s_mouseMoveEvent" % self.mode, None)
            if fn:
                return fn(e)

    def mouseReleaseEvent(self, e):
        if self.mode[:5] == 'input':
            return self.generalCanvas_mouseReleaseEvent(e)

        else:
            fn = getattr(self, "%s_mouseReleaseEvent" % self.mode, None)
            if fn:
                return fn(e)

    def mouseDoubleClickEvent(self, e):
        fn = getattr(self, "%s_mouseDoubleClickEvent" % self.mode, None)
        if fn:
            return fn(e)
    
    def view_mousePressEvent(self, event):
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def view_mouseMoveEvent(self, event):
        if self.mouse_state == 'MID':
            self.makeCurrent()
            self.update_camera(mouse_x=event.pos().x(), mouse_y=event.pos().y())

        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def view_mouseReleaseEvent(self, event):
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def generalCanvas_mousePressEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.canvas_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.canvas_manager.solve_mouse_event('press')
            
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
    def generalCanvas_mouseMoveEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.canvas_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.canvas_manager.solve_mouse_event('move')

        elif self.mouse_state == 'MID':
            self.makeCurrent()
            self.update_camera(mouse_x=event.pos().x(), mouse_y=event.pos().y())

        else:
            pass

        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def generalCanvas_mouseReleaseEvent(self, event):
        self.makeCurrent()
        BASEOPENGL.canvas_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
        if self.mouse_state == 'LEFT':
            BASEOPENGL.canvas_manager.solve_mouse_event('release')
            
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
    # select work plane
    def workPlaneSelect_mousePressEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.work_plane_select_manager.set_current_view_port(current_camera_pos=self.camera.get_cartesian_camera_pos(),
                                                                       current_screen_width=self.width,
                                                                       current_screen_height=self.height,
                                                                       current_projection_matrix=self.get_projection_matrix(),
                                                                       current_model_view_matrix=self.get_model_view_matrix())  
            BASEOPENGL.work_plane_select_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.work_plane_select_manager.solve_mouse_press()
            
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
    def workPlaneSelect_mouseMoveEvent(self, event):
        if BASEOPENGL.work_plane_select_manager.state != 'CONFIRM' and self.mouse_state == 'MID':
            self.update_camera(mouse_x=event.pos().x(), mouse_y=event.pos().y())
        
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def workPlaneSelect_mouseReleaseEvent(self, event):
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
    # sketch work plane
    def workPlaneSketch_mousePressEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.work_plane_sketch_manager.set_current_view_port(current_camera_pos=self.camera.get_cartesian_camera_pos(),
                                                                       current_screen_width=self.width,
                                                                       current_screen_height=self.height,
                                                                       current_projection_matrix=self.get_projection_matrix(),
                                                                       current_model_view_matrix=self.get_model_view_matrix())
            BASEOPENGL.work_plane_sketch_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.work_plane_sketch_manager.solve_mouse_event('press')

        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
    def workPlaneSketch_mouseMoveEvent(self, event):
        if self.mouse_state == 'LEFT':
            BASEOPENGL.work_plane_sketch_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.work_plane_sketch_manager.solve_mouse_event('move')

        elif self.mouse_state == 'MID' and BASEOPENGL.work_plane_sketch_manager.state != 'CONFIRM':
            self.update_camera(mouse_x=event.pos().x(), mouse_y=event.pos().y())

        else:
            pass

        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
    def workPlaneSketch_mouseReleaseEvent(self, event):
        if self.mouse_state == 'LEFT':
            BASEOPENGL.work_plane_sketch_manager.solve_mouse_event('release')
            
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
    # transform work plane
    def workPlaneTransform_mousePressEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.work_plane_transform_manager.set_current_view_port(current_camera_pos=self.camera.get_cartesian_camera_pos(),
                                                                          current_screen_width=self.width,
                                                                          current_screen_height=self.height,
                                                                          current_projection_matrix=self.get_projection_matrix(),
                                                                          current_model_view_matrix=self.get_model_view_matrix())
            BASEOPENGL.work_plane_transform_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.work_plane_transform_manager.solve_mouse_event('press')
            
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def workPlaneTransform_mouseMoveEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.work_plane_transform_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.work_plane_transform_manager.solve_mouse_event('move')
            
        elif self.mouse_state == 'MID':
            self.update_camera(mouse_x=event.pos().x(), mouse_y=event.pos().y())
        
        else:
            pass

        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def workPlaneTransform_mouseReleaseEvent(self, event):
        if self.mouse_state == 'LEFT':
            BASEOPENGL.work_plane_transform_manager.solve_mouse_event('release')

        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
    # select point cloud
    def pointCloudSelect_mousePressEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            self.point_cloud_select_manager.set_current_view_port(current_camera_pos=self.camera.get_cartesian_camera_pos(),
                                                                  current_screen_width=self.width,
                                                                  current_screen_height=self.height,
                                                                  current_projection_matrix=self.get_projection_matrix(),
                                                                  current_model_view_matrix=self.get_model_view_matrix())
            self.point_cloud_select_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            self.point_cloud_select_manager.mouse_pressed()
        
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def pointCloudSelect_mouseMoveEvent(self, event):
        if self.mouse_state == 'MID':
            self.update_camera(mouse_x=event.pos().x(), mouse_y=event.pos().y())
        
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def pointCloudSelect_mouseReleaseEvent(self, event):
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
    # comp select point cloud
    def pointCloudCompSelect_mousePressEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.point_cloud_comp_select_manager.set_current_view_port(current_camera_pos=self.camera.get_cartesian_camera_pos(),
                                                                             current_screen_width=self.width,
                                                                             current_screen_height=self.height,
                                                                             current_projection_matrix=self.get_projection_matrix(),
                                                                             current_model_view_matrix=self.get_model_view_matrix())
            BASEOPENGL.point_cloud_comp_select_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.point_cloud_comp_select_manager.solve_mouse_event('press')
            
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def pointCloudCompSelect_mouseMoveEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.point_cloud_comp_select_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.point_cloud_comp_select_manager.solve_mouse_event('move')

        elif self.mouse_state == 'MID':
            self.makeCurrent()
            self.update_camera(mouse_x=event.pos().x(), mouse_y=event.pos().y())

        else:
            pass

        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def pointCloudCompSelect_mouseReleaseEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.point_cloud_comp_select_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.point_cloud_comp_select_manager.solve_mouse_event('release')
            
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    # deform point cloud
    def pointCloudDeform_mousePressEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.point_cloud_deform_manager.set_current_view_port(current_camera_pos=self.camera.get_cartesian_camera_pos(),
                                                                  current_screen_width=self.width,
                                                                  current_screen_height=self.height,
                                                                  current_projection_matrix=self.get_projection_matrix(),
                                                                  current_model_view_matrix=self.get_model_view_matrix())
            BASEOPENGL.point_cloud_deform_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.point_cloud_deform_manager.solve_mouse_event('press')
            
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def pointCloudDeform_mouseMoveEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.point_cloud_deform_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.point_cloud_deform_manager.solve_mouse_event('move')
            
        elif self.mouse_state == 'MID':
            self.update_camera(mouse_x=event.pos().x(), mouse_y=event.pos().y())

        else:
            pass
        
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
    def pointCloudDeform_mouseDoubleClickEvent(self, event):
        if self.mouse_state == 'LEFT':
            BASEOPENGL.point_cloud_deform_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.point_cloud_deform_manager.solve_mouse_event('double_click')

        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def pointCloudDeform_mouseReleaseEvent(self, event):
        if self.mouse_state == 'LEFT':
            BASEOPENGL.point_cloud_deform_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.point_cloud_deform_manager.solve_mouse_event('release')
            
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
    ########################################################################
    #    Sketch to 3D                                                      #
    ########################################################################
    def inference(self):
        self.makeCurrent()
        pc_points = inference_net(cfg, upload_image=False)
        GM.base_point_cloud = PointCloud(pc_points)
        GM.base_point_cloud.set_color_according_camera_pos(camera_pos=[1.5, 1.5, 0.0])
        self.update()
    

    def inferenceByLoadImage(self):
        self.makeCurrent()
        pc_points = inference_net(cfg, upload_image=True)
        GM.base_point_cloud = PointCloud(pc_points)
        GM.base_point_cloud.set_color_according_camera_pos(camera_pos=[1.5, 1.5, 0.0])
        
        self.update()
    

    def refine(self):
        self.makeCurrent()
        azi, ele = self.camera.get_azi_ele()
        
        """
        print('view')
        print('azi:')
        print(azi)
        print('ele:')
        print(ele)
        """

        pc_points = refine_net(cfg, GM.base_point_cloud.positions, azi, ele)
        GM.base_point_cloud = PointCloud(pc_points)
        GM.base_point_cloud.set_color_according_camera_pos(camera_pos=[1.5, 1.5, 0.0])
        self.update()
    

    ########################################################################
    #    SET OPENGL                                                        #
    ########################################################################
    def initializeGL(self):
        print("initializeGL")
        
        # set texture
        self.confirm_texture_id = self.read_texture(cfg.INFERENCE.ICONS_PATH + 'confirm.png')
        self.candidate_texture_id = self.read_texture(cfg.INFERENCE.ICONS_PATH + 'candidate.png')
        self.delete_texture_id = self.read_texture(cfg.INFERENCE.ICONS_PATH + 'delete.png')
        self.deform_confirm_texture_id = self.read_texture(cfg.INFERENCE.ICONS_PATH + 'delete.png')
        self.lock_texture_id = self.read_texture(cfg.INFERENCE.ICONS_PATH + 'lock.png')
        self.unlock_texture_id = self.read_texture(cfg.INFERENCE.ICONS_PATH + 'unlock.png')

    def resizeGL(self, width, height):
        pass
    
    ########################################################################
    #    DRAW SCENE                                                        #
    ########################################################################
    def update_camera(self, mouse_x, mouse_y):
        d_theta = float(self.lastPos_x - mouse_x) / 300.
        d_phi = float(self.lastPos_y - mouse_y) / 600.
        self.camera.rotate(d_theta, d_phi)

    def paintGL(self):
        self.makeCurrent()

        azi, ele = self.camera.get_azi_ele()
        
        camera_pos = self.camera.get_cartesian_camera_pos()

        # main window
        glViewport(0, 0, self.width, self.height)
        glClearColor (0.8, 0.8, 0.8, 0.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        self.set_projection()
        self.set_model_view(camera_pos)
        self.draw_main_scene()
        self.show_counter_widget.update_label()

    # scene  
    def draw_main_scene(self):
        # draw point cloud
         
        # self.draw_reference_point_cloud()
        
        self.draw_base_point_cloud()
        self.draw_generate_point_clouds()
        
        # draw lines on work face
        # self.draw_lines_on_work_planes()

        if self.mode[:5] == 'input':
            self.draw_canvas(BASEOPENGL.canvas_manager.sketch_canvas)
            self.draw_canvas(BASEOPENGL.canvas_manager.detail_canvas)
        if self.mode == 'workPlaneSelect':
            self.draw_work_plane_select()
        elif self.mode == 'workPlaneSketch':
            self.draw_work_plane_sketch()
            self.draw_canvas(BASEOPENGL.work_plane_sketch_manager.canvas)
        elif self.mode == 'workPlaneTransform':
            self.draw_work_plane_transform()
        elif self.mode == 'pointCloudCompSelect':
            self.draw_canvas(BASEOPENGL.point_cloud_comp_select_manager.canvas)
        elif self.mode == 'pointCloudDeform':
            self.draw_point_cloud_deform()
            self.draw_canvas(BASEOPENGL.point_cloud_deform_manager.canvas)

    def draw_preview_scene(self):
        self.draw_point_cloud()
        self.draw_lines_on_work_planes()
        
        if self.mode == 'sketch':
            if self.work_plane_manager.sketch_state == 'CONFIRM':
                self.draw_candidate_work_plane()
                self.draw_lines_on_candidate_work_planes()

    # icon
    def draw_canvas(self, canvas):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), 
                                canvas,
                                canvas.rect())
        canvasPainter.end()

    def draw_lock_icon(self):
        icon_size = 0.5
        lock_vertices = [[0.0, -icon_size, icon_size],
                         [0.0, -icon_size, -icon_size],
                         [0.0, icon_size, -icon_size],
                         [0.0, icon_size, icon_size]]

        if self.work_plane_manager.state == 'CONFIRM':
            self.draw_texture_face(lock_vertices, self.lock_texture_id)
        else:
            self.draw_texture_face(lock_vertices, self.unlock_texture_id)
    
    def draw_work_plane_select(self):
        if BASEOPENGL.work_plane_select_manager.state == 'UN_SELECTED':
            pass

        if BASEOPENGL.work_plane_select_manager.state == 'CONFIRM':
           # work plane
           self.draw_transparent_face(BASEOPENGL.work_plane_select_manager.get_selected_work_plane().bounding_rec_3d, 
                                      color=[1.0, 0.0, 0.0, 0.9])
           
           # bill board
           self.draw_texture_face(BASEOPENGL.work_plane_select_manager.confirm_board.bounding_rec_3d, self.confirm_texture_id)
           self.draw_texture_face(BASEOPENGL.work_plane_select_manager.delete_board.bounding_rec_3d, self.delete_texture_id)

        if BASEOPENGL.work_plane_select_manager.state == 'SELECTED':
           # work plane
           self.draw_transparent_face(BASEOPENGL.work_plane_select_manager.get_current_work_plane().bounding_rec_3d,
                                      color=[1.0, 0.0, 0.0, 0.9])
      
    def draw_work_plane_sketch(self):
        if BASEOPENGL.work_plane_sketch_manager.state == 'UN_SELECTED':
            pass

        if BASEOPENGL.work_plane_sketch_manager.state == 'CONFIRM':
            # work plane
            self.draw_transparent_face(BASEOPENGL.work_plane_sketch_manager.get_current_work_plane().bounding_rec_3d, 
                                       color=[1.0, 0.0, 0.0, 0.9])
            
            # if the candidate work plane exist
            if BASEOPENGL.work_plane_sketch_manager.get_candidate_work_plane():
                self.draw_transparent_face(BASEOPENGL.work_plane_sketch_manager.get_candidate_work_plane().bounding_rec_3d,
                                           color=[0.0, 0.0, 1.0, 0.9])

                 # candidate point cloud
                self.draw_point_cloud(BASEOPENGL.work_plane_sketch_manager.get_candidate_work_plane().generate_point_clouds[0])
                
                # bill board
                self.draw_texture_face(BASEOPENGL.work_plane_sketch_manager.confirm_board.bounding_rec_3d, 
                                       self.confirm_texture_id)
                self.draw_texture_face(BASEOPENGL.work_plane_sketch_manager.candidate_board.bounding_rec_3d,
                                       self.candidate_texture_id)

        if BASEOPENGL.work_plane_sketch_manager.state == 'SELECTED':
            # work plane
            self.draw_transparent_face(BASEOPENGL.work_plane_sketch_manager.get_current_work_plane().bounding_rec_3d,
                                       color=[1.0, 0.0, 0.0, 0.9]) 

    def draw_work_plane_transform(self):
         # work plane
        self.draw_transparent_face(BASEOPENGL.work_plane_transform_manager.get_current_work_plane().bounding_rec_3d, 
                                   color=[1.0, 0.0, 0.0, 0.9])

        # transform
        self.draw_rotate(GM.work_planes[GM.current_id].get_transform())
        self.draw_translate(GM.work_planes[GM.current_id].get_transform())
        
    def draw_point_cloud_deform(self):
        if BASEOPENGL.point_cloud_deform_manager.state == 'SELECT_CONTROL_POINTS':
            # control points
            self.draw_control_points()
            self.draw_deform_hints()

        elif BASEOPENGL.point_cloud_deform_manager.state == 'DRAG_CONTROL_POINTS':
            # deform point cloud
            self.draw_point_cloud(BASEOPENGL.point_cloud_deform_manager.get_deform_point_cloud())
            
            # control points
            self.draw_control_points()
            
            # deform axis
            self.draw_translate(BASEOPENGL.point_cloud_deform_manager.lattice.get_transform())

        else:
            pass
    
    def draw_base_point_cloud(self):
        if GM.base_point_cloud:
            self.draw_point_cloud(GM.base_point_cloud)

    def draw_reference_point_cloud(self):
        if GM.reference_point_cloud:
            self.draw_point_cloud(GM.reference_point_cloud)

    def draw_symmetric_point_cloud(self):
        if self.optimize_pc:
            self.draw_point_cloud(self.optimize_pc)
    
    def draw_generate_point_clouds(self):
        for work_plane in GM.work_planes:
            for point_cloud in work_plane.generate_point_clouds:
                self.draw_point_cloud(point_cloud)

    def draw_lines_on_canvas(self):
        for line in GM.canvas.lines_3d:
            if line != {}:
                if line['color'] == [0.8, 0.8, 0.8]:
                    self.draw_line_strip(line['points'], color=line['color'], width=20)
                else:
                    self.draw_line_strip(line['points'], color=line['color'])
    
    def draw_lines_on_work_planes(self):
        for work_plane in GM.work_planes:
            for line in work_plane.lines_3d:
                self.draw_line_strip(line, color=[0.0, 1.0, 0.0], width=BASEOPENGL.work_plane_sketch_manager.current_sketch_width*5)

    def draw_lines_on_candidate_work_planes(self):
        for line in BASEOPENGL.work_plane_select_manager.get_candidate_work_plane().lines_3d:
            self.draw_line_strip(line, color=[0.0, 0.0, 0.0], width=BASEOPENGL.current_sketch_width)
    
    def draw_deform_hints(self):
        glDisable(GL_DEPTH_TEST)
        
        x_axis = BASEOPENGL.point_cloud_deform_manager.lattice.box_x_axis
        y_axis = BASEOPENGL.point_cloud_deform_manager.lattice.box_y_axis
        z_axis = BASEOPENGL.point_cloud_deform_manager.lattice.box_z_axis

        hints_origin = BASEOPENGL.point_cloud_deform_manager.lattice.box_origin \
                       - 0.02*x_axis \
                       - 0.02*y_axis \
                       - 0.02*z_axis

        x_end = hints_origin + 0.05*x_axis
        y_end = hints_origin + 0.05*y_axis
        z_end = hints_origin + 0.05*z_axis
        
        self.draw_line(line=[hints_origin, x_end],
                       color=[1., 0., 0.],
                       width=5)
        self.draw_line(line=[hints_origin, y_end],
                       color=[0., 1., 0.],
                       width=5)
        self.draw_line(line=[hints_origin, z_end],
                       color=[0., 0., 1.],
                       width=5)

        glEnable(GL_DEPTH_TEST)

    def draw_control_points(self):
        for i, pos in enumerate(BASEOPENGL.point_cloud_deform_manager.lattice.control_point_positions):
            if i in BASEOPENGL.point_cloud_deform_manager.lattice.current_ids:
                self.draw_sphere(pos, color=[1.0, 1.0, 0.0], size=0.015)
            else:
                self.draw_sphere(pos, color=[1.0, 0.0, 0.0], size=0.015)
        
        BASEOPENGL.point_cloud_deform_manager.lattice.form_connection()
        
        for connection in BASEOPENGL.point_cloud_deform_manager.lattice.connection_list:
            line = [BASEOPENGL.point_cloud_deform_manager.lattice.control_point_positions[connection[0]],
                    BASEOPENGL.point_cloud_deform_manager.lattice.control_point_positions[connection[1]]]
            self.draw_line(line, color=[0.0, 0.0, 0.0], width=3)

    def draw_selected_center(self):
        self.draw_sphere(BASEOPENGL.point_cloud_deform_manager.lattice.get_selected_center(), color=[0.0, 1.0, 0.0], size=0.012)
    
    def draw_load_image(self):
        # TO DO
        pass
    
    def changeCurser(self):
        if self.mode == 'workPlaneSketch':
            current_width = self.work_plane_sketch_manager.current_sketch_width
            pixmap = QPixmap(QSize(1, 1)*current_width)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.black, 2))
            painter.drawEllipse(pixmap.rect())
            painter.drawPoint(current_width/2., current_width/2.)
            painter.end()
            cursor = QCursor(pixmap)
            QApplication.setOverrideCursor(cursor)
        else:
            QApplication.restoreOverrideCursor()

if __name__ == '__main__':
    glutInit( sys.argv )
    app = QApplication(sys.argv)

    window = EditWidget()
    window.show()
    sys.exit(app.exec_())
