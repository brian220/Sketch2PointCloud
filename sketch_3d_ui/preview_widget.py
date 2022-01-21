import numpy as np
from PIL import Image
import math
import json
import os, sys
import cv2

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from pyntcloud import PyntCloud

# For DL Model
from configs.config_refine import cfg as cfg_refine
from core.inference import inference_net
from core.refine import refine_net

# For UI
import sketch_3d_ui.geometry.geometry_utils as geometry_utils
from sketch_3d_ui.geometry.point_cloud import PointCloud

from sketch_3d_ui.base_opengl_widget import BaseOpenGLWidget as BASEOPENGL
from sketch_3d_ui.manager.geometry_manager import GeometryManager as GM
from sketch_3d_ui.view.camera import Camera_Z_UP

class PreviewWidget(BASEOPENGL):
    def __init__(self, parent=None):
        super(PreviewWidget, self).__init__(parent)
         
        self.width = 896
        self.height = 896
        self.sub_width = 448
        self.sub_height = 448

        # mode
        self.mode = 'inputSketch'

        # eye
        self.azi = cfg_refine.OPT.AZI
        self.ele = cfg_refine.OPT.ELE
        
        """
        self.azi = 0.
        self.ele = 0.
        """

        self.camera = Camera_Z_UP(theta=self.azi*math.pi/180., \
                                  phi= (90. - self.ele)*math.pi/180., \
                                  distance=2.0)
        
        self.mouse_state = None
        
    ########################################################################
    #    Mouse Event                                                       #
    ########################################################################
    def mousePressEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            self.mouse_state = 'LEFT'
        elif e.buttons() == Qt.MidButton:
            self.mouse_state = 'MID'
        else:
            return

        fn = getattr(self, "%s_mousePressEvent" % self.mode, None)
        if fn:
            return fn(e)

        self.update()

    def mouseDoubleClickEvent(self, e):
        fn = getattr(self, "%s_mouseDoubleClickEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseMoveEvent(self, e):
        fn = getattr(self, "%s_mouseMoveEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseReleaseEvent(self, e):
        fn = getattr(self, "%s_mouseReleaseEvent" % self.mode, None)
        if fn:
            return fn(e)
      
    ########################################################################
    #    SET OPENGL                                                        #
    ########################################################################
    def initializeGL(self):
        print("initializeGL")
        
    def resizeGL(self, width, height):
        pass

    ########################################################################
    #    SET SCREEN                                                        #
    ########################################################################
    def minimumSizeHint(self):
        return QSize(self.sub_width, self.sub_height)

    def sizeHint(self):
        return QSize(self.sub_width, self.sub_height)
    
    ########################################################################
    #    DRAW SCENE                                                        #
    ########################################################################
    def update_camera(self, mouse_x, mouse_y):
        d_theta = float(self.lastPos_x - mouse_x) / 300.
        d_phi = float(self.lastPos_y - mouse_y) / 600.
        self.camera.rotate(d_theta, d_phi)

    def update_canvas(self):
        self.makeCurrent()

        # update canvas
        BASEOPENGL.canvas_manager.set_current_view_port(current_camera_pos=self.camera.get_cartesian_camera_pos(),
                                                              current_screen_width=self.width,
                                                              current_screen_height=self.height,
                                                              current_projection_matrix=self.get_projection_matrix(),
                                                              current_model_view_matrix=self.get_model_view_matrix())
        BASEOPENGL.canvas_manager.update_canvas()

    def paintGL(self):
        print("paint GL")

        # preview window
        glViewport(0, self.sub_height, self.sub_width, self.sub_height)
        glClearColor (0.8, 0.8, 0.8, 0.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        self.set_projection()
        self.set_model_view(self.camera.get_cartesian_camera_pos())
        self.draw_preview_scene()

        """
        # lock window
        glViewport(0, 0, 100, 100)
        self.set_projection()
        self.set_model_view([2.0, 0.0, 0.0])
        self.draw_lock_icon()
        """

    def draw_preview_scene(self):
        self.draw_base_point_cloud()
        self.draw_generate_point_clouds()
            
    def draw_base_point_cloud(self):
        if GM.base_point_cloud:
            print('GM.base_point_cloud')
            print(GM.base_point_cloud.positions)
    
            if GM.base_point_cloud:
                self.draw_point_cloud(GM.base_point_cloud)
    
    def draw_generate_point_clouds(self):
        for work_plane in GM.work_planes:
            for point_cloud in work_plane.generate_point_clouds:
                self.draw_point_cloud(point_cloud)
