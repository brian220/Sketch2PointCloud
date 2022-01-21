import numpy as np
from PIL import Image
import math
import json
import os, sys
import torch
import torch.backends.cudnn
import torch.utils.data
import torch.nn as nn

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from pyntcloud import PyntCloud

# For UI
import sketch_3d_ui.geometry.geometry_utils as geometry_utils
from sketch_3d_ui.geometry.point_cloud import PointCloud

from sketch_3d_ui.manager.geometry_manager import GeometryManager as GM
from sketch_3d_ui.manager.canvas_manager import CanvasManager
from sketch_3d_ui.manager.work_plane_select_manager import WorkPlaneSelectManager
from sketch_3d_ui.manager.work_plane_sketch_manager import WorkPlaneSketchManager
from sketch_3d_ui.manager.work_plane_transform_manager import WorkPlaneTransformManager
from sketch_3d_ui.manager.point_cloud_select_manager import PointCloudSelectManager
from sketch_3d_ui.manager.point_cloud_comp_select_manager import PointCloudCompSelectManager
from sketch_3d_ui.manager.point_cloud_deform_manager import PointCloudDeformManager

from sketch_3d_ui.view.camera import Camera_Z_UP

from sketch_3d_ui.utils.objloader import OBJ

class BaseOpenGLWidget(QOpenGLWidget):
    # sketch
    canvas_manager = CanvasManager()
    
    # work plane
    work_plane_select_manager = WorkPlaneSelectManager()
    work_plane_sketch_manager = WorkPlaneSketchManager()
    work_plane_transform_manager = WorkPlaneTransformManager()
        
    # point cloud
    point_cloud_select_manager = PointCloudSelectManager()
    point_cloud_comp_select_manager = PointCloudCompSelectManager()
    point_cloud_deform_manager = PointCloudDeformManager()

    def __init__(self, parent=None):
        super(BaseOpenGLWidget, self).__init__(parent)
         
        self.width = 896
        self.height = 896
        self.sub_width = 448
        self.sub_height = 448

        # mode
        self.mode = 'select'

        # eye
        self.azi = 0.
        self.ele = 0.
        self.camera = Camera_Z_UP(theta=self.azi*math.pi/180., \
                                  phi= (90. - self.ele)*math.pi/180., \
                                  distance=2.0)
        
        
        
        self.mouse_state = None
    
    ########################################################################
    #    Mode                                                              #
    ########################################################################
    def set_mode(self, mode):
        self.mode = mode

    def init_mode(self, mode):
        pass
    
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
    
    def mouseClickedEvent(self, e):
        fn = getattr(self, "%s_mouseClickedEvent" % self.mode, None)
        if fn:
            return fn(e)

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
    #    SET SCREEN                                                        #
    ########################################################################
    def minimumSizeHint(self):
        # return QSize(self.width + self.sub_width, self.height)
        return QSize(self.width, self.height)

    def sizeHint(self):
        # return QSize(self.width + self.sub_width, self.height)
        return QSize(self.width, self.height)
    
    ########################################################################
    #    CAMERA                                                            #
    ########################################################################
    def update_camera(self, mouse_x, mouse_y):
        d_theta = float(self.lastPos_x - mouse_x) / 300.
        d_phi = float(self.lastPos_y - mouse_y) / 600.
        self.camera.rotate(d_theta, d_phi)
    
    def get_model_view_matrix(self):
        # glGetFloatv output the column order vector
        # for matrix multiplication, need to be transposed
        model_view_matrix = np.array(glGetFloatv(GL_MODELVIEW_MATRIX))

        return model_view_matrix.transpose()

    def get_projection_matrix(self):
        # glGetFloatv output the column order vector
        # for matrix multiplication, need to be transposed
        projection_matrix = np.array(glGetFloatv(GL_PROJECTION_MATRIX))

        return projection_matrix.transpose()
    
    def set_projection(self):
        # set up projection matrix
        near = 0.5
        far = 100.
        A = (near + far)
        B = near*far
        persp = np.array([
                           [420., 0., -112., 0.],
                           [0., 420., -112., 0.],
                           [0., 0., A, B],
                           [0., 0., -1., 0.]
                        ]).transpose()  

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 224, 0, 224, near, far)
        glMultMatrixf(persp)
    
    def set_model_view(self, camera_pos):
        # set up model view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    def set_model_view_by_candidate_plane(self, candidate_center):
        camera_pos = (candidate_center / np.sqrt(np.sum(candidate_center**2)))*2.

        # set up model view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        # self.point_cloud.set_color(camera_pos = camera_pos)
    
    ########################################################################
    #    SET OPENGL                                                        #
    ########################################################################
    def initializeGL(self):
        print("initializeGL")
        pass
        
    def resizeGL(self, width, height):
        pass
    
    ########################################################################
    #    DRAW SCENE                                                        #
    ########################################################################
    def paintGL(self):
        pass
    
    # basic shape
    def draw_cube(self, pos, color, size):
        glPushMatrix()

        glTranslated(pos[0], pos[1], pos[2])
        glColor3fv(color)
        glutSolidCube(size)
        
        glPopMatrix()
    
    def draw_point_cloud(self, point_cloud, size=0.012):
        for i, (pos, color) in enumerate(zip(point_cloud.positions, point_cloud.colors)):
            self.draw_sphere(pos, color, size=size)

    def draw_sphere(self, pos, color, size):
        glPushMatrix()

        glTranslated(pos[0], pos[1], pos[2])
        glColor3fv(color)
        glutSolidSphere(size, 8, 6)
        
        glPopMatrix()
    
    def draw_skeleton_cube(self, pos, color, size):
        cube_size = 0.03
        verticies = (
            (cube_size, -cube_size, -cube_size),
            (cube_size, cube_size, -cube_size),
            (-cube_size, cube_size, -cube_size),
            (-cube_size, -cube_size, -cube_size),
            (cube_size, -cube_size, cube_size),
            (cube_size, cube_size, cube_size),
            (-cube_size, -cube_size, cube_size),
            (-cube_size , cube_size, cube_size)
        )

        edges = (
            (0,1),
            (0,3),
            (0,4),
            (2,1),
            (2,3),
            (2,7),
            (6,3),
            (6,4),
            (6,7),
            (5,1),
            (5,4),
            (5,7)
        )

        glPushMatrix()

        glTranslated(pos[0], pos[1], pos[2])
       
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glColor3fv(color)
                glVertex3fv(verticies[vertex])
        glEnd()

        glPopMatrix()
    
    def draw_face(self, vertices, color):
        glBegin(GL_QUADS)
        for vertex in vertices:
            glColor4fv(color)
            glVertex3fv(vertex)
        glEnd()

    def draw_transparent_face(self, bounding_rec_3d, color):
        glEnable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        self.draw_face(bounding_rec_3d, color=color)
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
    
    def draw_transparent_texture_face(self, bounding_rec_3d, texture_id):
        glEnable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        self.draw_texture_face(bounding_rec_3d, texture_id)

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    def draw_texture_face(self, bounding_rec_3d, texture_id):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        glBegin(GL_QUADS)

        glTexCoord2f(0.0, 0.0)
        glVertex3fv(bounding_rec_3d[1])
        
        glTexCoord2f(1.0, 0.0)
        glVertex3fv(bounding_rec_3d[2])
        
        glTexCoord2f(1.0, 1.0)
        glVertex3fv(bounding_rec_3d[3])

        glTexCoord2f(0.0, 1.0)
        glVertex3fv(bounding_rec_3d[0])

        glEnd()

        glDisable(GL_TEXTURE_2D)
    
    def draw_line(self, line, color, width=3):
        glLineWidth(width)
        glBegin(GL_LINES)
        for vertex in line:
            glColor3fv(color)
            glVertex3fv(vertex)
        glEnd()

    def draw_line_strip(self, line, color, width=5):
        glLineWidth(width)
        glBegin(GL_LINE_STRIP)
        for vertex in line:
            glColor3fv(color)
            glVertex3fv(vertex)
        glEnd()

    def draw_mesh(self, mesh, color):
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
           for vertex_id in face:
               glColor3fv(color)
               vertex = mesh.vertices[vertex_id]
               glVertex3fv(vertex)
        glEnd()

    def read_texture(self, texture_path):
        img = Image.open(texture_path)
        flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = flipped_img.convert("RGBA").tobytes()

        glEnable(GL_TEXTURE_2D)
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
         
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glDisable(GL_TEXTURE_2D)

        return texture_id    

    def draw_translate(self, transform):
        glDisable(GL_DEPTH_TEST)

        self.draw_line(line=[transform.start_x, transform.end_x],
                       color=[1.0, 0.0, 0.0],
                       width=5)
        self.draw_line(line=[transform.start_y, transform.end_y],
                       color=[1.0, 0.0, 0.0],
                       width=5)
        self.draw_line(line=[transform.start_z, transform.end_z],
                       color=[1.0, 0.0, 0.0],
                       width=5)

        glEnable(GL_DEPTH_TEST)

    def draw_rotate(self, transform):
        glDisable(GL_DEPTH_TEST)

        circle_points = []
        r = 0.1
        for i in range(360):
            theta = (math.pi*i)/180.
            x = r*math.cos(theta)
            y = r*math.sin(theta)
            circle_points.append([x, y, 0.])
        
        ori_vector = [0.0, 0.0, 1.0]
        
        x_r_mat = geometry_utils.rotation_matrix_from_vectors(ori_vector, transform.vector_x)
        x_r_circle_points = np.dot(circle_points, np.transpose(x_r_mat))
        x_t_r_circle_points = x_r_circle_points + transform.get_center()

        y_r_mat = geometry_utils.rotation_matrix_from_vectors(ori_vector, transform.vector_y)
        y_r_circle_points = np.dot(circle_points, np.transpose(y_r_mat))
        y_t_r_circle_points = y_r_circle_points + transform.get_center()

        z_r_mat = geometry_utils.rotation_matrix_from_vectors(ori_vector, transform.vector_z)
        z_r_circle_points = np.dot(circle_points, np.transpose(z_r_mat))
        z_t_r_circle_points = z_r_circle_points + transform.get_center()

        glLineWidth(3)
        glPushMatrix()
        
        # draw x circle
        glBegin(GL_LINE_STRIP)
        for point in x_t_r_circle_points:
            glColor3fv([1.0, 0.0, 0.0])
            glVertex3fv(point)
        glEnd()

        # draw y circle
        glBegin(GL_LINE_STRIP)
        for point in y_t_r_circle_points:
            glColor3fv([0.0, 1.0, 0.0])
            glVertex3fv(point)
        glEnd()

        # draw z circle
        glBegin(GL_LINE_STRIP)
        for point in z_t_r_circle_points:
            glColor3fv([0.0, 0.0, 1.0])
            glVertex3fv(point)
        glEnd()

        glPopMatrix()

        glEnable(GL_DEPTH_TEST)

    def draw_obj(self, filename):
        obj = OBJ(filename)

        glPushMatrix()
        obj.render()
        glPopMatrix()

