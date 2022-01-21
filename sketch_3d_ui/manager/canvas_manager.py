import numpy as np
from sketch_3d_ui.manager.geometry_manager import GeometryManager as GM
import sketch_3d_ui.geometry.geometry_utils as geometry_utils

from PyQt5.QtCore import QSize, Qt, QRect, QPoint
from PyQt5.QtGui import QColor, QIcon, QPixmap, QScreen, QPainter, QPen, QImage
from PyQt5.QtWidgets import QOpenGLWidget

class CanvasManager(GM):
    def __init__(self):
        super(CanvasManager, self).__init__()
        self.mode = None
        self.line_mode = 'free'
        self.current_canvas_name = 'sketch'

        self.sketch_canvas = QImage(896, 896, QImage.Format_ARGB32)
        self.sketch_canvas.fill(Qt.transparent)

        self.tmp_sketch_canvas = QImage(896, 896, QImage.Format_ARGB32)
        self.tmp_sketch_canvas.fill(Qt.transparent)

        self.detail_canvas = QImage(896, 896, QImage.Format_ARGB32)
        self.detail_canvas.fill(Qt.transparent)

        self.tmp_detail_canvas = QImage(896, 896, QImage.Format_ARGB32)
        self.tmp_detail_canvas.fill(Qt.transparent)
        
        self.brushSize = 5
        # self.brushSize = 20
        self._clear_size = 20
       
    def init_manager(self, mode):
        self.mode = mode
        if self.mode == 'inputSketch':
            self.current_canvas_name = 'sketch'
            self.color = QColor(Qt.green)

        elif self.mode == 'inputDetail':
            self.current_canvas_name = 'detail'
            self.color = QColor(Qt.red)
            
        else:
            pass

    def init_state(self):
        pass
    
    def solve_mouse_event(self, event):
        if event == 'press':
            self.last_pos = QPoint(self.mouse_x, self.mouse_y)
            self.start_pos = QPoint(self.mouse_x, self.mouse_y)
            self.draw_on_canvas()

        elif event == 'move':
            self.draw_on_canvas()
  
        elif event == 'release':
            if self.current_canvas_name == 'sketch':
                self.tmp_sketch_canvas = self.sketch_canvas.copy()
            elif self.current_canvas_name == 'detail':
                self.tmp_detail_canvas = self.detail_canvas.copy()

        else:
            pass

    def draw_on_canvas(self):
        current_pos = QPoint(self.mouse_x, self.mouse_y)

        if self.current_canvas_name == 'sketch':
            if self.line_mode == 'straight' and self.mode == 'inputSketch':
                self.sketch_canvas = self.tmp_sketch_canvas.copy()

            painter = QPainter(self.sketch_canvas)
            painter.setPen(QPen(self.color,
                           self.brushSize,
                           Qt.SolidLine,
                           Qt.RoundCap,
                           Qt.RoundJoin))

        elif self.current_canvas_name == 'detail':
            if self.line_mode == 'straight' and self.mode == 'inputDetail':
                self.detail_canvas = self.tmp_detail_canvas.copy()

            painter = QPainter(self.detail_canvas)
            painter.setPen(QPen(self.color,
                           self.brushSize,
                           Qt.SolidLine,
                           Qt.RoundCap,
                           Qt.RoundJoin))

        if self.mode == 'inputErase': 
            r = QRect(QPoint(), self._clear_size*QSize())
            r.moveCenter(current_pos)
            painter.save()
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.eraseRect(r)
            painter.restore()
        
        elif self.line_mode == 'straight':
            painter.drawLine(self.start_pos, current_pos)

        elif self.line_mode == 'free':
            painter.drawLine(self.last_pos, current_pos)

        else:
            pass

        painter.end()
        self.last_pos = QPoint(self.mouse_x, self.mouse_y)
        
