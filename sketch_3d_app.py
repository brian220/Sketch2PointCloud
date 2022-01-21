import sys
import math
import os
from shutil import copyfile

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from configs.config_ui import cfg

from sketch_3d_ui.edit_widget import EditWidget
from sketch_3d_ui.preview_widget import PreviewWidget

from sketch_3d_ui.manager.geometry_manager import GeometryManager as GM

from functools import partial

TOOLS = [
    'input',
    'workPlane',
    'pointCloud',
]

MODES = [
    'inputSketch',
    'inputErase',
    'inputDetail',
    'workPlaneSelect',
    'workPlaneSketch',
    'workPlaneTransform',
    'pointCloudSelect',
    'pointCloudCompSelect',
    'pointCloudDeform',
]

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()

        self.current_path = None
        
        self.button_size = 80
        
        self.setDrawLineDialog = setDrawLineDialog()
        self.setFFDDialog = setFFDDialog()
        self.setWorkPlaneDialog = SetWorkPlaneDialog()
        
        mainWidget = QWidget()
        layout = QHBoxLayout()
        self.referenceModelLayout = QVBoxLayout()

        self.edit_widget = EditWidget()
        
        self.createMenuBars()
        self.createMenuActions()
        self.createToolBars()
        self.createToolBarActions()
        self.createReferenceActions()
        
        self.menuBarAddActions()
        self.toolBarAddActions()
        
        scrollArea = QScrollArea()
        scrollArea.setWidget(self.referenceToolBar)
        
        self.addToolBar(self.inputToolBar)

        self.inputToolBar.show()
        
        layout.addWidget(scrollArea)
        layout.addWidget(self.edit_widget)
        
        mainWidget.setLayout(layout)
         
        self.setCentralWidget(mainWidget)

        self.menuBarTriggerActions()
        self.toolBarTriggerActions()
    
    # Menu bars
    def createMenuBars(self):
        self.fileMenu = QMenu("&File", self)
        self.editMenu = QMenu("&Edit", self)
        self.sketch3DMenu = QMenu("&sketch3D", self)
        self.settingMenu = QMenu("&settings", self)
        self.optimizeMenu = QMenu("&optimize", self)

    def createMenuActions(self):
        # For
        self.saveResultAction = QAction("&Save result", self)
        self.loadPointCloudAction = QAction("&Load point cloud", self)
        self.uploadSketchAction = QAction("&Upload sketch", self)

        # For edit menu
        self.inputAction = QAction("&Input sketch", self)
        self.workPlaneAction = QAction("&Edit work plane", self)
        self.pointCloudAction = QAction("&Edit point cloud", self)
        
        # For sketch 3d menu
        self.inferenceAction = QAction("&Inference", self)
        self.refineAction = QAction("&Refine", self)

        self.symmetricOptimizeAction = QAction("&Symmetric optimize", self)
        
        # For setting menu
        self.setDrawLineAction = QAction("&Draw Line Settings", self)
        self.setFFDAction = QAction("&FFD Setings", self)
        self.setWorkPlaneAction = QAction("&Work Plane Settings", self)
    
    def menuBarAddActions(self):
        menuBar = self.menuBar()
        
        self.fileMenu.addAction(self.saveResultAction)
        self.fileMenu.addAction(self.loadPointCloudAction)
        self.fileMenu.addAction(self.uploadSketchAction)
        
        self.editMenu.addAction(self.inputAction)
        self.editMenu.addAction(self.workPlaneAction)
        self.editMenu.addAction(self.pointCloudAction)
        
        self.sketch3DMenu.addAction(self.inferenceAction)
        self.sketch3DMenu.addAction(self.refineAction)
        
        self.settingMenu.addAction(self.setDrawLineAction)
        self.settingMenu.addAction(self.setWorkPlaneAction)
        self.settingMenu.addAction(self.setFFDAction)

        self.optimizeMenu.addAction(self.symmetricOptimizeAction)

        menuBar.addMenu(self.fileMenu)
        menuBar.addMenu(self.editMenu)
        menuBar.addMenu(self.sketch3DMenu)
        menuBar.addMenu(self.settingMenu)
        menuBar.addMenu(self.optimizeMenu)

    def menuBarTriggerActions(self):
        self.saveResultAction.triggered.connect(lambda: self.saveResult())
        self.loadPointCloudAction.triggered.connect(lambda: self.loadPointCloud())
        self.uploadSketchAction.triggered.connect(lambda: self.uploadSketch())

        self.inputAction.triggered.connect(lambda: self.updateToolBars("input"))
        self.inputAction.triggered.connect(lambda: self.edit_widget.set_mode("view"))

        self.workPlaneAction.triggered.connect(lambda: self.updateToolBars("workPlane"))
        self.workPlaneAction.triggered.connect(lambda: self.edit_widget.set_mode("view"))
        
        self.pointCloudAction.triggered.connect(lambda: self.updateToolBars("pointCloud"))
        self.pointCloudAction.triggered.connect(lambda: self.edit_widget.set_mode("view"))
        
        self.inferenceAction.triggered.connect(lambda: self.inference())
        self.refineAction.triggered.connect(lambda: self.refine())
        
        self.setDrawLineAction.triggered.connect(lambda: self.setDrawLine())
        self.setFFDAction.triggered.connect(lambda: self.setFFD())
        self.setWorkPlaneAction.triggered.connect(lambda: self.setWorkPlane())

        self.symmetricOptimizeAction.triggered.connect(lambda: self.symmetricOptimize())
    
    def saveResult(self):
        myDir = QFileDialog.getExistingDirectory(self,
                                                 "Choose Save Folder",
                                                 cfg.INFERENCE.USER_EVALUATION_PATH,
                                                 QFileDialog.ShowDirsOnly)
        if not myDir:
            return
        
        self.edit_widget.save_result(myDir)
        self.updateToolBars("input")
        self.resetToolBarIcons()
        self.edit_widget.set_mode("view")
    
    def loadPointCloud(self):
        # ATTENTION: ONLY FOR DEBUG
        myDir = QFileDialog.getExistingDirectory(self,
                                                 "Choose Point Cloud Folder",
                                                 cfg.INFERENCE.USER_EVALUATION_PATH,
                                                 QFileDialog.ShowDirsOnly)
        if not myDir:
           return
        
        self.edit_widget.load_point_cloud(myDir)
        self.updateToolBars("input")
        self.resetToolBarIcons()
        self.edit_widget.set_mode("view")

    def uploadSketch(self, filename=None):
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(), 'Images (*.png *.jpg)')
            if not filename:
                return
            print(filename)
    
    def setDrawLine(self):
        if self.setDrawLineDialog.exec():
            self.edit_widget.set_draw_line(self.setDrawLineDialog.getInputs())

    def setFFD(self):
        if self.setFFDDialog.exec():
            self.edit_widget.set_lattice_dimension(self.setFFDDialog.getInputs())
   
    def setWorkPlane(self):
        if self.setWorkPlaneDialog.exec():
            self.edit_widget.set_work_plane_sketch_width(self.setWorkPlaneDialog.getInputs())

    def inference(self):
        self.screenshot(cfg.INFERENCE.SCREENSHOT_IMAGE_PATH)
        self.edit_widget.inference()
        
    def refine(self):
        self.screenshot(cfg.INFERENCE.REFINE_IMAGE_PATH)
        self.edit_widget.refine()

    def symmetricOptimize(self):
        self.edit_widget.symmetric_optimize()

    def screenshot(self, filename):
        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow( self.edit_widget.winId() )
        screenshot.save(filename, 'png')
    
    # Tool bars
    def createToolBars(self):
        self.referenceToolBar = QToolBar("referenceToolBar", self)
        self.referenceToolBar.setOrientation(Qt.Vertical)
        self.referenceToolBar.setStyleSheet("QToolBar {background: rgb(205, 205, 205) }")
        self.referenceToolBar.setIconSize(QSize(150, 150))

        self.inputToolBar = QToolBar("inputToolBar", self)
        self.inputToolBar.setStyleSheet("QToolBar {background: rgb(205, 205, 205) }")
        self.inputToolBar.setIconSize(QSize(70, 70))
        
        self.workPlaneToolBar = QToolBar("workPlaneToolBar", self)
        self.workPlaneToolBar.setStyleSheet("QToolBar {background: rgb(205, 205, 205) }")
        self.workPlaneToolBar.setIconSize(QSize(70, 70))

        self.workPlaneSizeToolBar = QToolBar("workPlaneSizeToolBar", self)
        self.workPlaneSizeToolBar.setStyleSheet("QToolBar {background: rgb(205, 205, 205) }")
        self.workPlaneSizeToolBar.setIconSize(QSize(30, 30))
        
        self.workPlanePaintSizeToolBar = QToolBar("workPlanePaintSizeToolBar", self)
        self.workPlanePaintSizeToolBar.setStyleSheet("QToolBar {background: rgb(205, 205, 205) }")
        self.workPlanePaintSizeToolBar.setIconSize(QSize(30, 30))

        self.pointCloudToolBar = QToolBar("pointCloudToolBar", self)
        self.pointCloudToolBar.setStyleSheet("QToolBar {background: rgb(205, 205, 205) }")
        self.pointCloudToolBar.setIconSize(QSize(70, 70))
    
    def createToolBarActions(self):
        self.inputSketchAction = QAction(QIcon(cfg.INFERENCE.ICONS_PATH  + 'inputSketch_off.png'), "&inputSketch", self)
        self.inputEraseAction = QAction(QIcon(cfg.INFERENCE.ICONS_PATH  + 'inputErase_off.png'), "&inputErase", self)
        self.inputDetailAction = QAction(QIcon(cfg.INFERENCE.ICONS_PATH  + 'inputDetail_off.png'), "&inputDetail", self)
        self.inputInferenceAction = QAction(QIcon(cfg.INFERENCE.ICONS_PATH  + 'inputInference_off.png'), "&inputInference", self)
        self.inputRefineAction = QAction(QIcon(cfg.INFERENCE.ICONS_PATH  + 'inputRefine_off.png'), "&inputRefine", self)

        self.workPlaneSelectAction = QAction(QIcon(cfg.INFERENCE.ICONS_PATH  + 'workPlaneSelect_off.png'), "&workPlaneSelect", self)
        self.workPlaneSketchAction = QAction(QIcon(cfg.INFERENCE.ICONS_PATH  + 'workPlaneSketch_off.png'),"&workPlaneSketch", self)
        self.workPlaneTransformAction = QAction(QIcon(cfg.INFERENCE.ICONS_PATH  + 'workPlaneTransform_off.png'), "&workPlaneTransform", self)
        
        self.pointCloudSelectAction = QAction(QIcon(cfg.INFERENCE.ICONS_PATH  + 'pointCloudSelect_off.png'), "&pointCloudSelect", self)
        self.pointCloudCompSelectAction = QAction(QIcon(cfg.INFERENCE.ICONS_PATH  + 'pointCloudCompSelect_off.png'), "&pointCloudCompSelect", self)
        self.pointCloudDeformAction = QAction(QIcon(cfg.INFERENCE.ICONS_PATH  + 'pointCloudDeform_off.png'), "&pointCloudDeform", self)
    
    def toolBarAddActions(self):
        self.referenceToolBar.addAction(self.reference1Action)
        self.referenceToolBar.addAction(self.reference2Action)
        self.referenceToolBar.addAction(self.reference3Action)
        self.referenceToolBar.addAction(self.reference4Action)
        self.referenceToolBar.addAction(self.reference5Action)
        self.referenceToolBar.addAction(self.reference6Action)
        self.referenceToolBar.addAction(self.reference7Action)
        self.referenceToolBar.addAction(self.reference8Action)
        self.referenceToolBar.addAction(self.reference9Action)
        self.referenceToolBar.addAction(self.reference10Action)

        self.inputToolBar.addAction(self.inputSketchAction)
        self.inputToolBar.addAction(self.inputEraseAction)
        self.inputToolBar.addAction(self.inputDetailAction)
        
        self.workPlaneToolBar.addAction(self.workPlaneSelectAction)
        self.workPlaneToolBar.addAction(self.workPlaneSketchAction)
        self.workPlaneToolBar.addAction(self.workPlaneTransformAction)

        self.pointCloudToolBar.addAction(self.pointCloudSelectAction)
        self.pointCloudToolBar.addAction(self.pointCloudCompSelectAction)
        self.pointCloudToolBar.addAction(self.pointCloudDeformAction)
    
    def updateToolBars(self, toolBarName):
        for tool in TOOLS:
            bar = getattr(self, '%sToolBar' % tool)
            if tool == toolBarName:
                self.addToolBar(bar)
                bar.show()

            else:
                self.removeToolBar(bar)

    def toolBarTriggerActions(self):
        self.inputSketchAction.triggered.connect(lambda: self.edit_widget.set_mode("inputSketch"))
        self.inputSketchAction.triggered.connect(lambda: self.edit_widget.init_mode())
        self.inputSketchAction.triggered.connect(lambda: self.updateToolBarIcons('inputSketch'))
 
        self.inputEraseAction.triggered.connect(lambda: self.edit_widget.set_mode("inputErase"))
        self.inputEraseAction.triggered.connect(lambda: self.edit_widget.init_mode())
        self.inputEraseAction.triggered.connect(lambda: self.updateToolBarIcons('inputErase'))

        self.inputDetailAction.triggered.connect(lambda: self.edit_widget.set_mode("inputDetail"))
        self.inputDetailAction.triggered.connect(lambda: self.edit_widget.init_mode())
        self.inputDetailAction.triggered.connect(lambda: self.updateToolBarIcons('inputDetail'))

        self.workPlaneSelectAction.triggered.connect(lambda: self.edit_widget.set_mode("workPlaneSelect"))
        self.workPlaneSelectAction.triggered.connect(lambda: self.edit_widget.init_mode())
        self.workPlaneSelectAction.triggered.connect(lambda: self.updateToolBarIcons('workPlaneSelect'))

        self.workPlaneSketchAction.triggered.connect(lambda: self.edit_widget.set_mode("workPlaneSketch"))
        self.workPlaneSketchAction.triggered.connect(lambda: self.edit_widget.init_mode())
        self.workPlaneSketchAction.triggered.connect(lambda: self.updateToolBarIcons('workPlaneSketch'))
        
        self.workPlaneTransformAction.triggered.connect(lambda: self.workPlaneTransform())

        self.pointCloudSelectAction.triggered.connect(lambda: self.edit_widget.set_mode("pointCloudSelect"))
        self.pointCloudSelectAction.triggered.connect(lambda: self.edit_widget.init_mode())
        self.pointCloudSelectAction.triggered.connect(lambda: self.updateToolBarIcons('pointCloudSelect'))

        self.pointCloudCompSelectAction.triggered.connect(lambda: self.edit_widget.set_mode("pointCloudCompSelect"))
        self.pointCloudCompSelectAction.triggered.connect(lambda: self.edit_widget.init_mode())
        self.pointCloudCompSelectAction.triggered.connect(lambda: self.updateToolBarIcons('pointCloudCompSelect'))
        
        self.pointCloudDeformAction.triggered.connect(lambda: self.pointCloudDeform())

        # reference
        self.reference1Action.triggered.connect(lambda: self.edit_widget.setReferenceModel(cfg.INFERENCE.REFERENCE1_PATH))
        self.reference2Action.triggered.connect(lambda: self.edit_widget.setReferenceModel(cfg.INFERENCE.REFERENCE2_PATH))
        self.reference3Action.triggered.connect(lambda: self.edit_widget.setReferenceModel(cfg.INFERENCE.REFERENCE3_PATH))
        self.reference4Action.triggered.connect(lambda: self.edit_widget.setReferenceModel(cfg.INFERENCE.REFERENCE4_PATH))
        self.reference5Action.triggered.connect(lambda: self.edit_widget.setReferenceModel(cfg.INFERENCE.REFERENCE5_PATH))
        self.reference6Action.triggered.connect(lambda: self.edit_widget.setReferenceModel(cfg.INFERENCE.REFERENCE6_PATH))
        self.reference7Action.triggered.connect(lambda: self.edit_widget.setReferenceModel(cfg.INFERENCE.REFERENCE7_PATH))
        self.reference8Action.triggered.connect(lambda: self.edit_widget.setReferenceModel(cfg.INFERENCE.REFERENCE8_PATH))
        self.reference9Action.triggered.connect(lambda: self.edit_widget.setReferenceModel(cfg.INFERENCE.REFERENCE9_PATH))
        self.reference10Action.triggered.connect(lambda: self.edit_widget.setReferenceModel(cfg.INFERENCE.REFERENCE10_PATH))
        
    def updateToolBarIcons(self, triggerMode):
        for mode in MODES:
            action = getattr(self, '%sAction' % mode)
            if mode == triggerMode:
                action.setIcon(QIcon(cfg.INFERENCE.ICONS_PATH + "%s.png" % mode))
            else:
                action.setIcon(QIcon(cfg.INFERENCE.ICONS_PATH + "%s_off.png" % mode))
        
    def resetToolBarIcons(self):
        for mode in MODES:
            action = getattr(self, '%sAction' % mode)
            action.setIcon(QIcon(cfg.INFERENCE.ICONS_PATH + "%s_off.png" % mode))

    # Reference layout
    def createReferenceActions(self):
        self.reference1Action = QAction(QIcon(cfg.INFERENCE.REFERENCE1_ICON_PATH), "&reference1", self)
        self.reference2Action = QAction(QIcon(cfg.INFERENCE.REFERENCE2_ICON_PATH), "&reference2", self)
        self.reference3Action = QAction(QIcon(cfg.INFERENCE.REFERENCE3_ICON_PATH), "&reference3", self)
        self.reference4Action = QAction(QIcon(cfg.INFERENCE.REFERENCE4_ICON_PATH), "&reference4", self)
        self.reference5Action = QAction(QIcon(cfg.INFERENCE.REFERENCE5_ICON_PATH), "&reference5", self)
        self.reference6Action = QAction(QIcon(cfg.INFERENCE.REFERENCE6_ICON_PATH), "&reference6", self)
        self.reference7Action = QAction(QIcon(cfg.INFERENCE.REFERENCE7_ICON_PATH), "&reference7", self)
        self.reference8Action = QAction(QIcon(cfg.INFERENCE.REFERENCE8_ICON_PATH), "&reference8", self)
        self.reference9Action = QAction(QIcon(cfg.INFERENCE.REFERENCE9_ICON_PATH), "&reference9", self)
        self.reference10Action = QAction(QIcon(cfg.INFERENCE.REFERENCE10_ICON_PATH), "&reference10", self)

    def workPlaneTransform(self):
        if GM.current_id != None:
            self.edit_widget.set_mode("workPlaneTransform")
            self.edit_widget.init_mode()
            self.updateToolBarIcons('workPlaneTransform')
    
    def pointCloudDeform(self):
        if GM.select_point_cloud:
            self.edit_widget.set_mode("pointCloudDeform")
            self.edit_widget.init_mode()
            self.updateToolBarIcons('pointCloudDeform')

class setDrawLineDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.line_mode = 'free'
        
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QVBoxLayout(self)
        
        self.cb = QComboBox()
        self.cb.addItem("free")
        self.cb.addItem("straight")
        self.cb.currentIndexChanged.connect(self.selectionchange)
        layout.addWidget(self.cb)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def selectionchange(self):
        self.line_mode = self.cb.currentText()

    def getInputs(self):
        return self.line_mode

class setFFDDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.inputX = QLineEdit(self)
        self.inputY = QLineEdit(self)
        self.inputZ = QLineEdit(self)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        
        layout = QFormLayout(self)
        layout.addRow("X", self.inputX)
        layout.addRow("Y", self.inputY)
        layout.addRow("Z", self.inputZ)
        layout.addWidget(buttonBox)
        
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
      
    def getInputs(self):
        if self.inputX.text() == '':
            x_value = 3
        else:
            x_value = int(self.inputX.text())
        
        if self.inputY.text() == '':
            y_value = 3
        else:
            y_value = int(self.inputY.text())
        
        if self.inputZ.text() == '':
            z_value = 3
        else:
            z_value = int(self.inputZ.text())

        return [x_value, y_value, z_value]

class SetWorkPlaneDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QFormLayout(self)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        self.sizeselect = QSlider()
        self.sizeselect.setRange(1,50)
        self.sizeselect.setOrientation(Qt.Horizontal)
        layout.addRow("sketch brush size", self.sizeselect)
        self.sizeselect.valueChanged.connect(lambda s: self.setSize(s))
        layout.addRow(buttonBox)
        
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
    
    def setSize(self, s):
        self.size = s

    def getInputs(self):
        return self.size
    
if __name__ == '__main__':
    glutInit( sys.argv )
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

