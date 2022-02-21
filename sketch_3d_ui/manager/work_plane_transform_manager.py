'''
The manager can solve the work plane transformation,
users can translate or rotate the work planes.
'''

from sketch_3d_ui.manager.geometry_manager import GeometryManager as GM
import sketch_3d_ui.geometry.geometry_utils as geometry_utils

from sketch_3d_ui.counter import COUNTER

class WorkPlaneTransformManager(GM):
    def __init__(self):
         # hit_type: R for rotation, T for translation
        self.hit_type = None
        self.last_hit_point = []
        self.hit_axis = None

    def init_manager(self):
        COUNTER.count_plane_transformation += 1

        self.init_transform()
    
    def init_transform(self):
        GM.work_planes[GM.current_id].init_transform()

    def solve_mouse_event(self, event):
        if event == 'press':
            hit, hit_type, hit_point, hit_axis = self.check_click_transform()
            if hit:
                self.hit_axis = hit_axis
                self.hit_type = hit_type
                self.last_hit_point = hit_point

        elif event == 'move':
            hit, hit_type, hit_point, hit_axis = self.check_click_transform()
            if hit:
                if self.hit_type == 'R': 
                    self.rotate_work_plane(hit_point)  
                if self.hit_type == 'T':
                    self.translate_work_plane(hit_point)
                
                self.last_hit_point = hit_point

        elif event == 'release':
            self.reset_transform()

        else:
            pass

    def check_click_transform(self):
        hit = False
        hit_type = None
        hit_axis = None
        hit_point = None
        
        if self.hit_type == 'R' or self.hit_type == None:
            hit, hit_point, hit_axis = \
                self.ray_rotate_hit_detection(GM.work_planes[GM.current_id].get_transform())
            if hit:
                hit_type = 'R'
           
        if not hit and (self.hit_type == 'T' or self.hit_type == None):
            hit, hit_point, hit_axis = \
                self.ray_translate_hit_detection(GM.work_planes[GM.current_id].get_transform())
            if hit:
                hit_type = 'T'

        return hit, hit_type, hit_point, hit_axis

    def rotate_work_plane(self, hit_point):
        last_vector = self.last_hit_point - GM.work_planes[GM.current_id].get_center()
        last_vector = geometry_utils.normalized_vector(last_vector)

        current_vector = hit_point - GM.work_planes[GM.current_id].get_center()
        current_vector = geometry_utils.normalized_vector(current_vector)

        r_mat = geometry_utils.rotation_matrix_from_vectors(last_vector, current_vector)
        GM.work_planes[GM.current_id].rotate(r_mat)

    def translate_work_plane(self, hit_point):
        vector = hit_point - self.last_hit_point
        GM.work_planes[GM.current_id].translate(vector)
    
    def reset_transform(self):
        self.hit_type = None
