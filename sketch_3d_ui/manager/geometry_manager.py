import numpy as np

import sketch_3d_ui.geometry.geometry_utils as geometry_utils
from sketch_3d_ui.view.viewport import ViewPort
from sketch_3d_ui.geometry.canvas import Canvas
from sketch_3d_ui.geometry.point_cloud import PointCloud

class GeometryManager:
    rendering_mask = []

    # point cloud
    base_point_cloud = None

    reference_point_cloud = None
    
    select_point_cloud = False
    current_point_cloud = None
    current_point_cloud_select_mode = None
    current_point_cloud_data = {}
    current_point_cloud_comp_data = {}
    
    # work plane
    work_planes = []
    current_id = None

    def __init__(self):
        self.state = None

    def get_current_work_plane(self):
        return GeometryManager.work_planes[GeometryManager.current_id]

    def get_current_point_cloud(self):
        if GeometryManager.current_point_cloud_select_mode == 'click':
            if GeometryManager.current_point_cloud_data['select_base_model']:
                return GeometryManager.base_point_cloud
    
            else:
                work_plane_id = GeometryManager.current_point_cloud_data['work_plane_id']
                line_id = GeometryManager.current_point_cloud_data['line_id']
                return GeometryManager.work_planes[work_plane_id].generate_point_clouds[line_id]
        
        elif GeometryManager.current_point_cloud_select_mode == 'comp':
            positions = []
            for data in GeometryManager.current_point_cloud_comp_data:
                if data['type'] == 'base':
                    pos = GeometryManager.base_point_cloud.positions[data['id']]
                    positions.append(pos)

                elif data['type'] == 'work_plane':
                    work_plane_id = data['work_plane_id']
                    line_id = data['line_id']
                    point_id = data['id']
                    pos = \
                    GeometryManager.work_planes[work_plane_id].generate_point_clouds[line_id].positions[point_id]
                    positions.append(pos)
           
            positions = np.array(positions).astype(np.float32)
            point_cloud = PointCloud(positions)

            return point_cloud
    
    def get_work_plane_number(self):
        return len(GeometryManager.work_planes)

    def set_current_point_cloud(self, point_cloud):
        if GeometryManager.current_point_cloud_select_mode == 'click':
            if GeometryManager.current_point_cloud_data['select_base_model']:
                GeometryManager.base_point_cloud = point_cloud
    
            else:
                work_plane_id = GeometryManager.current_point_cloud_data['work_plane_id']
                line_id = GeometryManager.current_point_cloud_data['line_id']
                GeometryManager.work_planes[work_plane_id].generate_point_clouds[line_id] = point_cloud

        elif GeometryManager.current_point_cloud_select_mode == 'comp':
            for deform_id, data in enumerate(GeometryManager.current_point_cloud_comp_data):
                if data['type'] == 'base':
                    GeometryManager.base_point_cloud.positions[data['id']] = point_cloud.positions[deform_id]
                
                elif data['type'] == 'work_plane':
                    work_plane_id = data['work_plane_id']
                    line_id = data['line_id']
                    point_id = data['id']
                    GeometryManager.work_planes[work_plane_id].generate_point_clouds[line_id].positions[point_id] = \
                    point_cloud.positions[deform_id]
    
    def reset_point_cloud_color(self, reset_base):
        if reset_base:
            GeometryManager.base_point_cloud.set_color_according_camera_pos()
        
        for work_plane_id, work_plane in enumerate(GeometryManager.work_planes):
            for line_id, point_cloud in enumerate(work_plane.generate_point_clouds):
                GeometryManager.work_planes[work_plane_id].generate_point_clouds[line_id].set_color_according_camera_pos()

    ########################################################################
    #    State Machine                                                     #
    ########################################################################
    def init_mode(self):
        self.init_state()
    
    def init_state(self):
        self.state = 'UN_SELECTED'

    def update_state(self):
        pass

    ########################################################################
    #    View port                                                         #
    ########################################################################
    def set_current_view_port(self, 
                              current_camera_pos,
                              current_screen_width,
                              current_screen_height,
                              current_projection_matrix,
                              current_model_view_matrix):

        self.current_view_port  = ViewPort(camera_pos=current_camera_pos,
                                           screen_width=current_screen_width,
                                           screen_height=current_screen_height,
                                           projection_matrix=current_projection_matrix,
                                           model_view_matrix=current_model_view_matrix)

    def set_mouse_xy(self, mouse_x, mouse_y):
        self.mouse_x = mouse_x
        self.mouse_y = mouse_y
    
    ########################################################################
    #    Ray point cloud hit detection                                     #
    ########################################################################
    # ray v.s point cloud
    def mouse_ray_and_point_cloud_hit_detection(self, mouse_x, mouse_y, point_cloud):
        hit = False

        ray_dir = geometry_utils.screen_pos_to_world_ray(mouseX=mouse_x, mouseY=mouse_y,
                                                         screenWidth=self.current_view_port.screen_width, 
                                                         screenHeight=self.current_view_port.screen_height,
                                                         ProjectionMatrix=self.current_view_port.projection_matrix,
	                                                     ViewMatrix=self.current_view_port.model_view_matrix)

        hit_point_id = geometry_utils.ray_point_cloud_hit_detection(ray_origin=self.current_view_port.camera_pos,
                                                                    ray_dir=ray_dir,
                                                                    point_cloud_positions=point_cloud,
                                                                    hit_radius=0.02)

        if hit_point_id:
            hit = True
        
        return hit, hit_point_id

    # ray v.s work planes point cloud
    def mouse_ray_and_work_plane_point_cloud_hit_detection(self, mouse_x, mouse_y, work_planes):
        hit = False
        work_plane_id = None
        line_id = None
        
        ray_dir = geometry_utils.screen_pos_to_world_ray(mouseX=mouse_x, mouseY=mouse_y,
                                                         screenWidth=self.current_view_port.screen_width,
                                                         screenHeight=self.current_view_port.screen_height,
                                                         ProjectionMatrix=self.current_view_port.projection_matrix,
	                                                     ViewMatrix=self.current_view_port.model_view_matrix)

        
        for i, work_plane in enumerate(work_planes):
            for j, point_cloud in enumerate(work_plane.generate_point_clouds):
                hit_id = geometry_utils.ray_point_cloud_hit_detection(ray_origin=self.current_view_port.camera_pos,
                                                                      ray_dir=ray_dir,
                                                                      point_cloud_positions=point_cloud.positions,
                                                                      hit_radius=0.02)
                if hit_id:
                    hit = True
                    work_plane_id = i
                    line_id = j
                    return hit, work_plane_id, line_id

        return hit, work_plane_id, line_id
        
    ########################################################################
    #    Ray plane hit detection                                           #
    ########################################################################
    # ray v.s planes
    def mouse_ray_and_planes_hit_detection(self, mouse_x, mouse_y, planes, boundary=True):
        hit = False
        hit_point = None
        hit_id = None

        ray_dir = geometry_utils.screen_pos_to_world_ray(mouseX=mouse_x, mouseY=mouse_y,
                                                         screenWidth=self.current_view_port.screen_width, 
                                                         screenHeight=self.current_view_port.screen_height,
                                                         ProjectionMatrix=self.current_view_port.projection_matrix,
	                                                     ViewMatrix=self.current_view_port.model_view_matrix)
        
        if boundary == True:
            for i, plane in enumerate(planes):
                hit, hit_point, _ = geometry_utils.ray_plane_hit_detection_with_boundary(ray_origin=self.current_view_port.camera_pos,
                                                                                         ray_dir=ray_dir,
                                                                                         vertices=plane.bounding_rec_3d)
                if hit:
                    hit_id = i
                    hit_point = hit_point
                    break

        else:
            for i, plane in enumerate(planes):
                hit, hit_point, _ = geometry_utils.ray_plane_hit_detection(plane_point=plane.point, 
                                                                           plane_normal=plane.normal,
                                                                           ray_origin=self.current_view_port.camera_pos,
                                                                           ray_dir=ray_dir)
                if hit:
                    hit_id = i
                    hit_point = hit_point
                    break
        
        return hit, hit_point, hit_id

    ########################################################################
    #    Ray transform hit detection                                       #
    ########################################################################
    def ray_translate_hit_detection(self, transform):
        hit = False
        hit_point = None
        hit_axis = None
        min_t = 1000.

        ray_dir = geometry_utils.screen_pos_to_world_ray(mouseX=self.mouse_x, mouseY=self.mouse_y,
                                                         screenWidth=self.current_view_port.screen_width,
                                                         screenHeight=self.current_view_port.screen_height,
                                                         ProjectionMatrix=self.current_view_port.projection_matrix,
	                                                     ViewMatrix=self.current_view_port.model_view_matrix)
        
        hit_x, hit_point_x, t_x = geometry_utils.ray_axis_hit_detection(axis_start=transform.start_x,
                                                                        axis_end=transform.end_x,
                                                                        axis=transform.vector_x,
                                                                        ray_origin=self.current_view_port.camera_pos,
                                                                        ray_dir=ray_dir,
                                                                        thickness=0.02)

        if hit_x and t_x < min_t:
            hit_axis = 'X'
            hit_point = hit_point_x
            min_t = t_x
            hit = True

        hit_y, hit_point_y, t_y = geometry_utils.ray_axis_hit_detection(axis_start=transform.start_y,
                                                                        axis_end=transform.end_y,
                                                                        axis=transform.vector_y,
                                                                        ray_origin=self.current_view_port.camera_pos,
                                                                        ray_dir=ray_dir,
                                                                        thickness=0.02)
        
        if hit_y and t_y < min_t:
            hit_axis = 'Y'
            hit_point = hit_point_y
            min_t = t_y
            hit = True

        hit_z, hit_point_z, t_z = geometry_utils.ray_axis_hit_detection(axis_start=transform.start_z,
                                                                        axis_end=transform.end_z,
                                                                        axis=transform.vector_z,
                                                                        ray_origin=self.current_view_port.camera_pos,
                                                                        ray_dir=ray_dir,
                                                                        thickness=0.02)

        if hit_z and t_z < min_t:
            hit_axis = 'Z'
            hit_point = hit_point_z
            min_t = t_z
            hit = True

        return hit, hit_point, hit_axis

    def ray_rotate_hit_detection(self, transform):
        hit = False
        hit_point = None
        hit_axis = None
        min_t = 1000.

        ray_dir = geometry_utils.screen_pos_to_world_ray(mouseX=self.mouse_x, mouseY=self.mouse_y,
                                                         screenWidth=self.current_view_port.screen_width,
                                                         screenHeight=self.current_view_port.screen_height,
                                                         ProjectionMatrix=self.current_view_port.projection_matrix,
	                                                     ViewMatrix=self.current_view_port.model_view_matrix)

        # detect hit for circle x
        hit_x, hit_point_x, t_x = geometry_utils.ray_circle_hit_detection(plane_center=transform.center,
                                                                          plane_normal=transform.vector_x,
                                                                          ray_origin=self.current_view_port.camera_pos,
                                                                          ray_dir=ray_dir,
                                                                          hit_radius=0.1,
                                                                          thickness=0.02)
        if hit_x and t_x < min_t:
            hit_axis = 'X'
            hit_point = hit_point_x
            min_t = t_x
            hit = True

        # detect hit for circle y
        hit_y, hit_point_y, t_y = geometry_utils.ray_circle_hit_detection(plane_center=transform.center,
                                                                          plane_normal=transform.vector_y,
                                                                          ray_origin=self.current_view_port.camera_pos,
                                                                          ray_dir=ray_dir,
                                                                          hit_radius=0.1,
                                                                          thickness=0.02)

        if hit_y and t_y < min_t:
            hit_axis = 'Y'
            hit_point = hit_point_y
            min_t = t_y
            hit = True

        # detect hit for circle z
        hit_z, hit_point_z, t_z = geometry_utils.ray_circle_hit_detection(plane_center=transform.center,
                                                                          plane_normal=transform.vector_z,
                                                                          ray_origin=self.current_view_port.camera_pos,
                                                                          ray_dir=ray_dir,
                                                                          hit_radius=0.1,
                                                                          thickness=0.02)
        
        if hit_z and t_z < min_t:
            hit_axis = 'Z'
            hit_point = hit_point_z
            min_t = t_z
            hit = True
        
        return hit, hit_point, hit_axis

    ########################################################################
    #    Bill board                                                        #
    ########################################################################
    def init_bill_board_list(self, left_top, bill_boards=[], bill_board_size=0.03):
        view_matrix = self.current_view_port.model_view_matrix
        # camera space y in world space
        camera_up_world_space = np.array(view_matrix[1][:-1])
        # camera space x in world space
        camera_right_world_space = np.array(view_matrix[0][:-1])
        
        camera_vector = self.current_view_port.camera_pos - left_top
        
        offset = 0.
        for bill_board in bill_boards:
            bboard = getattr(self, bill_board)
            bboard.init_bill_board(left_top - offset + camera_vector*0.5, 
                                   camera_up_world_space,
                                   camera_right_world_space,
                                   bill_board_size)
            offset += camera_up_world_space*bboard.bill_board_size