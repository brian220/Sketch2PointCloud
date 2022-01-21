# Counting the numbers of different operations for user study

class COUNTER:
    count_3d_line = 0
    count_work_plane = 0
    
    count_plane_selection = 0
    count_plane_deletion = 0
    count_plane_creation = 0
    count_plane_transformation = 0
    
    count_point_cloud_selection = 0
    count_point_cloud_deformation = 0

def reset_counter():
    COUNTER.count_3d_line = 0
    COUNTER.count_work_plane = 0
    
    # for operations
    COUNTER.count_plane_selection = 0
    COUNTER.count_plane_deletion = 0
    COUNTER.count_plane_creation = 0
    COUNTER.count_plane_transformation = 0
    
    COUNTER.count_point_cloud_selection = 0
    COUNTER.count_point_cloud_deformation = 0

