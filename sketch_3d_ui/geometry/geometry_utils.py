import numpy as np

import math
import trimesh
import open3d as o3d

def reconstruct_surface(input_point_pos):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(input_point_pos)

    point_cloud.estimate_normals()
    point_cloud.orient_normals_consistent_tangent_plane(10)
    
    distances = point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud ,o3d.utility.DoubleVector([radius, radius * 2]))
    
    return mesh

def in_segment(point, start_point, end_point):
    v_start = start_point - point
    v_end = end_point - point
    
    if np.dot(v_start, v_end) > 0:
        return False
    else:
        return True

def rotate_according_to_origin(points, center_pos, r_mat):
    # move to origin
    points_origin = points - center_pos
    # rotate
    points_rotate = np.dot(points_origin, np.transpose(r_mat))
    # move back
    points_back = points_rotate + center_pos

    return points_back

def rotation_matrix_from_vectors(src, des):
    a, b = (src / np.linalg.norm(src)).reshape(3), (des / np.linalg.norm(des)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix

def compute_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    angle = np.arccos(dot_product)
    return angle

def vector_length(vector):
    return np.sqrt(np.sum(vector**2))

def normalized_vector(vector):
    return vector/vector_length(vector)

def two_points_distances(p1, p2):
    vector_p1_p2 = p2 - p1
    distance = vector_length(vector_p1_p2)
    return distance

def align_points_to_plane(line,
                          rec,
                          ori_normal,
                          des_normal,
                          align_end_points_vector,
                          align_end_points_center):
    
    # compute center
    rec_center = (rec[0] + rec[2]) / 2.

    # translate to the origin
    line = line - rec_center
    rec = rec - rec_center

    # rotate to align plane
    # check if the normal is same direction
    r_mat_to_plane = rotation_matrix_from_vectors(ori_normal, des_normal)
    line = np.dot(line, np.transpose(r_mat_to_plane))
    rec = np.dot(rec, np.transpose(r_mat_to_plane))
    
    # rotate to align start, end point
    vector_start_end = line[len(line) - 1] - line[0]
    r_mat_to_vector = rotation_matrix_from_vectors(vector_start_end, align_end_points_vector)
    line = np.dot(line, np.transpose(r_mat_to_vector))
    rec = np.dot(rec, np.transpose(r_mat_to_vector))

    # translate to the position according to the center of end points
    translate_vector = align_end_points_center - (line[0] + line[len(line) - 1])/2.
    line = line + translate_vector
    rec = rec + translate_vector
    
    return line, rec, r_mat_to_plane, r_mat_to_vector

def scale_points(points, x_factor, y_factor, z_factor):
    s_mat = np.array([[x_factor, 0., 0.], [0., y_factor, 0.], [0., 0., z_factor]])
    points = np.dot(points, np.transpose(s_mat))

    return points

def screen_pos_to_world_ray(mouseX, mouseY,             # Mouse position, in pixels, from top-left corner of the window
	                        screenWidth, screenHeight,  # Window size, in pixels
                            ProjectionMatrix,           # Camera parameters (ratio, field of view, near and far planes)
	                        ViewMatrix,                 # Camera position and orientation
                            ):
    InverseProjectionMatrix = np.linalg.inv(ProjectionMatrix)
    InverseViewMatrix = np.linalg.inv(ViewMatrix)

    # Transform into normalised device coordinates
    x = (2.0 * float(mouseX)) / float(screenWidth) - 1.0
    y = 1.0 - (2.0 * float(mouseY)) / float(screenHeight)

    # 4d Homogeneous Clip Coordinates
    ray_clip = np.array([x, y, -1.0, 1.0])

    # 4d Eye (Camera) Coordinates
    ray_eye = np.matmul(InverseProjectionMatrix, ray_clip)
    ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0])

    # 4d World Coordinates
    ray_world = np.matmul(InverseViewMatrix, ray_eye)[:3]
    ray_world = ray_world / np.sqrt(np.sum(ray_world**2)) # normalize
    
    return ray_world

def world_pos_to_screen_pos(worldPos,                   # Mouse position, in pixels, from top-left corner of the window
	                        screenWidth, screenHeight,  # Window size, in pixels
                            ProjectionMatrix,           # Camera parameters (ratio, field of view, near and far planes)
	                        ViewMatrix,                 # Camera position and orientation
                            ):
    # 4d Eye (Camera) Coordinates
    world_pos = np.array([worldPos[0], worldPos[1], worldPos[2], 1.])
    camera_pos = np.matmul(ViewMatrix, world_pos)

    # 4d Homogeneous Clip Coordinates
    clip_pos = np.matmul(ProjectionMatrix, camera_pos)

    # Transform into normalised device coordinates
    # divide w
    ndc_pos = clip_pos
    if clip_pos[3] != 0:
        ndc_pos = clip_pos / clip_pos[3]

    # screen pos
    screen_x = (ndc_pos[0] + 1.0) * float(screenWidth) / 2.0
    screen_y = (ndc_pos[1] - 1.0) * float(screenHeight) / -2.0
    screen_pos = [screen_x, screen_y]

    return screen_pos

# ray v.s. faces, faces number can be 1
def ray_mesh_face_hit_detection(ray_origin, ray_dir, vertices):
    hit = False

    if len(vertices) == 0:
        return []

    face_mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2, 3]])
    hit_point, _, _ = face_mesh.ray.intersects_location([ray_origin], [ray_dir], multiple_hits=False)
    
    if len(hit_point) != 0:
        hit = True

    return hit, hit_point

def ray_plane_hit_detection(plane_point, plane_normal, ray_origin, ray_dir):
    hit = True
    hit_point = None
    t = (np.dot(plane_point, plane_normal) - np.dot(ray_origin, plane_normal)) / np.dot(ray_dir, plane_normal)

    if t < 0. or t > 10.:
        hit = False
        t = None
    else:
        hit_point = ray_origin + ray_dir*t

    return hit, hit_point, t

# vertices
#  0................3
#  .                .
#  .                .
#  .                .
#  .                .
#  1 ...............2
#
def ray_plane_hit_detection_with_boundary(ray_origin, ray_dir, vertices):
    hit = False
    hit_point = None

    vector_x = vertices[3] - vertices[0]
    vector_y = vertices[1] - vertices[0]
    vector_z = np.cross(vector_x, vector_y)

    hit_plane, hit_point, t = ray_plane_hit_detection(plane_point=vertices[0],
                                                      plane_normal=vector_z,
                                                      ray_origin=ray_origin ,
                                                      ray_dir=ray_dir)

    if hit_plane:
        # check if the hit point is inside the rectangle
        check_vector = hit_point - vertices[0]

        x_length = np.sqrt(np.sum(vector_x**2))
        y_length = np.sqrt(np.sum(vector_y**2))
        normalize_x = vector_x/x_length
        normalize_y = vector_y/y_length
        
        # compute the projection on x, y axis
        x_proj_length = np.dot(check_vector, normalize_x)
        y_proj_length = np.dot(check_vector, normalize_y)
        
        if (x_proj_length > 0 and y_proj_length > 0) \
           and (x_proj_length < x_length and y_proj_length < y_length):
           hit = True
        
    return hit, hit_point, t

# ray v.s. point cloud
def ray_point_cloud_hit_detection(ray_origin, ray_dir, point_cloud_positions, hit_radius):
    valid_hit_point_id = []
    point_ts = np.array([ ray_point_hit_detection(point_pos, ray_origin, ray_dir, hit_radius = hit_radius) for point_pos in point_cloud_positions ])
    hit_point_id = np.argmin(point_ts)
    # if no hit 
    if point_ts[hit_point_id] == 1000.:
        return valid_hit_point_id
    else:
        valid_hit_point_id.append(hit_point_id)
        return valid_hit_point_id

def ray_point_hit_detection(point_pos, ray_origin, ray_dir, hit_radius):
    b = np.dot(ray_dir, (ray_origin - point_pos))
    c = np.dot((ray_origin - point_pos), (ray_origin - point_pos)) - hit_radius*hit_radius
    
    check_hit_value = b*b - c
    # if no hit
    if check_hit_value < 0.:
        return 1000.
    elif check_hit_value > 0.:
        t_plus = -b + math.sqrt(check_hit_value)
        t_minus = -b - math.sqrt(check_hit_value)
        return min(t_plus, t_minus)
    else:
        return -b

# ray detection with the surface of a circle
def ray_circle_hit_detection(plane_center, plane_normal, ray_origin, ray_dir, hit_radius, thickness):
    hit = False
    hit_point = None
    
    # implement ray plane hit detection first with the circle plane
    hit_plane, hit_point, t = ray_plane_hit_detection(plane_center, plane_normal, ray_origin, ray_dir)
    if hit_plane:
        # check the distance between the hit point and the center
        vector_center_hit = hit_point - plane_center
        distance = np.sqrt(np.sum(vector_center_hit**2))
        
        if distance < (hit_radius + thickness) and distance > (hit_radius - thickness):
            hit = True

    return hit, hit_point, t

def ray_axis_hit_detection(axis_start, axis_end, axis, ray_origin, ray_dir, thickness):
    hit = False
    hit_point_on_axis = None
    ray_t = None
    
    # find the nearest points of two lines
    # compute the vector that is prependicular to two lines
    n = np.cross(axis, ray_dir)

    # compute a plane by line1 and n, the nearest point on line2 is on the plane
    n1 = np.cross(axis, n)
    # compute a plane by line2 and n, the nearest point on line1 is on the plane
    n2 = np.cross(ray_dir, n)
    
    # find the intersection point of ray1 and the plane of line2
    hit1, hit_point1, t1 = ray_plane_hit_detection(ray_origin, n2, axis_start, axis)
    # find the intersection point of ray2 and the plane of line1
    hit2, hit_point2, t2 = ray_plane_hit_detection(axis_start, n1, ray_origin, ray_dir)
    
    if hit1 and hit2:
        vector_hit1_hit2 = hit_point2 - hit_point1
        vector_hit1_hit2_length = np.sqrt(np.sum(vector_hit1_hit2**2))

        if vector_hit1_hit2_length < thickness and in_segment(hit_point1, axis_start, axis_end):
            hit_point_on_axis = hit_point1
            ray_t = t2
            hit = True

    return hit, hit_point_on_axis, ray_t

# 
def fix_rec(rec):
    print("fix_rec")
    rec = np.array(rec)

    fixed_rec = []
    for corner_id, corner in enumerate(rec):
        last_id = corner_id - 1
        next_id = corner_id + 1

        if corner_id == 0:
            last_id = 3

        elif corner_id == 3:
            next_id = 0

        last_corner = rec[last_id]
        next_corner = rec[next_id]

        last_vector = last_corner - corner
        next_vector = next_corner - corner
        
        last_length = vector_length(last_vector)
        next_length = vector_length(next_vector)

        cos_value = np.dot(normalized_vector(last_vector), normalized_vector(next_vector))

        if cos_value < 0.:
            print('cos')
            if next_length > last_length:
                projection_length = abs(last_length*cos_value)
                point = corner - normalized_vector(next_vector)*projection_length
                fixed_rec.append(point)
            else:
                projection_length = abs(next_length*cos_value)
                point = corner - normalized_vector(last_vector)*projection_length
                fixed_rec.append(point)
        else:
            fixed_rec.append(corner)
    
    return fixed_rec 
    
