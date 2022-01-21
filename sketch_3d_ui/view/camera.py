import math

class Camera_Z_UP:
    def __init__(self, theta, phi, distance):
        self.theta = theta
        self.phi = phi
        self.distance = distance

    def rotate(self, d_theta, d_phi):
        self.theta += d_theta
        self.phi += d_phi

    def zoom(self, d_distance):
        self.distance += d_distance
    
    def get_azi_ele(self):
        azi = self.theta*180./math.pi
        ele = 90. - self.phi*180./math.pi

        return azi, ele
    
    def get_cartesian_camera_pos(self):
        camera_pos = [
                      self.distance*math.cos(self.theta)*math.sin(self.phi), \
                      self.distance*math.sin(self.theta)*math.sin(self.phi), \
                      self.distance*math.cos(self.phi)
                     ]

        return camera_pos

class Camera_Y_UP:
    def __init__(self, theta, phi, distance):
        self.theta = theta
        self.phi = phi
        self.distance = distance

    def rotate(self, d_theta, d_phi):
        self.theta += d_theta
        self.phi += d_phi

    def zoom(self, d_distance):
        self.distance += d_distance
    
    def get_cartesian_camera_pos(self):
        camera_pos = [
                      self.distance*math.cos(self.phi)*math.cos(self.theta),
                      self.distance*math.sin(self.phi),
                      -self.distance*math.cos(self.phi)*math.sin(self.theta)
                     ]

        return camera_pos
        
