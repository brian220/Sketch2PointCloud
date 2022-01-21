from pyntcloud.geometry.models.plane import Plane as PyntCloudPlane

class Plane:
    def __init__(self):
        self.point = None
        self.normal = None

        self.bounding_rec_2d = []
        self.bounding_rec_3d = []
    
    def init_from_point_normal(self, point, normal):
        self.point = point
        self.normal = normal

    def init_from_point_cloud(self, point_cloud):
        p = PyntCloudPlane()
        p.from_point_cloud(point_cloud)

        self.point = p.point
        self.normal = p.normal
