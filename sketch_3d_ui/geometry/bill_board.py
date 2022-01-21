from sketch_3d_ui.geometry.plane import Plane

class BillBoard(Plane):
    def __init__(self):
        super(BillBoard, self).__init__()
        self.bill_board_size = None

    def init_bill_board(self, 
                        left_top,
                        camera_up_world_space,
                        camera_right_world_space,
                        bill_board_size=0.03):

        self.bill_board_size = bill_board_size
        
        v0 = left_top
        v1 = v0 - camera_up_world_space*self.bill_board_size
        v2 = v1 + camera_right_world_space*self.bill_board_size
        v3 = v2 + camera_up_world_space*self.bill_board_size
        self.bounding_rec_3d = [v0, v1, v2, v3]
