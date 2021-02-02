#reference: https://github.com/val-iisc/capnet/blob/master/src/proj_codes.py

from __future__ import division
import math
import numpy as np

import torch

class Projector(torch.nn.Module):
    '''
    Project the 3D point cloud to 2D plane
    args:
            xyz: float tensor, (BS,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
            az: float tensor, (BS); azimuthal angle of camera in radians
            el: float tensor, (BS); elevation of camera in radians
            N_PTS: float, (); number of points in point cloud
    returns:
            grid_val: float, (N_batch,H,W); 
                      output silhouette
    '''
    def __init__(self, cfg):
        super(Projector, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.batch_size = cfg.CONST.BATCH_SIZE
        self.n_pts = cfg.CONST.NUM_POINTS
        self.grid_h = cfg.PROJECTION.GRID_H
        self.grid_w = cfg.PROJECTION.GRID_W
        self.sigma_sq = cfg.PROJECTION.SIGMA_SQ


    def forward(self, xyz, az, el):
        # World co-ordinates to camera co-ordinates
        pcl_out_rot = world2cam(xyz, az, el, , batch_size=self.batch_size, N_PTS=self.n_pts)

        # Perspective transform
        pcl_out_persp = perspective_transform(pcl_out_rot, batch_size=self.batch_size)

        # 3D to 2D Projection
        proj_pred = cont_proj(pcl_out_persp, grid_h=self.grid_h, grid_w=self.grid_w, sigma_sq=self.sigma_sq)

        return proj_pred
        

    def cont_proj(pcl, grid_h, grid_w, sigma_sq=0.5):
        '''
        Continuous approximation of Orthographic projection of point cloud
        to obtain Silhouette
        args:
                pcl: float, (N_batch,N_PTS,3); input point cloud
                         values assumed to be in (-1,1)
                grid_h, grid_w: int, ();
                         output depth map height and width
        returns:
                grid_val: float, (N_batch,H,W); 
                          output silhouette
        '''
        x, y, z = pcl.chunk(3, dim=2) # divide to three parts
        pcl_norm = torch.cat([x, y, z], dim=2)
        pcl_xy = torch.cat([x,y], dim=2)
        out_grid = torch.meshgrid(torch.arange(0, grid_h), torch.arange(0, grid_w)) 
        out_grid = [out_grid[0].type(torch.FloatTensor), out_grid[1].type(torch.FloatTensor)]
        
        grid_z = torch.unsqueeze(torch.zeros_like(out_grid[0]), 2) # (H,W,1)
        grid_xyz = torch.cat([torch.stack(out_grid, 2), grid_z], dim=2) # (H,W,3)
        grid_xy = torch.stack(out_grid, 2) # (H,W,2)
        grid_diff = torch.unsqueeze(torch.unsqueeze(pcl_xy, 2), 2) - grid_xy # (BS,N_PTS,H,W,2)
        grid_val = apply_kernel(grid_diff, sigma_sq)  # (BS,N_PTS,H,W,2)
        grid_val = grid_val[:,:,:,:,0]*grid_val[:,:,:,:,1]  # (BS,N_PTS,H,W)
        grid_val = torch.sum(grid_val, dim=1) # (BS,H,W)
        grid_val = torch.tanh(grid_val)
    
        return grid_val


    def apply_kernel(x, sigma_sq=0.5):
        '''
        Get the un-normalized gaussian kernel with point co-ordinates as mean and 
        variance sigma_sq
        args:
                x: float, (BS,N_PTS,H,W,2); mean subtracted grid input 
                sigma_sq: float, (); variance of gaussian kernel
        returns:
                out: float, (BS,N_PTS,H,W,2); gaussian kernel
        '''
        out = (torch.exp(-(x**2)/(2.*sigma_sq)))
        return out
    
    
    def perspective_transform(xyz, batch_size):
        '''
        Perspective transform of pcl; Intrinsic camera parameters are assumed to be
        known (here, obtained using parameters of GT image renderer, i.e. Blender)
        Here, output grid size is assumed to be (64,64) in the K matrix
        TODO: use output grid size as argument
        args:
                xyz: float, (BS,N_PTS,3); input point cloud
                         values assumed to be in (-1,1)
        returns:
                xyz_out: float, (BS,N_PTS,3); perspective transformed point cloud 
        '''
        K = np.array([
                [120., 0., -32.],
                [0., 120., -32.],
                [0., 0., 1.]]).astype(np.float32)
        K = np.expand_dims(K, 0)
        K = np.tile(K, [batch_size,1,1])
        K = torch.from_numpy(K)
    
        xyz_out = torch.matmul(K, xyz.permute(0, 2, 1))
        xy_out = xyz_out[:,:2]/abs(torch.unsqueeze(xyz[:,:,2],1))
        xyz_out = torch.cat([xy_out, abs(xyz_out[:,2:])],dim=1)
    
        return xyz_out.permute(0, 2, 1)
    
    
    def world2cam(xyz, az, el, batch_size, N_PTS=1024):
        '''
        Convert pcl from world co-ordinates to camera co-ordinates
        args:
                xyz: float tensor, (BS,N_PTS,3); input point cloud
                         values assumed to be in (-1,1)
                az: float tensor, (BS); azimuthal angle of camera in radians
                elevation: float tensor, (BS); elevation of camera in radians
                batch_size: int, (); batch size
                N_PTS: float, (); number of points in point cloud
        returns:
                xyz_out: float tensor, (BS,N_PTS,3); output point cloud in camera
                            co-ordinates
        '''
        # Camera origin calculation - az,el,d to 3D co-ord
    
        # Rotation
        rotmat_az=[
                    [torch.ones_like(az),torch.zeros_like(az),torch.zeros_like(az)],
                    [torch.zeros_like(az),torch.cos(az),-torch.sin(az)],
                    [torch.zeros_like(az),torch.sin(az),torch.cos(az)]
                    ]
        rotmat_az = [ torch.stack(x) for x in rotmat_az ]
    
        rotmat_el=[
                    [torch.cos(el),torch.zeros_like(az), torch.sin(el)],
                    [torch.zeros_like(az),torch.ones_like(az),torch.zeros_like(az)],
                    [-torch.sin(el),torch.zeros_like(az), torch.cos(el)]
                    ]
        rotmat_el = [ torch.stack(x) for x in rotmat_el ]
    
        rotmat_az = torch.stack(rotmat_az, 0) # [3,3,B]
        rotmat_el = torch.stack(rotmat_el, 0) # [3,3,B]
        rotmat_az = rotmat_az.permute(2, 0, 1) # [B,3,3]
        rotmat_el = rotmat_el.permute(2, 0, 1) # [B,3,3]
        rotmat = torch.matmul(rotmat_el, rotmat_az)
        
        # Transformation(t)
        # Distance of object from camera - fixed to 2
        d = 2.
        # Calculate translation params
        tx, ty, tz = [0, 0, d]
        
        tr_mat = torch.unsqueeze(torch.tensor([tx, ty, tz]), 0).repeat(batch_size,1) # [B,3]
        tr_mat = torch.unsqueeze(tr_mat,2) # [B,3,1]
        tr_mat = tr_mat.permute(0, 2, 1) # [B,1,3]
        tr_mat = tr_mat.repeat(1, N_PTS, 1) # [B,1024,3]
    
        xyz_out = torch.matmul(rotmat, xyz.permute(0, 2, 1)) - tr_mat.permute(0, 2, 1)
        
        return xyz_out.permute(0, 2, 1)

