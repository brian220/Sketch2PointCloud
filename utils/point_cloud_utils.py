import torch

class Scale(torch.nn.Module):
    '''
    Scale GT and predicted PCL to a bounding cube with edges from [-0.5,0.5] in
    each axis. 
    args:
            gt_pc: float, (BS,N_PTS,3); GT point cloud
            pr_pc: float, (BS,N_PTS,3); predicted point cloud
    returns:
            gt_scaled: float, (BS,N_PTS,3); scaled GT point cloud
            pred_scaled: float, (BS,N_PTS,3); scaled predicted point cloud
    '''
    def __init__(self, cfg):
        super(Scale, self).__init__()
        self.cfg = cfg
        
    def forward(self, pr_pc, gt_pc):
        
        pr = pr_pc.type(torch.FloatTensor)
        gt = gt_pc.type(torch.FloatTensor)
        
        min_pr = torch.stack([torch.min(pr[:,:,i], dim=1)[0] for i in range(3)])
        max_pr = torch.stack([torch.max(pr[:,:,i], dim=1)[0] for i in range(3)])
        min_gt = torch.stack([torch.min(gt[:,:,i], dim=1)[0] for i in range(3)])
        max_gt = torch.stack([torch.max(gt[:,:,i], dim=1)[0] for i in range(3)])
        
        length_pr = torch.abs(max_pr - min_pr)
        length_gt = torch.abs(max_gt - min_gt)
        
        diff_pr = torch.max(length_pr, dim=0, keepdim=True)[0] - length_pr
        diff_gt = torch.max(length_gt, dim=0, keepdim=True)[0] - length_gt
        
        new_min_pr = torch.stack([min_pr[i,:] - diff_pr[i,:]/2. for i in range(3)])
        new_max_pr = torch.stack([max_pr[i,:] + diff_pr[i,:]/2. for i in range(3)])
        new_min_gt = torch.stack([min_gt[i,:] - diff_gt[i,:]/2. for i in range(3)])
        new_max_gt = torch.stack([max_gt[i,:] + diff_gt[i,:]/2. for i in range(3)])
        
        size_pr = torch.max(length_pr, dim=0)[0]
        size_gt = torch.max(length_gt, dim=0)[0]
        
        scaling_factor_pr = 1. / size_pr
        scaling_factor_gt = 1. / size_gt
    
        box_min = torch.ones_like(new_min_gt) * -0.5
        
        adjustment_factor_pr = box_min - scaling_factor_pr * new_min_pr
        adjustment_factor_gt = box_min - scaling_factor_gt * new_min_gt
        
        pr_scaled = (pr.permute(2, 1, 0) * scaling_factor_pr).permute(2, 1, 0) + torch.reshape(adjustment_factor_pr.permute(1, 0), (-1, 1, 3))
        gt_scaled = (gt.permute(2, 1, 0) * scaling_factor_gt).permute(2, 1, 0) + torch.reshape(adjustment_factor_gt.permute(1, 0), (-1, 1, 3))
    
        return pr_scaled, gt_scaled


class Scale_one(torch.nn.Module):
    '''
    Scale GT PCL to a bounding cube with edges from [-0.5,0.5] in
    each axis. 
    args:
            gt_pc: float, (BS,N_PTS,3); GT point cloud
    returns:
            gt_scaled: float, (BS,N_PTS,3); scaled GT point cloud
    '''
    def __init__(self, cfg):
        super(Scale_one, self).__init__()
        self.cfg = cfg
        
    def forward(self, gt_pc):
        
        gt = gt_pc.type(torch.FloatTensor)
        
        min_gt = torch.stack([torch.min(gt[:,:,i], dim=1)[0] for i in range(3)])
        max_gt = torch.stack([torch.max(gt[:,:,i], dim=1)[0] for i in range(3)])
        
        length_gt = torch.abs(max_gt - min_gt)
        
        diff_gt = torch.max(length_gt, dim=0, keepdim=True)[0] - length_gt
        
        new_min_gt = torch.stack([min_gt[i,:] - diff_gt[i,:]/2. for i in range(3)])
        new_max_gt = torch.stack([max_gt[i,:] + diff_gt[i,:]/2. for i in range(3)])
        
        size_gt = torch.max(length_gt, dim=0)[0]
        
        scaling_factor_gt = 1. / size_gt
    
        box_min = torch.ones_like(new_min_gt) * -0.5
        
        adjustment_factor_gt = box_min - scaling_factor_gt * new_min_gt
        
        gt_scaled = (gt.permute(2, 1, 0) * scaling_factor_gt).permute(2, 1, 0) + torch.reshape(adjustment_factor_gt.permute(1, 0), (-1, 1, 3))
    
        return gt_scaled