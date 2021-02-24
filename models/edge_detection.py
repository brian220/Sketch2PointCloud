import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian


class EdgeDetector(nn.Module):
    '''
    Detect edges from 2d projection
    args:
        img: input 2d projection, size: (BS, 1, H, W)
    returns:
        normalize_grad_mag: normalized edge value between [0, 1], size: (BS, 1, H, W)
    '''

    def __init__(self, cfg):
        super(EdgeDetector, self).__init__()
        self.cfg = cfg

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))


    def forward(self, img):
        blur_horizontal = self.gaussian_filter_horizontal(img)
        blurred_img = self.gaussian_filter_vertical(blur_horizontal)

        grad_x = self.sobel_filter_horizontal(blurred_img)
        grad_y = self.sobel_filter_vertical(blurred_img)

        # compute thick edges
        epsilon = 1.e-8
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + epsilon)
        
        # devide the max value to move the value between [0, 1]
        B, C, H, W = grad_mag.shape
        grad_mag_ = grad_mag.view(B, C, -1)
        grad_mag_max = grad_mag_.max(dim=2)[0].unsqueeze(2).unsqueeze(2)
        normalize_grad_mag = (grad_mag/grad_mag_max).view(*grad_mag.shape)

        return normalize_grad_mag