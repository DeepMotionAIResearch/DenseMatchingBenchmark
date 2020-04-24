import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_correlation_sampler import SpatialCorrelationSampler

def correlation1d_cost(reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1, disp_sample=None,
                       kernel_size=1, stride=1, padding=0, dilation_patch=1,):
    # for a pixel of left image at (x, y), it will calculates correlation cost volume
    # with pixel of right image at (xr, y), where xr in [x-max_disp, x+max_disp]
    # but we only need the left half part, i.e., [x-max_disp, 0]
    correlation_sampler = SpatialCorrelationSampler(patch_size=(1, max_disp * 2 - 1),
                                                    kernel_size=kernel_size,
                                                    stride=stride, padding=padding,
                                                    dilation_patch=dilation_patch)
    # [B, 1, max_disp*2-1, H, W]
    out = correlation_sampler(reference_fm, target_fm)

    # [B, max_disp*2-1, H, W]
    out = out.squeeze(1)

    # [B, max_disp, H, W], grad the left half searching part
    out = out[:, :max_disp, :, :]

    cost = F.leaky_relu(out, negative_slope=0.1, inplace=True)

    return cost

COR_FUNCS = dict(
    default=correlation1d_cost,
)
