import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_correlation_sampler import SpatialCorrelationSampler
from dmb.modeling.stereo.layers.inverse_warp_3d import inverse_warp_3d

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


def fast_corr_fms(reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1, disp_sample=None):
    device = reference_fm.device
    B, C, H, W = reference_fm.shape

    if disp_sample is None:
        end_disp = start_disp + max_disp - 1

        disp_sample_number = (max_disp + dilation - 1) // dilation
        D = disp_sample_number

        # generate disparity samples, in [B,D, H, W] layout
        disp_sample = torch.linspace(start_disp, end_disp, D)
        disp_sample = disp_sample.view(1, D, 1, 1).expand(B, D, H, W).to(device).float()

    else: # direct provide disparity samples
        # the number of disparity samples
        D = disp_sample.shape[1]

    # expand D dimension
    concat_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
    concat_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

    # shift target feature according to disparity samples
    # [B, C, D, H, W]
    concat_target_fm = inverse_warp_3d(concat_target_fm, -disp_sample, padding_mode='zeros')

    # mask out features in reference
    # [B, C, D, H, W]
    concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).float()

    # [B, C, D, H, W]
    corr_fms = (concat_reference_fm * concat_reference_fm)

    return corr_fms


COR_FUNCS = dict(
    default=correlation1d_cost,
    fast_mode=fast_corr_fms,
)
