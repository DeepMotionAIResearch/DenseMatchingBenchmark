import torch
import torch.nn.functional as F

from dmb.modeling.stereo.layers.inverse_warp_3d import inverse_warp_3d


def cat_fms(reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1, disp_sample=None):
    """
    Concat left and right in Channel dimension to form the raw cost volume.
    Args:
        max_disp, (int): under the scale of feature used,
            often equals to (end disp - start disp + 1), the maximum searching range of disparity
        start_disp (int): the start searching disparity index, usually be 0
            dilation (int): the step between near disparity index
        dilation (int): the step between near disparity index

    Inputs:
        reference_fm, (Tensor): reference feature, i.e. left image feature, in [BatchSize, Channel, Height, Width] layout
        target_fm, (Tensor): target feature, i.e. right image feature, in [BatchSize, Channel, Height, Width] layout

    Output:
        concat_fm, (Tensor): the formed cost volume, in [BatchSize, Channel*2, disp_sample_number, Height, Width] layout

    """
    device = reference_fm.device
    N, C, H, W = reference_fm.shape

    end_disp = start_disp + max_disp - 1
    disp_sample_number = (max_disp + dilation - 1) // dilation
    disp_index = torch.linspace(start_disp, end_disp, disp_sample_number)

    concat_fm = torch.zeros(N, C * 2, disp_sample_number, H, W).to(device)
    idx = 0
    for i in disp_index:
        i = int(i) # convert torch.Tensor to int, so that it can be index
        if i > 0:
            concat_fm[:, :C, idx, :, i:] = reference_fm[:, :, :, i:]
            concat_fm[:, C:, idx, :, i:] = target_fm[:, :, :, :-i]
        elif i == 0:
            concat_fm[:, :C, idx, :, :] = reference_fm
            concat_fm[:, C:, idx, :, :] = target_fm
        else:
            concat_fm[:, :C, idx, :, :i] = reference_fm[:, :, :, :i]
            concat_fm[:, C:, idx, :, :i] = target_fm[:, :, :, abs(i):]
        idx = idx + 1

    concat_fm = concat_fm.contiguous()
    return concat_fm


def fast_cat_fms(reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1, disp_sample=None):
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
    concat_target_fm = inverse_warp_3d(concat_target_fm, -disp_sample, padding_mode='zeros')

    # mask out features in reference
    concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).float()

    # [B, 2C, D, H, W)
    concat_fm = torch.cat((concat_reference_fm, concat_target_fm), dim=1)

    return concat_fm


CAT_FUNCS = dict(
    default=cat_fms,
    fast_mode=fast_cat_fms,
)
