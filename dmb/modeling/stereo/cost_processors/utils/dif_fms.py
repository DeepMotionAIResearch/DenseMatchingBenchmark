import torch
import torch.nn.functional as F

from dmb.modeling.stereo.layers.inverse_warp_3d import inverse_warp_3d


def dif_fms(reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1, disp_sample=None,
            normalize=False, p=1.0):
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
        dif_fm, (Tensor): the formed cost volume, in [BatchSize, Channel, disp_sample_number, Height, Width] layout

    """
    device = reference_fm.device
    N, C, H, W = reference_fm.shape

    end_disp = start_disp + max_disp - 1
    disp_sample_number = (max_disp + dilation - 1) // dilation
    disp_index = torch.linspace(start_disp, end_disp, disp_sample_number)

    dif_fm = torch.zeros(N, C, disp_sample_number, H, W).to(device)
    idx = 0
    for i in disp_index:
        i = int(i) # convert torch.Tensor to int, so that it can be index
        if i > 0:
            dif_fm[:, :, idx, :, i:] = reference_fm[:, :, :, i:] - target_fm[:, :, :, :-i]
        elif i == 0:
            dif_fm[:, :, idx, :, :] = reference_fm - target_fm
        else:
            dif_fm[:, :, idx, :, :i] = reference_fm[:, :, :, :i] - target_fm[:, :, :, abs(i):]
        idx = idx + 1

    dif_fm = dif_fm.contiguous()
    return dif_fm


def fast_dif_fms(reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1, disp_sample=None,
                 normalize=False, p=1.0,):
    device = reference_fm.device
    B, C, H, W = reference_fm.shape

    if disp_sample is None:
        end_disp = start_disp + max_disp - 1

        disp_sample_number = (max_disp + dilation - 1) // dilation
        D = disp_sample_number

        # generate disparity samples, in [B,D, H, W] layout
        disp_sample = torch.linspace(start_disp, end_disp, D)
        disp_sample = disp_sample.view(1, D, 1, 1).expand(B, D, H, W).to(device).float()

    else:  # direct provide disparity samples
        # the number of disparity samples
        D = disp_sample.shape[1]

    # expand D dimension
    dif_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
    dif_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

    # shift reference feature map with disparity through grid sample
    # shift target feature according to disparity samples
    dif_target_fm = inverse_warp_3d(dif_target_fm, -disp_sample, padding_mode='zeros')

    # mask out features in reference
    dif_reference_fm = dif_reference_fm * (dif_target_fm > 0).type_as(dif_reference_fm)

    # [B, C, D, H, W)
    dif_fm = dif_reference_fm - dif_target_fm

    if normalize:
        # [B, D, H, W]
        dif_fm = torch.norm(dif_fm, p=p, dim=1, keepdim=False)

    return dif_fm


DIF_FUNCS = dict(
    default=dif_fms,
    fast_mode=fast_dif_fms,
)
