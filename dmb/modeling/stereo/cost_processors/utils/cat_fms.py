import torch
import torch.nn.functional as F


def cat_fms(reference_fm, target_fm, max_disp, start_disp=0, dilation=1):
    end_disp = start_disp + max_disp - 1
    disp_sample_number = (max_disp + dilation - 1) // dilation

    device = reference_fm.device
    N, C, H, W = reference_fm.shape
    concat_fm = torch.zeros(N, C * 2, disp_sample_number, H, W).to(device)

    idx = 0
    for i in range(start_disp, end_disp + 1, dilation):
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


def fast_cat_fms(reference_fm, target_fm, max_disp, start_disp=0, dilation=1):
    end_disp = start_disp + max_disp - 1
    disp_sample_number = (max_disp + dilation - 1) // dilation

    device = reference_fm.device
    B, C, H, W = reference_fm.shape
    D = disp_sample_number

    # get mesh grid for each dimension
    grid_d = torch.linspace(0, D - 1, D).view(1, D, 1, 1).expand(B, D, H, W).to(device)
    grid_h = torch.linspace(0, H - 1, H).view(1, 1, H, 1).expand(B, D, H, W).to(device)
    grid_w = torch.linspace(0, W - 1, W).view(1, 1, 1, W).expand(B, D, H, W).to(device)

    # get shift offset, i.e. disparity
    offset = torch.linspace(start_disp, end_disp, D).view(1, D, 1, 1).expand(B, D, H, W).to(device)
    # shift the index of W dimension with offset
    grid_w = grid_w - offset

    # normalize the grid value into [-1, 1]; (0, D-1), (0, H-1), (0, W-1)
    grid_d = (grid_d / (D - 1) * 2) - 1
    grid_h = (grid_h / (H - 1) * 2) - 1
    grid_w = (grid_w / (W - 1) * 2) - 1

    # concatenate the grid_* to [B, D, H, W, 3]
    grid_d = grid_d.unsqueeze(4)
    grid_h = grid_h.unsqueeze(4)
    grid_w = grid_w.unsqueeze(4)
    grid = torch.cat((grid_w, grid_h, grid_d), 4)

    # expand D dimension
    concat_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
    concat_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

    # shift reference feature map with disparity through grid sample
    concat_target_fm = F.grid_sample(concat_target_fm, grid, padding_mode='zeros')

    # mask out features in reference
    concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).type_as(concat_reference_fm)

    # [B, 2C, D, H, W)
    concat_fm = torch.cat((concat_reference_fm, concat_target_fm), dim=1)

    return concat_fm


CAT_FUNCS = dict(
    default=cat_fms,
    fast_mode=fast_cat_fms,
)
