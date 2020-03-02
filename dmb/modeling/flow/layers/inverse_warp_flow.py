import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_warp_flow(img, flow, padding_mode='zeros'):
    """
    Args:
        img (Tensor): the source image (where to sample pixels) -- [B, C, H, W]
        flow (Tensor): flow map of the target image -- [B, 2, H, W]
        padding_mode (str): padding mode, default is zero padding
    Returns:
        projected_img (Tensor): source image warped to the target image -- [B, C, H, W]
    """
    b, _, h, w = flow.size()

    # [1, H, W]  copy 0-height for w times : y coord
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).float()
    # [1, H, W]  copy 0-width for h times  : x coord
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).float()

    pixel_coords = torch.stack((j_range, i_range), dim=1).float().to(flow.device)  # [1, 2, H, W]
    batch_pixel_coords = pixel_coords.expand(b, 2, h, w).contiguous().view(b, 2, -1)  # [B, 2, H*W]

    X = batch_pixel_coords[:, 0, :] + flow[:, 0, :].contiguous().view(b, -1)  # [B, H*W]
    Y = batch_pixel_coords[:, 1, :] + flow[:, 1, :].contiguous().view(b, -1)  # [B, H*W]

    X_norm = 2 * X / (w - 1) - 1
    Y_norm = 2 * Y / (h - 1) - 1

    # If grid has values outside the range of [-1, 1], the corresponding outputs are handled as defined by padding_mode.
    # Details please refer to torch.nn.functional.grid_sample
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    pixel_coords = pixel_coords.view(b, h, w, 2)  # [B, H, W, 2]

    projected_img = torch.nn.functional.grid_sample(img, pixel_coords, padding_mode=padding_mode)

    return projected_img
