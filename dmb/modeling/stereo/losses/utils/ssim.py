import torch
import torch.nn.functional as F


def SSIM(x, y, mask=None, C1=0.01 ** 2, C2=0.03 ** 2):
    """
    Calculate the SSIM between two given tensor.
    Details please refer to https://en.wikipedia.org/wiki/Structural_similarity
    Args:
        x (torch.Tensor): in [BatchSize, Channels, Height, Width] layout
        y (torch.Tensor): in [BatchSize, Channels, Height, Width] layout
        mask (None or torch.Tensor): the mask of valid index, in [BatchSize, Channels, Height, Width] layout
        C1 (double or int): a variable to stabilize the division with weak denominator
        C2 (double or int): a variable to stabilize the division with weak denominator
    Outputs:
        (double): the average difference between x and y, value ranges from [0, 1]
    """

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x_sq
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    if mask is not None:
        SSIM = SSIM[mask]

    # Here, we calculate the difference between x and y, and limit its value in [0,1]
    return torch.clamp((1 - SSIM) / 2, 0, 1).mean()
