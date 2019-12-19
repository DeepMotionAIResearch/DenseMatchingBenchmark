import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-12


class _calcConf(nn.Module):
    """
    Implementation of confidence measure base class
    Attention:
        Here, cost_volume is exactly  (- cost_volume), algorithm blow will pick out the max cost_volume as c1
    Args:
        cost_volume: tensor, in [BatchSize, MaxDisparity, Height, Width] layout
    Outputs:
        est_confidence: tensor, in [BatchSize, 1, Height, Width] layout, range in [0,1]
    """

    def __init__(self):
        super(_calcConf, self).__init__()

    def forward(self, cost_volume):
        raise NotImplementedError

    def checkout(self, cost_volume):
        if cost_volume.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(cost_volume.dim()))

    def gradient(self, pred):
        D_dz = pred[:, 1:, :, :, ] - pred[:, :-1, :, :]
        return D_dz

    def get_c1_c2(self, cost_volume):
        self.checkout(cost_volume)

        # grab max disparity
        max_disp = cost_volume.shape[1]

        cost_volume = cost_volume - cost_volume.min(dim=1, keepdim=True)[0]

        pad_cost_volume = F.pad(cost_volume, (0, 0, 0, 0, 1, 0), 'constant', 0)
        cost_volume_grad = self.gradient(pad_cost_volume)

        # for example, a pixel with cost_volume = [1, 2, 3,  2,  1, 2, 3,  1]
        # it's gradient sign is                 = [1, 1, 1, -1, -1, 1, 1, -1]
        # correspond positive gradient is       = [1, 1, 1,  0,  0, 1, 1,  0]
        # correspond negative gradient is       = [0, 0, 0,  1,  1, 0, 0,  1]
        # shift negative gradient left 1        = [0, 0, 1,  1,  0, 0, 1,  1]
        # we append 1 at the end of shifted negative gradient
        # now we can use & with shift negative gradient, positive gradient
        # cost_volume:      [1, 2, 3,  2,  1, 2, 3,  1]
        # positive:         [1, 1, 1,  0,  0, 1, 1,  0]
        # shift negative:   [0, 0, 1,  1,  0, 0, 1,  1]
        # & result:         [0, 0, 1,  0,  0, 0, 1,  0]
        # the position of 1 in & result is correspond to the convex peak of cost_volume
        cost_volume_grad_pos = (cost_volume_grad > 0)
        cost_volume_grad_neg = (cost_volume_grad < 0)
        cost_volume_grad_neg_shift = cost_volume_grad_neg[:, 1:, :, :]
        cost_volume_grad_neg_shift = F.pad(
            cost_volume_grad_neg_shift, (0, 0, 0, 0, 0, 1), 'constant', 1
        ).type_as(cost_volume_grad_pos)
        local_max_index = cost_volume_grad_neg_shift & cost_volume_grad_pos

        # pick out local max
        local_max_value = cost_volume * local_max_index.type_as(cost_volume)

        # along local max, Maximum is correspond to c1
        c1 = local_max_value.max(dim=1, keepdim=True)[0]

        # in order to pick out c2, we have to remove c1 first
        max_value = c1.repeat(1, max_disp, 1, 1)
        max_value_mask = torch.ge(local_max_value, max_value)
        local_max_value_mv_peak = local_max_value * (1.0 - max_value_mask.type_as(local_max_value))
        c2 = local_max_value_mv_peak.max(dim=1, keepdim=True)[0]
        return c1, c2


class pkrConf(_calcConf):
    """ Peak Ratio confidence """

    def __init__(self):
        super(pkrConf, self).__init__()

    def forward(self, cost_volume):
        c1, c2 = self.get_c1_c2(cost_volume)

        # Our cost_volume is exactly equal to (- real meaning CostVolume)
        est_confidence = (c2 + eps) / (c1 + eps)
        est_confidence = est_confidence.abs()
        est_confidence = (1.0 - est_confidence)

        assert (est_confidence.min() >= 0 and est_confidence.max() <= 1.0)

        return est_confidence


class apkrConf(_calcConf):
    """ Average Peak Ratio confidence """

    def __init__(self, kernel_size):
        super(apkrConf, self).__init__()
        self.conv = nn.Conv2d(
            1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size // 2), bias=False
        )

        self.conv.weight.data.fill_(1.0 / (kernel_size ** 2))
        self.conv.weight.data.requires_grad = False

    def forward(self, cost_volume):
        c1, c2 = self.get_c1_c2(cost_volume)

        # Our cost_volume is exactly equal to (- real meaning CostVolume)
        est_confidence = (c2 - eps) / (c1 + eps)
        est_confidence = est_confidence.abs()
        est_confidence = 1.0 - est_confidence

        est_confidence = self.conv(est_confidence).clamp(0.0, 1.0)

        return est_confidence


class nlmConf(_calcConf):
    """ Non Linear Margin """

    def __init__(self, sigma=2.0):
        super(nlmConf, self).__init__()
        self.sigma = sigma

    def forward(self, cost_volume):
        c1, c2 = self.get_c1_c2(cost_volume)
        est_confidence = (-(c2 - c1) / self.sigma ** 2).exp()

        return est_confidence
