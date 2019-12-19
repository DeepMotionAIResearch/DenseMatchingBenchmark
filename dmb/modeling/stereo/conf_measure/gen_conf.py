import torch
import torch.nn as nn


class ConfGenerator(nn.Module):
    """
    Implementation of Confidence ground-truth label generation
    Args:
        gtDisp: tensor, in (Height, Width) or (BatchSize, Height, Width) or (BatchSize, 1, Height, Width) layout
        estDisp: tensor, in (Height, Width) or (BatchSize, Height, Width) or (BatchSize, 1, Height, Width) layout
        theta: a threshold parameter to compare the ground-truth disparity map and the estimated disparity map
    Outputs:
        confidence_gt_label, in (BatchSize, 1, Height, Width) layout
    """

    def __init__(self, theta):
        super(ConfGenerator, self).__init__()

        if not isinstance(theta, (int, float)):
            raise TypeError('(int,float) is expected, got {}'.format(type(theta)))

        self.theta = theta

    def forward(self, estDisp, gtDisp):

        if not torch.is_tensor(gtDisp):
            raise TypeError('ground truth disparity map is expected to be tensor, got {}'.format(type(gtDisp)))
        if not torch.is_tensor(estDisp):
            raise TypeError('estimated disparity map is expected to be tensor, got {}'.format(type(estDisp)))

        assert estDisp.shape == gtDisp.shape

        if gtDisp.dim() == 2:  # single image H x W
            h, w = gtDisp.size(0), gtDisp.size(1)
            gtDisp = gtDisp.view(1, 1, h, w)
            estDisp = estDisp.view(1, 1, h, w)

        if gtDisp.dim() == 3:  # multi image B x H x W
            b, h, w = gtDisp.size(0), gtDisp.size(1), gtDisp.size(2)
            gtDisp = gtDisp.view(b, 1, h, w)
            estDisp = estDisp.view(b, 1, h, w)

        if gtDisp.dim() == 4:
            if gtDisp.size(1) == 1:  # mult image B x 1 x H x W
                self.gtDisp = gtDisp
                self.estDisp = estDisp
            else:
                raise ValueError('2nd dimension size should be 1, got {}'.format(gtDisp.size(1)))

        confidence_gt_label = torch.lt(torch.abs(self.estDisp - self.gtDisp), self.theta).type_as(self.gtDisp)

        return confidence_gt_label
