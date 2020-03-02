import torch
import torch.nn as nn

class HMPredictor(nn.Module):
    """
    Args:
        in_planes, (int): in channels of input
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        cost, (tensor): cost volume in [BatchSize, Channels, Height, Width] layout

    Outputs:
        flow, (tensor): predicted flow in [BatchSize, 2, Height, Width] layout
    """
    def __init__(self, in_planes, batch_norm=True):
        super(HMPredictor, self).__init__()
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.predictor = nn.Conv2d(self.in_planes, 2, kernel_size=3, stride=1,
                                   padding=1, dilation=1, bias=True)

    def forward(self, cost):

        flow = self.predictor(cost)

        return flow
