import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.ops import Correlation

class CorrelationCost(nn.Module):
    def __init__(self, max_displacement):
        super(CorrelationCost, self).__init__()

        self.corr = Correlation(pad_size=max_displacement, kernel_size=1,
                                max_displacement=max_displacement,
                                stride1=1, stride2=1, corr_multiply=1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, feat1, feat2):
        out = self.corr(feat1, feat2)
        out = self.relu(out)

        return out

