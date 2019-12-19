import torch
import torch.nn as nn

eps = 1e-5


class _CostVolumeNorm(nn.Module):
    """
        Normalize Cost Volume
        Args:
            dim (int): which dim to apply normalization operation, default dim is for the cost dim.
            affine (bool): whether the parameters are learnable, default is True
            weight (float): weight for cost re-range
            bias (float): bias for cost
        Shape:
            - Input: :math:`(N, *)`
            - Output: :math:`(N, *)` (same shape as input)
    """

    def __init__(self, dim=1, affine=True, weight=1, bias=0):
        super(_CostVolumeNorm, self).__init__()
        self.dim = dim
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(data=torch.Tensor(1), requires_grad=True)
            self.bias = nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        else:
            self.weight = nn.Parameter(data=torch.Tensor(1), requires_grad=False)
            self.bias = nn.Parameter(data=torch.Tensor(1), requires_grad=False)

        # init weight and bias
        self.weight.data.fill_(weight)
        self.bias.data.fill_(bias)

    def forward(self, input):
        raise NotImplementedError


class RangeNorm(_CostVolumeNorm):
    def __init__(self, dim=1, affine=True, weight=1, bias=0):
        super(RangeNorm, self).__init__(dim=dim, affine=affine, weight=weight, bias=bias)

    def forward(self, input):
        # compute mean value
        mean = input.min(dim=self.dim, keepdim=True)[0]
        # compute margin
        var = input.max(dim=self.dim, keepdim=True)[0] - input.min(dim=self.dim, keepdim=True)[0]
        # normalize
        normalized_input = (input - mean) / (var + eps)
        # apply weight and bias
        output = normalized_input * self.weight + self.bias

        return output


class VarNorm(_CostVolumeNorm):
    def __init__(self, dim=1, affine=True, weight=1, bias=0):
        super(VarNorm, self).__init__(dim=dim, affine=affine, weight=weight, bias=bias)

    def forward(self, input):
        # compute mean value
        mean = input.mean(dim=self.dim, keepdim=True)
        # compute var value
        var = input.var(dim=self.dim, keepdim=True)
        # normalize
        normalized_input = (input - mean).abs() / (var + eps)
        # apply weight and bias
        output = normalized_input * self.weight + self.bias

        return output


class StdNorm(_CostVolumeNorm):
    def __init__(self, dim=1, affine=True, weight=1, bias=0):
        super(StdNorm, self).__init__(dim=dim, affine=affine, weight=weight, bias=bias)

    def forward(self, input):
        # compute mean value
        mean = input.mean(dim=self.dim, keepdim=True)
        # compute var value
        var = input.std(dim=self.dim, keepdim=True)
        # normalize
        normalized_input = (input - mean).abs() / (var + eps)
        # apply weight and bias
        output = normalized_input * self.weight + self.bias

        return output


class SigmoidNorm(_CostVolumeNorm):
    def __init__(self, dim=1, affine=True, weight=1, bias=0):
        super(SigmoidNorm, self).__init__(dim=dim, affine=affine, weight=weight, bias=bias)

    def forward(self, input):
        # normalize
        normalized_input = torch.sigmoid(input)
        # apply weight and bias
        output = normalized_input * self.weight + self.bias

        return output
