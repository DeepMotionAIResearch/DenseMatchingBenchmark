from torch.nn.modules.module import Module
import numpy as np
from torch.autograd import Variable
from .function import (SgaFunction,
                       LgaFunction, Lga2Function, Lga3Function,
                       Lga3dFunction, Lga3d2Function, Lga3d3Function)


class SGA(Module):
    def __init__(self):
        super(SGA, self).__init__()

    def forward(self, input, g0, g1, g2, g3):
        result = SgaFunction()(input, g0, g1, g2, g3)
        return result


class LGA3D3(Module):
    def __init__(self, radius=2):
        super(LGA3D3, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga3d3Function(self.radius)(input1, input2)
        return result


class LGA3D2(Module):
    def __init__(self, radius=2):
        super(LGA3D2, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga3d2Function(self.radius)(input1, input2)
        return result


class LGA3D(Module):
    def __init__(self, radius=2):
        super(LGA3D, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga3dFunction(self.radius)(input1, input2)
        return result


class LGA3(Module):
    def __init__(self, radius=2):
        super(LGA3, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga3Function(self.radius)(input1, input2)
        return result


class LGA2(Module):
    def __init__(self, radius=2):
        super(LGA2, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga2Function(self.radius)(input1, input2)
        return result


class LGA(Module):
    def __init__(self, radius=2):
        super(LGA, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = LgaFunction(self.radius)(input1, input2)
        return result

