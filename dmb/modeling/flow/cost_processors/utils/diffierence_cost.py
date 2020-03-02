import torch
import torch.nn as nn

class DifferenceCost(nn.Module):
    def __init__(self, max_displacement):
        super(DifferenceCost, self).__init__()

        self.max_displacement = max_displacement
        self.sample_number = (max_displacement * 2 + 1)**2
        self.pad = nn.ConstantPad2d(padding=max_displacement, value=0)

    def forward(self, reference_fm, target_fm):
        B, C, H, W = reference_fm.shape
        device = reference_fm.device


        cost = torch.zeros(B, self.sample_number, H, W).to(device)
        right = self.pad(target_fm)
        idx = 0
        for i in range(2*self.max_displacement+1):
            for j in range(2*self.max_displacement+1):
                valid_mask = (right[:, :, i:H+i, j:W+j] != 0).float()
                diff = reference_fm * valid_mask - right[:, :, i:H+i, j:W+j]
                cost[:, idx:idx+1, :, :] = torch.norm(diff, p=2, dim=1, keepdim=True)
                idx = idx + 1

        cost = cost.contiguous()

        return cost
