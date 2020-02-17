import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.patch_match import PatchMatch


class DisparitySampleRangeHead(nn.Module):
    """
    Determine the disparity sample range.
    Description:
        Generates the disparity search range depending upon the stage it is called.
        If stage is "pre" (Pre-PatchMatch and Pre-ConfidenceRangePredictor), the search range is
        the entire disparity search range.
        If stage is "post" (Post-ConfidenceRangePredictor), then the ConfidenceRangePredictor search range
        is adjusted for maximum efficiency.

    Args:
        max_disp, (int): max disparity search range

    Inputs:
        stage, (str): "pre"(Pre-PatchMatch) or "post"(Post-ConfidenceRangePredictor)
        disparity_sample_number, (int): Number of disparity samples to be generated
        left, (tensor): Left image feature, in [BatchSize, Channels, Height, Width] layout
        min_disparity, (tensor): Min Disparity of the disparity search range,
                       in [BatchSize, 1, Height, Width] layout, default None
        max_disparity, (tensor): Max Disparity of the disparity search range,
                       in [BatchSize, 1, Height, Width] layout, default None

    Outputs:
        min_disparity, (tensor): lower bound of the disparity search range,
                       in [BatchSize, 1, Height, Width] layout
        max_disparity, (tensor): upper bound of the disparity search range,
                       in [BatchSize, 1, Height, Width] layout
    """
    def __init__(self, max_disp):
        super(DisparitySampleRangeHead, self).__init__()
        self.max_disp = max_disp

    def forward(self, stage, disparity_sample_number, left, min_disparity=None, max_disparity=None):
        device = left.device
        B, _, H, W = left.shape

        if stage == 'pre':
            min_disparity = torch.zeros((B, 1, H, W), device=device)
            max_disparity = torch.zeros((B, 1, H, W), device=device) + self.max_disp

        else: # 'post'
            global_min_disparity = torch.min(min_disparity, max_disparity)
            global_max_disparity = torch.max(min_disparity, max_disparity)

            # To promise the interval between near disparity samples: interval >= 1
            # too close disparity sample interval, i.e., interval < 1, makes no sense
            # if (global_max_disparity - global_min_disparity) > disparity_sample_number:
            #     sample uniformly from (global_min_disparity, global_max_disparity)
            # else:
            #     stretch global_min_disparity and global_max_disparity such that
            #     (global_max_disparity - global_min_disparity) == disparity_sample_number

            # get the disparity samples which overflow the expected number of disparity samples
            # if not , clamp to 0
            overflow_disparity_samples = torch.clamp((global_min_disparity + disparity_sample_number
                                                     - global_max_disparity), min=0)
            min_disparity = torch.clamp((global_min_disparity - overflow_disparity_samples) / 2.0, min=0.0, max=self.max_disp)

            max_disparity = torch.clamp((global_max_disparity + overflow_disparity_samples) / 2.0, min=0.0, max=self.max_disp)

        return min_disparity, max_disparity


class UniformSampler(nn.Module):
    """
    Uniform Disparity Sampler
    Description:
        The Confidence Range Predictor predicts a reduced disparity sample range R(i) = [l(i), u(i)]
        for each pixel i. We then, generate disparity samples from this reduced search range
        for Cost Aggregation or second stage of Patch Match.
        From experiments, we found Uniform sampling to work better.

    Args:
        disparity_sample_number, (int): Number of disparity samples to be generated,
                                        including the min and max disparity, default 9.

    Inputs:
        min_disparity, (tensor): Min Disparity of the disparity search range,
                        in [BatchSize, 1, Height, Width] layout, default None
        max_disparity, (tensor): Max Disparity of the disparity search range,
                        in [BatchSize, 1, Height, Width] layout, default None

    Outputs:
        disparity_samples, (tensor): The generated disparity samples for each pixel,
                           in [BatchSize, disparity_sample_number, Height, Width] layout

    """
    def __init__(self, disparity_sample_number=9):
        super(UniformSampler, self).__init__()
        self.disparity_sample_number = disparity_sample_number

    def forward(self, min_disparity, max_disparity):

        device = min_disparity.device

        # to get 'disparity_sample_number' samples, and except the min, max disparity,
        # it means divide [min, max] into 'disparity_sample_number -2 + 1' segments
        sample_index = torch.arange(1.0, self.disparity_sample_number - 2 + 1, 1, device=device).float()
        # [disparity_sample_number - 2, 1, 1]
        sample_index = sample_index.view(self.disparity_sample_number - 2, 1, 1) / (self.disparity_sample_number - 2 + 1)

        # [B, disparity_sample_number - 2, H, W]
        disparity_samples = min_disparity + (max_disparity - min_disparity) * sample_index

        # [B, disparity_sample_number, H, W]
        disparity_samples = torch.cat((min_disparity, disparity_samples, max_disparity), dim=1)

        return disparity_samples


class DeepPrunerSampler(nn.Module):
    """
    The disparity sampler of DeepPruner
    Description:
        Generates "disparity_sample_number" number of disparity samples from the
        search range (min_disparity, max_disparity)
        Samples are generated either uniformly from the search range
        or are generated by using PatchMatch.

    Args:
        max_disp, (int): max disparity search range
        batch_norm (bool): whether use batch normalization layer, default True
        propagation_filter_size, (int): the filter size in propagation, default 3.
        iterations, (int): Number of PatchMatch iterations
        temperature, (int, float): To raise the max and lower the other values when using soft-max
                    details can refer to: https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
        patch_match_disparity_sample_number, (int): Number of disparity samples to be generated by patch match
        uniform_disparity_sample_number, (int): Number of disparity samples to be generated by uniform

    Inputs:
        stage, (str): "pre"(Pre-PatchMatch) using patch match as sampler,
                   or "post"(Post-ConfidenceRangePredictor) using uniform sampler
        left, (tensor): Left image feature, in [BatchSize, Channels, Height, Width] layout
        right, (tensor): Right image feature, in [BatchSize, Channels, Height, Width] layout
        min_disparity, (tensor): Min Disparity of the disparity search range,
                       in [BatchSize, 1, Height, Width] layout
        max_disparity, (tensor): Max Disparity of the disparity search range,
                       in [BatchSize, 1, Height, Width] layout

    Outputs:
        disparity_samples, (tensor): The generated disparity samples for each pixel,
                           in [BatchSize, disparity_sample_number, Height, Width] layout

    """
    def __init__(self, max_disp,
                 batch_norm=True,
                 propagation_filter_size=3,
                 iterations=3,
                 temperature=7,
                 patch_match_disparity_sample_number=14,
                 uniform_disparity_sample_number=9, ):
        super(DeepPrunerSampler, self).__init__()
        self.max_disp = max_disp
        self.batch_norm = batch_norm
        self.propagation_filter_size = propagation_filter_size
        self.iterations = iterations
        self.temperature = temperature
        self.patch_match_disparity_sample_number = patch_match_disparity_sample_number
        self.uniform_disparity_sample_number = uniform_disparity_sample_number

        self.disparity_sample_range = DisparitySampleRangeHead(max_disp=max_disp)

        self.patch_match = PatchMatch(propagation_filter_size=propagation_filter_size,
                                      disparity_sample_number=patch_match_disparity_sample_number,
                                      iterations=iterations,
                                      temperature=temperature)
        self.uniform_sampler = UniformSampler(disparity_sample_number=uniform_disparity_sample_number)

    def forward(self, stage, left, right, min_disparity=None, max_disparity=None):
        if stage == 'pre':
            # [B, 1, H, W], [B, 1, H, W]
            min_disparity, max_disparity = self.disparity_sample_range(stage, self.patch_match_disparity_sample_number,
                                                                       left, min_disparity, max_disparity)
            # [B, disparity_sample_number, H, W]
            disparity_samples = self.patch_match(left, right, min_disparity, max_disparity)

        else: # 'post'
            # [B, 1, H, W], [B, 1, H, W]
            min_disparity, max_disparity = self.disparity_sample_range(stage, self.uniform_disparity_sample_number,
                                                                       left, min_disparity, max_disparity)
            # [B, disparity_sample_number, H, W]
            disparity_samples = self.uniform_sampler(min_disparity, max_disparity)

        return disparity_samples