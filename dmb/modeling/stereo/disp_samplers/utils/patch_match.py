# ---------------------------------------------------------------------------
# DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch
#
# This is a reimplementation of patch match which is written by Shivam Duggal
#
# Original code: https://github.com/uber-research/DeepPruner
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.inverse_warp_3d import inverse_warp_3d


class DisparityInitialization(nn.Module):
    """
    PatchMatch Initialization Block
    Description:
        Rather than allowing each sample / particle to reside in the full disparity space,
        we divide the search space into 'disparity_sample_number' intervals, and force the
        i-th particle to be in a i-th interval. This guarantees the diversity of the
        particles and helps improve accuracy for later computations.

        As per implementation,
        this function divides the complete disparity search space into multiple intervals.

    Args:
        disparity_sample_number, (int): Number of disparity samples to be generated
                                        between min and max disparity,
                                        but exclude the  min and max disparity,
                                        default 10.

    Inputs:
        min_disparity, (tensor): Min Disparity of the disparity search range,
                       in [BatchSize, 1, Height, Width] layout
        max_disparity, (tensor): Max Disparity of the disparity search range,
                       in [BatchSize, 1, Height, Width] layout

    Outputs:
        disparity_sample_noise, (tensor): random value between 0-1.
                                Represents offset of the from the interval_min_disparity,
                                in [BatchSize, disparity_sample_number, Height, Width] layout
        interval_min_disparity, (tensor): the sampled disparity candidates,
                                i.e., the minimum disparity in each interval,
                                in [BatchSize, disparity_sample_number, Height, Width] layout
        disparity_sample_interval, (int): 1.0 / disparity_sample_number

    """
    def __init__(self, disparity_sample_number=12):
        super(DisparityInitialization, self).__init__()
        self.disparity_sample_number = disparity_sample_number


    def forward(self, min_disparity, max_disparity):

        device = min_disparity.device

        B = min_disparity.shape[0]
        H, W = min_disparity.shape[-2:]

        # to get 'disparity_sample_number' samples, and except the min, max disparity,
        # it means divide [min, max] into 'disparity_sample_number + 1' segments
        # disparity sample interval between near disparity samples
        disparity_sample_interval = 1.0 / (self.disparity_sample_number + 1)

        # Generate noise ranged in [0, 1], which submits to standard normal distribution
        # As each pixel should have its own disparity particles, rather than uniform for each pixel.
        # In [B, disparity_sample_number, H, W] layout
        disparity_sample_noise = min_disparity.new_empty(size=(B, self.disparity_sample_number, H, W), device=device).uniform_(0, 1)

        # the index for each sampled disparity candidates,
        # e.g., n = disparity_sample_number + 1, index = [1/n, 2/n, ..., (n-1)/n]
        # in [B, disparity_sample_number, H, W] layout
        index = torch.arange(1, (self.disparity_sample_number + 1), 1).float() / (self.disparity_sample_number + 1)
        index = index.view(1, self.disparity_sample_number, 1, 1).to(device)
        disparity_sample_index = index.expand(B, self.disparity_sample_number, H, W).type_as(min_disparity)

        # the sampled disparity candidates, i.e., the minimum disparity in each interval
        # in [B, disparity_sample_number, H, W] layout
        interval_min_disparity = min_disparity + (max_disparity - min_disparity) * disparity_sample_index

        return disparity_sample_noise, interval_min_disparity, disparity_sample_interval


class Propagation(nn.Module):
    """
    PatchMatch Propagation Block
    Description:
        Particles from adjacent pixels are propagated together through convolution with a
        pre-defined one-hot filter pattern, which en-codes the fact that we allow each pixel
        to propagate particles to its 4-neighbours.

        As per implementation,
        the complete disparity search range is discretized into intervals
        in DisparityInitialization() function.
        Now, propagation of samples from neighbouring pixels, is to be done in per interval.
        This implies that after propagation,
        number of disparity samples per pixel = (propagation_filter_size x number_of_intervals)

    Args:
        propagation_filter_size, (int): the filter size in propagation, default 3.

    Inputs:
        disparity_samples, (tensor): The disparity samples for each pixel,
                                     in [BatchSize, disparity_sample_number, Height, Width] layout
        propagation_type (str): In order to be memory efficient,
                                we use separable convolutions for propagation.
                                default: "horizontal"
    Outputs:
        aggregated_disparity_samples: Disparity Samples aggregated from the neighbours,
                in [BatchSize, disparity_sample_number * propagation_filter_size, Height, Width] layout

            """
    def __init__(self, propagation_filter_size=3):
        super(Propagation, self).__init__()
        self.propagation_filter_size = propagation_filter_size

    def forward(self, disparity_samples, propagation_type="horizontal"):

        device = disparity_samples.device

        # [B, 1, disparity_sample_number, H, W]
        disparity_samples = disparity_samples.unsqueeze(1)

        # aggregate information from neighbours
        # by integrating neighbours' disparity samples as own,
        # the number of disparity samples per pixel
        # equal to (propagation_filter_size x disparity_sample_number)
        kernel_size = self.propagation_filter_size
        if propagation_type is "horizontal":
            # when kernel_size=3, [0, 1, 2] -> [0, 1, 2, 0, 1, 2, 0, 1, 2]
            # [kernel_size, 1, 1, 1, kernel_size], in Height dimension
            index = torch.arange(0, kernel_size, device=device).\
                repeat(kernel_size).view(kernel_size, 1, 1, 1, kernel_size)


            # [kernel_size, 1, 1, 1, kernel_size], [1, 0, 0], [0, 1, 0], [0, 0, 1]
            one_hot_filter = torch.zeros_like(index).scatter_(0, index, 1).float()

            # weight of shape (out_channels, in_channels, kD, kH, kW), i.e. (3, 1, 1, 1, 3)
            # the effect is neighbors' disparity samples stored in Channels dimension
            # therefore, the shape extend from [B, 1, disparity_sample_number, H, W]
            # to [B, propagation_filter_size, disparity_sample_number, H, W]
            aggregated_disparity_samples = F.conv3d(disparity_samples,
                                                    weight=one_hot_filter,
                                                    padding=(0, 0, kernel_size // 2))

        else: # 'vertical'
            # when kernel_size=3, [0, 1, 2] -> [0, 1, 2, 0, 1, 2, 0, 1, 2]
            # [kernel_size, 1, 1, kernel_size, 1], in Width dimension
            index = torch.arange(0, kernel_size, device=device). \
                repeat(kernel_size).view(kernel_size, 1, 1, kernel_size, 1)

            # [kernel_size, 1, 1, kernel_size, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]
            one_hot_filter = torch.zeros_like(index).scatter_(0, index, 1).float()

            # same as the situation in 'horizontal'
            # [B, propagation_filter_size, disparity_sample_number, H, W]
            aggregated_disparity_samples = F.conv3d(disparity_samples,
                                                    weight=one_hot_filter,
                                                    padding=(0, kernel_size // 2, 0))

        # make pixel's sample in the same interval permute nearby each other
        # [s11, s12, s13, s21, s22, s23, s31, s32, s33] -> [s11, s21, s31, s12, s22, s32, ...]
        # [B, propagation_filter_size, disparity_sample_number, H, W] ->
        # [B, disparity_sample_number, propagation_filter_size, H, W]
        aggregated_disparity_samples = aggregated_disparity_samples.permute([0, 2, 1, 3, 4])
        B, C, D, H, W = aggregated_disparity_samples.shape

        # [B, disparity_sample_number * propagation_filter_size, H, W]
        aggregated_disparity_samples = aggregated_disparity_samples.contiguous().view(B, C*D, H, W)

        return aggregated_disparity_samples


class Evaluate(nn.Module):
    """
    PatchMatch Evaluation Block
    Description:
        For each pixel i, matching scores are computed by taking the inner product between the
        left feature and the right feature: score(i,j) = feature_left(i), feature_right(i+disparity(i,j))
        for all candidates j. The best k disparity value for each pixel is carried towards the next iteration.

        As per implementation,
        the complete disparity search range is discretized into intervals in
        DisparityInitialization() function.
        Corresponding to each disparity interval, we have multiple samples to evaluate.
        The best disparity sample per interval is the output of the function.

    Args:
        propagation_filter_size, (int): the filter size in propagation, default 3.
        temperature, (int, float): To raise the max and lower the other values when using soft-max
                    details can refer to: https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/

    Inputs:
        left, (tensor): Left image feature, in [BatchSize, Channels, Height, Width] layout
        right, (tensor): Right image feature, in [BatchSize, Channels, Height, Width] layout
        disparity_samples, (tensor): Disparity Samples to be evaluated. For each pixel, we have
                           ("number of intervals" x "number_of_samples_per_intervals") samples.
                           in [BatchSize, disparity_sample_number * propagation_filter_size, Height, Width] layout
        disparity_sample_noise, (tensor): random value between 0-1.
                                Represents offset of the from the interval_min_disparity,
                           in [BatchSize, disparity_sample_number * propagation_filter_size, Height, Width] layout

    Outputs:
        disparity_samples, (tensor): Evaluated disparity samples, one sample per disparity interval,
                           in [BatchSize, disparity_sample_number, Height, Width] layout
        disparity_sample_noise, (tensor): Evaluated disparity sample noise, one per disparity interval,
                           in [BatchSize, disparity_sample_number, Height, Width] layout

    """
    def __init__(self, propagation_filter_size=3, temperature=7):
        super(Evaluate, self).__init__()
        self.propagation_filter_size = propagation_filter_size
        self.temperature = temperature

    def forward(self, left, right, disparity_samples, disparity_sample_noise):

        B, C, H, W = left.shape
        # disparity_sample_number * propagation_filter_size
        D = disparity_samples.shape[1]

        # warp right image feature according to disparity samples
        # [B, C, disparity_sample_number * propagation_filter_size, H, W]
        left = left.unsqueeze(2).expand(B, C, D, H, W)
        right = right.unsqueeze(2).expand(B, C, D, H, W)
        warped_right = inverse_warp_3d(right, -disparity_samples)

        # matching scores are computed by taking the inner product
        cost_volume = torch.mean(left * warped_right, dim=1) * self. temperature
        cost_volume = cost_volume.view(B, D//self.propagation_filter_size, self.propagation_filter_size, H, W)
        # [B, propagation_filter_size, disparity_sample_number, H, W]
        cost_volume = cost_volume.permute([0, 2, 1, 3, 4])

        disparity_samples = disparity_samples.view(B, D//self.propagation_filter_size, self.propagation_filter_size, H, W)
        # [B, propagation_filter_size, disparity_sample_number, H, W]
        disparity_samples = disparity_samples.permute([0, 2, 1, 3, 4])

        disparity_sample_noise = disparity_sample_noise.view(B, D//self.propagation_filter_size, self.propagation_filter_size, H, W)
        # [B, propagation_filter_size, disparity_sample_number, H, W]
        disparity_sample_noise = disparity_sample_noise.permute([0, 2, 1, 3, 4])

        # pick the most possible matching disparity from neighbours
        # [B, 1, disparity_sample_number, H, W]
        prob_volume = F.softmax(cost_volume, dim=1)

        # [B, disparity_sample_number, H, W]
        disparity_samples = torch.sum(prob_volume * disparity_samples, dim=1)
        # [B, disparity_sample_number, H, W]
        disparity_sample_noise = torch.sum(prob_volume * disparity_sample_noise, dim=1)

        return disparity_samples, disparity_sample_noise


class PatchMatch(nn.Module):
    """
    Differential PatchMatch Block
    Description:
        In this work, we unroll generalized PatchMatch as a recurrent neural network,
        where each unrolling step is equivalent to each iteration of the algorithm.
        This is important as it allow us to train our full model end-to-end.
        Specifically, we design the following layers:
                        - Initialization or Particle Sampling
                        - Propagation
                        - Evaluation

    Args:
        propagation_filter_size, (int): the filter size in propagation, default 3.
        disparity_sample_number, (int): Number of disparity samples to be generated,
                                        including the min and max disparity, default 14.
        iterations, (int): Number of PatchMatch iterations
        temperature, (int, float): To raise the max and lower the other values when using soft-max
                     details can refer to: https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
    Inputs:
        left, (tensor): Left image feature, in [BatchSize, Channels, Height, Width] layout
        right, (tensor): Right image feature, in [BatchSize, Channels, Height, Width] layout
        min_disparity, (tensor): Min Disparity of the disparity search range,
                       in [BatchSize, 1, Height, Width] layout
        max_disparity, (tensor): Max Disparity of the disparity search range,
                       in [BatchSize, 1, Height, Width] layout

    Outputs:
        disparity_samples, (tensor): The generated disparity samples for each pixel,
                                     including the min and max disparity,
                           in [BatchSize, disparity_sample_number, Height, Width] layout
    """
    def __init__(self,
                 propagation_filter_size=3,
                 disparity_sample_number=14,
                 iterations=3,
                 temperature=7):
        super(PatchMatch, self).__init__()
        self.propagation_filter_size = propagation_filter_size
        self.disparity_sample_number = disparity_sample_number
        self.iterations = iterations
        self.temperature = temperature

        # except the min and max disparity, there are 'disparity_sample_number-2' need to be generated
        self.disparity_initialization = DisparityInitialization(disparity_sample_number-2)
        self.propagation = Propagation(propagation_filter_size=propagation_filter_size)
        self.evaluate = Evaluate(propagation_filter_size=propagation_filter_size,
                                 temperature=temperature)

    def forward(self, left, right, min_disparity, max_disparity):

        device = left.device
        B = min_disparity.shape[0]
        H, W = min_disparity.shape[-2:]

        # Initialize patch match
        # disparity_sample_noise: random value between 0-1.
        #                         Represents offset of the from the interval_min_disparity,
        #                         in [B, disparity_sample_number, H, W] layout
        # interval_min_disparity: the minimum disparity in each interval,
        #                         in [B, disparity_sample_number, H, W] layout
        # disparity_sample_interval: 1.0 / disparity_sample_number, in [1] layout
        disparity_sample_noise, interval_min_disparity, disparity_sample_interval = self.disparity_initialization(
            min_disparity, max_disparity
        )

        # [B, disparity_sample_number, H, W] -> [B, disparity_sample_number, propagation_filter_size, H, W]
        interval_min_disparity = interval_min_disparity.unsqueeze(2).repeat(1, 1, self.propagation_filter_size, 1, 1)
        # [B, (disparity_sample_number-2) * propagation_filter_size, H, W], exclude min and max disparity sample
        interval_min_disparity = interval_min_disparity.view(B, (self.disparity_sample_number-2) * self.propagation_filter_size, H, W)

        # propagation -> evaluation
        disparity_samples = None
        for prop_iter in range(self.iterations):
            # it's equal to propagate in disparity_sample_noise or real disparity_samples
            # integrate information from near pixels through horizontal propagation
            # [B, disparity_sample_number * propagation_filter_size, H, W]
            disparity_sample_noise = self.propagation(disparity_sample_noise,
                                                      propagation_type="horizontal")

            # noise in [0, 1] * [(max -min) / sample number] + minimum in each interval
            # [B, disparity_sample_number * propagation_filter_size, H, W]
            disparity_samples = (max_disparity - min_disparity) * disparity_sample_interval * \
                                disparity_sample_noise + interval_min_disparity

            # [B, disparity_sample_number, H, W], [B, disparity_sample_number, H, W]
            disparity_samples, disparity_sample_noise = self.evaluate(left, right, disparity_samples,
                                                                      disparity_sample_noise)

            # integrate information from near pixels through vertical propagation
            # [B, disparity_sample_number * propagation_filter_size, H, W]
            disparity_sample_noise = self.propagation(disparity_sample_noise,
                                                            propagation_type="vertical")

            # [B, disparity_sample_number * propagation_filter_size, H, W]
            disparity_samples = (max_disparity - min_disparity) * disparity_sample_interval * \
                                disparity_sample_noise + interval_min_disparity

            # [B, disparity_sample_number, H, W], [B, disparity_sample_number, H, W]
            disparity_samples, disparity_sample_noise = self.evaluate(left, right, disparity_samples,
                                                                      disparity_sample_noise)
        # not only the disparity samples generated in (min, max),
        # the min and max are also the disparity samples
        disparity_samples = torch.cat((min_disparity, disparity_samples, max_disparity), dim=1)

        return disparity_samples