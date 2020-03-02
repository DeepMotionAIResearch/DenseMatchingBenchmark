import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.inverse_warp import inverse_warp


class InverseWarpLoss(object):
    """
    calculate the inverse warp loss of disparity map, which is weighted by image gradient
        Args:
            weights (list of float or None): weight for each scale of estCost.
            theta (double): the threshold of identity difference for left-right consistency check
            eps (double): a small epsilon approximate to 0
            ssim_weight (double): the loss weight of SSIM
            rms_wegiht (double): the loss weight of root mean square loss
        Inputs:
            estLeftDisp (Tensor or list of Tensor): estimated left disparity map,
                in [BatchSize, 1, Height, Width] layout.
            estRightDisp (Tensor or list of Tensor): estimated right disparity mpa,
                in [BatchSize, 1, Height, Width] layout.
            leftImage (Tensor): in (BatchSize, ..., Height, Width) layout.
            rightImage (None or Tensor): in [BatchSize, ..., Height, Width] layout.
        Outputs:
            weighted_loss_all_level (Tensor): the weighted loss of all levels.
    """

    def __init__(self, weights=None, theta=1.0, eps=1e-6, ssim_weight=0.15, rms_weight=0.85):
        self.weights = weights
        self.theta = theta
        self.eps = eps
        self.ssim_weight = ssim_weight
        self.rms_weight=rms_weight

    def get_per_level_not_occlusion(self, estLeftDisp, estRightDisp):
        assert estLeftDisp.shape == estRightDisp.shape
        leftDisp_fromWarp = inverse_warp(estRightDisp, -estLeftDisp)
        rightDisp_fromWarp = inverse_warp(estLeftDisp, estRightDisp)

        # left and right consistency check
        leftOcclusion = ((torch.abs(leftDisp_fromWarp - estLeftDisp) > self.theta) |
                         (torch.abs(leftDisp_fromWarp) < self.eps))
        rightOcclusion = ((torch.abs(rightDisp_fromWarp - estRightDisp) > self.theta) |
                          (torch.abs(rightDisp_fromWarp) < self.eps))

        # get not occlusion mask
        leftNotOcclusion = (1 - leftOcclusion).type_as(leftOcclusion)
        rightNotOcclusion = (1 - rightOcclusion).type_as(rightOcclusion)

        return leftNotOcclusion, rightNotOcclusion

    def rms(self, est, gt):
        # root mean square
        return torch.sqrt((est - gt) ** 2 + self.eps).mean()

    def loss_per_level(self, estDisp, leftImage, rightImage, mask=None):
        from dmb.modeling.stereo.losses.utils import SSIM
        N, C, H, W = estDisp.shape
        leftImage = F.interpolate(leftImage, (H, W), mode='area')
        rightImage = F.interpolate(rightImage, (H, W), mode='area')

        leftImage_fromWarp = inverse_warp(rightImage, -estDisp)

        if mask is None:
            mask = torch.ones_like(leftImage > 0)
        loss = self.rms_weight * self.rms(leftImage[mask], leftImage_fromWarp[mask])
        loss += self.ssim_weight * SSIM(leftImage, leftImage_fromWarp, mask)

        return loss

    def lr_loss_per_level(self, leftEstDisp, rightEstDisp, leftImage, rightImage, leftMask=None, rightMask=None):
        from dmb.modeling.stereo.losses.utils import SSIM
        assert leftEstDisp.shape == rightEstDisp.shape, \
            'The shape of left and right disparity map should be the same!'
        N, C, H, W = leftEstDisp.shape
        leftImage = F.interpolate(leftImage, (H, W), mode='area')
        rightImage = F.interpolate(rightImage, (H, W), mode='area')

        leftImage_fromWarp = inverse_warp(rightImage, -leftEstDisp)
        rightImage_fromWarp = inverse_warp(leftImage, rightEstDisp)

        if leftMask is None:
            leftMask = torch.ones_like(leftImage > 0)
        loss = self.rms_weight * self.rms(leftImage[leftMask], leftImage_fromWarp[leftMask])
        loss += self.ssim_weight * SSIM(leftImage, leftImage_fromWarp, leftMask)

        if rightMask is None:
            rightMask = torch.ones_like(rightImage > 0)
        loss += self.rms_weight * self.rms(rightImage[rightMask], rightImage_fromWarp[rightMask])
        loss += self.ssim_weight * SSIM(rightImage, rightImage_fromWarp, leftMask)

        return loss

    def __call__(self, estLeftDisp, leftImage, rightImage, estRightDisp=None):
        if not isinstance(estLeftDisp, (list, tuple)):
            estLeftDisp = [estLeftDisp]

        if self.weights is None:
            self.weights = [1.0] * len(estLeftDisp)

        if estRightDisp is not None and not isinstance(estRightDisp, (list, tuple)):
            estRightDisp = [estRightDisp]
            assert len(estLeftDisp) == len(estRightDisp), \
                'The number of left and right disparity maps should be same'

        # compute loss for per level
        loss_all_level = []
        if estRightDisp is None:
            for est_disp_per_lvl in estLeftDisp:
                loss_all_level.append(
                    self.loss_per_level(est_disp_per_lvl, leftImage, rightImage)
                )
        else:
            for est_left_disp_per_lvl, est_right_disp_per_lvl in zip(estRightDisp, estRightDisp):
                leftMask, rightMask = self.get_per_level_not_occlusion(est_left_disp_per_lvl, est_right_disp_per_lvl)
                leftMask, rightMask = leftMask.expand_as(leftImage), rightMask.expand_as(leftImage)
                loss_all_level.append(
                    self.lr_loss_per_level(est_left_disp_per_lvl, est_right_disp_per_lvl, leftImage, rightImage,
                                           leftMask, rightMask)
                )

        # re-weight loss per level
        weighted_loss_all_level = dict()
        for i, loss_per_level in enumerate(loss_all_level):
            name = "inverse_warp_loss_lvl{}".format(i)
            weighted_loss_all_level[name] = self.weights[i] * loss_per_level

        return weighted_loss_all_level

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Loss weight: {}\n'.format(self.weights)
        repr_str += ' ' * 4 + 'Theta: {}\n'.format(self.theta)
        repr_str += ' ' * 4 + 'Epsilon: {}\n'.format(self.eps)
        repr_str += ' ' * 4 + 'SSIM loss weight: {}\n'.format(self.ssim_weight)

        return repr_str

    @property
    def name(self):
        return 'InverseWarpLoss'
