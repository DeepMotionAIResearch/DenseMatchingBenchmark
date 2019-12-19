import torch
import torch.nn as nn
import math

eps = 1e-12


class bilateralFilter(nn.Module):
    """
    Args:
        kernel_size(int, tuple): bilateral filter kernel size
        sigma_image(int, float): the derivation of Image Gaussian distribution
        sigma_gaussian(int, float): the derivation of Disparity Gaussian distribution
        leftImage(tensor): in [BatchSize, 1, Height, Width] layout, gray image
        estDisp(tensor): in [BatchSize, 1, Height, Width] layout, the estimated disparity map
    Outputs:
        fineDisp(tensor): in [BatchSize, 1, Height, Width] layout, the refined disparity map
    """

    def __init__(self, kernel_size, sigma_image, sigma_gaussian):
        super(bilateralFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma_image = sigma_image
        self.sigma_gaussian = sigma_gaussian
        self.image_conv = []
        self.image_kernel = self.create_image_kernel(self.kernel_size)
        for i in range(len(self.image_kernel)):
            self.image_conv.append(
                nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False))
            self.image_conv[i].weight.data = self.image_kernel[i]
            self.image_conv[i].weight.requires_grad = False

        self.disp_conv = []
        self.disp_kernel = self.create_disparity_kernel(self.kernel_size)
        for i in range(len(self.disp_kernel)):
            self.disp_conv.append(
                nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False))
            self.disp_conv[i].weight.data = self.disp_kernel[i]
            self.disp_conv[i].weight.requires_grad = False

    def forward(self, leftImage, estDisp):
        assert leftImage.shape == estDisp.shape
        assert estDisp.shape[1] == 1

        for i in range(len(self.disp_conv)):
            self.disp_conv[i] = self.disp_conv[i].to(leftImage.device)
        for i in range(len(self.image_conv)):
            self.image_conv[i] = self.image_conv[i].to(leftImage.device)

        index_image_conv = 0
        index_disp_conv = 0
        fineDisp = None
        weight = None
        for i in range(-(self.kernel_size // 2), (self.kernel_size // 2 + 1)):
            for j in range(-(self.kernel_size // 2), (self.kernel_size // 2 + 1)):
                if i == 0 and j == 0:
                    image_diff_weight = torch.ones_like(estDisp)
                else:
                    image_diff_weight = (
                        (-self.image_conv[index_image_conv](leftImage).pow(2.0) / (2 * self.sigma_image ** 2)).exp())
                    index_image_conv += 1

                dist = math.exp(-float(i ** 2 + j ** 2) / float(2 * self.sigma_gaussian ** 2))
                dist_diff_weight = torch.full_like(estDisp, dist)

                disp = self.disp_conv[index_disp_conv](estDisp)

                if index_disp_conv == 0:
                    weight = dist_diff_weight * image_diff_weight
                    fineDisp = disp * dist_diff_weight * image_diff_weight
                else:
                    weight += dist_diff_weight * image_diff_weight
                    fineDisp += disp * dist_diff_weight * image_diff_weight

        fineDisp = (fineDisp + eps) / (weight + eps)

        return fineDisp

    def create_disparity_kernel(self, kernel_size):
        total_direction = kernel_size * kernel_size
        kernel = []
        for i in range(total_direction):
            kernel.append(torch.zeros(1, 1, total_direction))
            kernel[i][:, :, i] = 1
            kernel[i] = kernel[i].reshape(1, 1, kernel_size, kernel_size)

        return kernel

    def create_image_kernel(self, kernel_size):
        total_direction = kernel_size * kernel_size
        kernel = []
        for i in range(total_direction):
            kernel.append(torch.zeros(1, 1, total_direction))
            kernel[i][:, :, i] = -1
            kernel[i][:, :, total_direction // 2] = 1
            kernel[i] = kernel[i].reshape(1, 1, kernel_size, kernel_size)

        return kernel[:total_direction // 2] + kernel[total_direction // 2 + 1:]
