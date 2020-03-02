import matplotlib.pyplot as plt
from collections import abc as container_abcs

import numpy as np

import torch

from dmb.visualization.flow.vis import flow_to_color, tensor_to_color, flow_max_rad, chw_to_hwc, group_color


# Attention: in this framework, we always set the first result, e.g., flow map, as the best.

class ShowFlow(object):
    """
    Show the result related to flow
    Args:
        result (dict): the result to show
            Flow (list, tuple, Tensor): in [1, 2, H, W]
            GroundTruth (torch.Tensor): in [1, 2, H, W]
            leftImage (numpy.array): in [H, W, 3]
            rightImage (numpy.array): in [H, W, 3]
    Returns:
        dict, mode in HWC is for save convenient, mode in CHW is for tensor-board convenient
            GrayFlow (numpy.array): the original flow map output of network, will be saved to disk
                in (H, W, 2) layout, value range [-inf, inf]
            ColorFlow (numpy.array): the converted flow color map, will be saved to disk
                in (H, W, 3) layout, value range [0,1]
            GroupColor (numpy.array): in (H, W, 3) layout, value range [0, 1], will be saved to disk
            Flow (list, tuple, numpy.array): the converted flow color map, will be showed on TensorBoard
                in (3, H, W) layout, value range [0,1]
            GroundTruth (numpy.array): in (3, H, W) layout, value range [0, 1], will be showed on TensorBoard
    """
    def __call__(self, result):

        self.result = result
        self.getItem()

        process_result = {}

        if self.estFlow is not None:
            firstFlow = self.getFirstItem(self.estFlow)
            if firstFlow is not None:
                # [H, W, 3], [H, W, 3]
                grayFlow, colorFlow = self.get_gray_and_color_flow(firstFlow, self.max_rad)
                process_result.update(GrayFlow=grayFlow)
                process_result.update(ColorFlow=colorFlow)
            # [H, W, 3]
            group = self.vis_group_color(self.estFlow[0], self.gtFlow, self.leftImage, self.rightImage)
            # [3, H, W]
            estFlowColor = self.vis_per_flow(self.estFlow, self.max_rad)
            process_result.update(Flow=estFlowColor)
            process_result.update(GroupColor=group)

        if self.gtFlow is not None:
            # [3, H, W]
            gtFlowColor = self.vis_per_flow(self.gtFlow, self.max_rad)
            process_result.update(GroundTruth=gtFlowColor)

        return process_result

    def getItem(self):
        if "GroundTruth" in self.result.keys() and self.result['GroundTruth'] is not None:
            # [1, 2, H, W] -> [2, H, W]
            self.gtFlow = self.result['GroundTruth'][0, :, :, :]
            # [2, H, W] -> [H, W, 2] -> scalar
            self.max_rad = flow_max_rad(chw_to_hwc(self.gtFlow))

        else:
            self.max_rad = None
            self.gtFlow = None

        if 'Flow' in self.result.keys():
            if isinstance(self.result['Flow'], (list, tuple)):
                self.estFlow = self.result['Flow']
            else:
                self.estFlow = [self.result['Flow']]
        else:
            self.estFlow = None

        if 'leftImage' in self.result.keys():
            self.leftImage = self.result['leftImage']
        else:
            self.leftImage = None
        if 'rightImage' in self.result.keys():
            self.rightImage = self.result['rightImage']
        else:
            self.rightImage = None

    def getFirstItem(self, item):
        if isinstance(item, container_abcs.Sequence):
            return item[0]
        if isinstance(item, container_abcs.Mapping):
            for key in item.keys():
                return item[key]
        if isinstance(item, (np.ndarray, torch.Tensor)):
            return item
        return None

    # For TensorBoard log flow map, [3, H, W]
    def vis_per_flow(self, Flow, max_rad):
        # change every flow map to color map
        error_msg = "Flow must contain tensors, dicts or lists; found {}"
        if isinstance(Flow, torch.Tensor):
            return tensor_to_color(Flow.clone(), max_rad)
        elif isinstance(Flow, container_abcs.Mapping):
            return {key: self.vis_per_flow(Flow[key], max_rad) for key in Flow}
        elif isinstance(Flow, container_abcs.Sequence):
            return [self.vis_per_flow(samples, max_rad) for samples in Flow]

        raise TypeError((error_msg.format(type(Flow))))

    # For saving flow map, [C, H, W]
    def get_gray_and_color_flow(self, Flow, max_rad=None):
        assert isinstance(Flow, (np.ndarray, torch.Tensor))

        if torch.is_tensor(Flow):
            Flow = Flow.clone().detach().cpu()

        if len(Flow.shape) == 4:
            Flow = Flow[0, :, :, :]

        # [2, H, W] -> [H, W, 2]
        Flow = chw_to_hwc(Flow)
        # [H, W, 2]
        grayFlow = Flow.copy()
        # [H, W, 3]
        colorFlow = flow_to_color(Flow.copy(), max_rad=max_rad)

        return grayFlow, colorFlow

    def vis_group_color(self, estFlow, gtFlow=None, leftImage=None, rightImage=None, save_path=None):
        """
        Args:
            estFlow, (tensor or numpy.array): in (1, 2, Height, Width) or (2, Height, Width) layout
            gtFlow, (None or tensor or numpy.array): in (1, 2, Height, Width) or (2, Height, Width) layout
            leftImage, (None or numpy.array), in (Height, Width, 3) layout
            rightImage, (None or numpy.array), in (Height, Width, 3) layout
            save_path, (None or String)
        Output:
            details refer to dmb.visualization.group_color, (Height, Width, 3)
        """
        assert isinstance(estFlow, (np.ndarray, torch.Tensor))

        if torch.is_tensor(estFlow):
            estFlow = estFlow.clone().detach().cpu().numpy()

        if estFlow.ndim == 4:
            estFlow = estFlow[0, :, :, :]

        if gtFlow is not None:
            assert isinstance(gtFlow, (np.ndarray, torch.Tensor))
            if torch.is_tensor(gtFlow):
                gtFlow = gtFlow.clone().detach().cpu().numpy()
            if gtFlow.ndim == 4:
                gtFlow = gtFlow[0, :, :, :]

        # [2, H, W] -> [H, W, 2]
        estFlow = chw_to_hwc(estFlow)
        gtFlow = chw_to_hwc(gtFlow)

        return group_color(estFlow, gtFlow, leftImage, rightImage, save_path)



class ShowResultTool(object):
    def __init__(self):
        self.show_flow_tool = ShowFlow()

    def __call__(self, result):
        process_result = {}
        process_result.update(self.show_flow_tool(result))
        return process_result
