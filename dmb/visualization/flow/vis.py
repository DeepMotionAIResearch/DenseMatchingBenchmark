import numpy as np
import torch
import matplotlib.pyplot as plt

UNKNOWN_FLOW_THRESH = 1e7


def make_color_wheel():
    """
    Generate color wheel according MiddleBury color code
    Outputs:
        color_wheel, (numpy.ndarray): color wheel, in [nCols, 3] layout
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    nCols = RY + YG + GC + CB + BM + MR

    color_wheel = np.zeros([nCols, 3])

    col = 0

    # RY
    color_wheel[0:RY, 0] = 255
    color_wheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    color_wheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    color_wheel[col:col+YG, 1] = 255
    col += YG

    # GC
    color_wheel[col:col+GC, 1] = 255
    color_wheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    color_wheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    color_wheel[col:col+CB, 2] = 255
    col += CB

    # BM
    color_wheel[col:col+BM, 2] = 255
    color_wheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    color_wheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    color_wheel[col:col+MR, 0] = 255

    return color_wheel


def flow_max_rad(flow):
    """
    Maximum sqrt(f_x^2 + f_y^2)
    Inputs:
        flow, (numpy.ndarray): flow map, in [Height, Width, 2] layout
    Outputs:
        max_rad, (float): the max flow in Euclidean Distance
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH) | np.isnan(u) | np.isnan(v)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    rad = np.sqrt(u ** 2 + v ** 2)
    max_rad = max(-1, np.max(rad))

    return max_rad


def flow_color(flow, max_rad=None):
    """
    compute optical flow color map
    Inputs:
        flow, (numpy.ndarray): optical flow map, in [Height, Width, 2] layout
        max_rad, (float): the max flow in Euclidean Distance
    Outputs:
        img, (numpy.ndarray): optical flow in color code, in [Height, Width, 3] layout
    """
    # [H, W]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    h, w = u.shape
    img = np.zeros([h, w, 3])

    # [H, W]
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH) | np.isnan(u) | np.isnan(v)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    # [H, W]
    if max_rad is None:
        rad = np.sqrt(u ** 2 + v ** 2)
        max_rad = max(-1, np.max(rad))

    # [H, W]
    u = u / (max_rad + np.finfo(float).eps)
    v = v / (max_rad + np.finfo(float).eps)

    color_wheel = make_color_wheel()
    nCols = np.size(color_wheel, 0)

    # [H, W]
    rad = np.sqrt(u**2+v**2)

    # angle, [H, W]
    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (nCols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == nCols+1] = 1
    f = fk - k0

    for i in range(0, np.size(color_wheel, 1)):
        tmp = color_wheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1-rad[idx]*(1-col[idx])
        notIdx = np.logical_not(idx)

        col[notIdx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-idxUnknow)))

    return img


def flow_to_color(flow, max_rad=None):
    """
    Convert flow into MiddleBury color code image
    Inputs:
        flow, (numpy.ndarray): optical flow map, in [Height, Width, 2] layout
        max_rad, (float): the max flow in Euclidean Distance
    Outputs:
        img, (numpy.ndarray): optical flow image in MiddleBury color, in [Height, Width, 3] layout
    """
    # [H, W]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # [H, W]
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH) | np.isnan(u) | np.isnan(v)

    # [H, W, 3]
    img = flow_color(flow, max_rad=max_rad)

    # [H, W, 3]
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    # [H, W, 3]
    img[idx] = 0

    # [H, W, 3]
    img = np.uint8(img)

    return img


def chw_to_hwc(flow):
    # [2, H, W] -> [H, W, 2]
    if isinstance(flow, torch.Tensor):
        flow = flow.clone().detach().cpu().numpy()
    assert isinstance(flow, np.ndarray)
    flow = flow.transpose((1, 2, 0))
    return flow


def tensor_to_color(flow_tensor, max_rad=None):
    """
    The main target is to convert the tensor to image format
      so that we can load it into tensor-board.add_image()
    Inputs:
        flow_tensor (Tensor): flow map
            in (BatchSize, 2, Height, Width) or (2, Height, Width) layout
        max_rad (flow): the max flow in Euclidean Distance
    Returns:
        flow_color (numpy.array): the converted flowarity color map
            in (3, Height, Width) layout, value range [0,1]
    """
    # [2, H, W]
    if flow_tensor.ndimension() == 4:
        flow_tensor = flow_tensor[0, :, :, :].detach().cpu()
    else:
        flow_tensor = flow_tensor.detach().cpu()

    assert flow_tensor.shape[0] == 2, 'the first dimension of flow Tensor should have {} channels, ' \
                                      'but got {}'.format(2, flow_tensor.shape[0])

    # [2, H, W] -> [H, W, 2]
    flow = chw_to_hwc(flow_tensor)

    # [H, W, 3]
    flow_color = flow_to_color(flow, max_rad)
    # [3, H, W]
    flow_color = flow_color.transpose((2, 0, 1))

    return flow_color


def flow_err_to_color(F_est, F_gt, F_gt_val=None):
    """
    Calculate the error map between optical flow estimation and optical flow ground-truth
    hot color -> big error, cold color -> small error
    Inputs:
        F_est, (numpy.ndarray): optical flow estimation map in (Height, Width, 2) layout, range [0,255]
        F_gt, (numpy.ndarray): optical flow ground-truth map in (Height, Width, 2) layout, range [0,255]
    Outputs:
        F_err, (numpy.ndarray): optical flow error map in (Height, Width, 3) layout, range [0,255]
    """

    F_shape = F_gt.shape[:2]

    # error color map with interval (0, 0.1875, 0.375, 0.75, 1.5, 3, 6, 12, 24, 48, inf)/3.0
    # different interval corresponds to different 3-channel projection
    cols = np.array([
        [0.0,       0.1875,         49,     54,     149],
        [0.1875,    0.375,          69,     117,    180],
        [0.375,     0.75,           116,    173,    209],
        [0.75,      1.5,            171,    217,    233],
        [1.5,       3.0,            224,    243,    248],
        [3.0,       6.0,            254,    224,    144],
        [6.0,       12.0,           253,    174,    97],
        [12.0,      24.0,           244,    109,    67],
        [24.0,      48.0,           215,    48,     39],
        [48.0,      float('inf'),   165,    0,      38]
    ])

    E_duv = F_gt - F_est
    E = np.square(E_duv)
    E = np.sqrt(E[:, :, 0] + E[:, :, 1])

    if F_gt_val is not None:
        E =  E * F_gt_val

    if F_gt_val is None:
        F_val = np.ones(F_shape, dtype=np.bool)
    else:
        F_val = (F_gt_val != 0.0)

    F_err = np.zeros((F_gt.shape[0], F_gt.shape[1], 3), dtype=np.uint8)
    for i in range(cols.shape[0]):
        F_find = F_val & (E >= cols[i, 0]) & (E <= cols[i, 1])
        F_find = np.where(F_find)
        F_err[:, :, 0][F_find] = int(cols[i, 2])
        F_err[:, :, 1][F_find] = int(cols[i, 3])
        F_err[:, :, 2][F_find] = int(cols[i, 4])

    return F_err


def group_color(est_flow, gt_flow=None, left_image=None, right_image=None, save_path=None):
    """
    Combine the Left Image, Flow Estimation, Flow Ground-Truth, Flow Error into one column
    Args:
        est_flow (numpy.array): estimated flow map
            in (Height, Width, 2) layout, values range between [0, 255]
        gt_flow (numpy.array): flow ground-truth map
            in (Height, Width, 2) layout, values range between [0, 255]
        left_image (numpy.array): left RGB image
            in (Height, Width, 3) layout, values range between [0, 255]
        right_image (numpy.array): left RGB image
            in (Height, Width, 3) layout, values range between [0, 255]
        save_path (str): the absolute/relative path you want to save the group image
    Returns:
        group (numpy.array): 4 maps grouped
            in (Height*4, Width, 3) layout, values range between [0, 1]
    """
    # Notes: All inputs should have the same height and width
    # TODO: add shape assert

    # plt.imshow only convert the value [0,1] into color-map, so we scale all the value to [0,1] below
    # concatenate Flow Estimation and Ground-Truth in axis=0, and convert it to color
    if gt_flow is not None:
        group_image = np.concatenate((est_flow, gt_flow), 0)
        group_image = flow_to_color(group_image) / 255.0

        # add error map
        err_flow = flow_err_to_color(est_flow, gt_flow) / 255.0
        group_image = np.concatenate((group_image, err_flow), 0).clip(0., 1.)
    else:
        group_image = flow_to_color(est_flow) / 255.0

    if right_image is not None:
        right_image = np.array(right_image, np.float32) / 255.0
        # concatenate maps in order as [right_image, est_flow, gt_flow, err_flow]
        group_image = np.concatenate((right_image, group_image), 0).clip(0., 1.)

    if left_image is not None:
        left_image = np.array(left_image, np.float32) / 255.0
        # concatenate maps in order as [left_image, right_image, est_flow, gt_flow, Errflow]
        group_image = np.concatenate((left_image, group_image), 0).clip(0., 1.)

    if save_path is not None:
        plt.imsave(save_path, group_image, cmap=plt.cm.hot)

    return group_image
