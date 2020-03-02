import numpy as np
import matplotlib.pyplot as plt


def disp_map(disp):
    """
    Based on color histogram, convert the gray disp into color disp map.
    The histogram consists of 7 bins, value of each is e.g. [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    Accumulate each bin, named cbins, and scale it to [0,1], e.g. [0.114, 0.299, 0.413, 0.587, 0.701, 0.886, 1.0]
    For each value in disp, we have to find which bin it belongs to
    Therefore, we have to compare it with every value in cbins
    Finally, we have to get the ratio of it accounts for the bin, and then we can interpolate it with the histogram map
    For example, 0.780 belongs to the 5th bin, the ratio is (0.780-0.701)/0.114,
    then we can interpolate it into 3 channel with the 5th [0, 1, 0] and 6th [0, 1, 1] channel-map
    Inputs:
        disp: numpy array, disparity gray map in (Height * Width, 1) layout, value range [0,1]
    Outputs:
        disp: numpy array, disparity color map in (Height * Width, 3) layout, value range [0,1]
    """
    map = np.array([
        [0, 0, 0, 114],
        [0, 0, 1, 185],
        [1, 0, 0, 114],
        [1, 0, 1, 174],
        [0, 1, 0, 114],
        [0, 1, 1, 185],
        [1, 1, 0, 114],
        [1, 1, 1, 0]
    ])
    # grab the last element of each column and convert into float type, e.g. 114 -> 114.0
    # the final result: [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    bins = map[0:map.shape[0] - 1, map.shape[1] - 1].astype(float)

    # reshape the bins from [7] into [7,1]
    bins = bins.reshape((bins.shape[0], 1))

    # accumulate element in bins, and get [114.0, 299.0, 413.0, 587.0, 701.0, 886.0, 1000.0]
    cbins = np.cumsum(bins)

    # divide the last element in cbins, e.g. 1000.0
    bins = bins / cbins[cbins.shape[0] - 1]

    # divide the last element of cbins, e.g. 1000.0, and reshape it, final shape [6,1]
    cbins = cbins[0:cbins.shape[0] - 1] / cbins[cbins.shape[0] - 1]
    cbins = cbins.reshape((cbins.shape[0], 1))

    # transpose disp array, and repeat disp 6 times in axis-0, 1 times in axis-1, final shape=[6, Height*Width]
    ind = np.tile(disp.T, (6, 1))
    tmp = np.tile(cbins, (1, disp.size))

    # get the number of disp's elements bigger than  each value in cbins, and sum up the 6 numbers
    b = (ind > tmp).astype(int)
    s = np.sum(b, axis=0)

    bins = 1 / bins

    # add an element 0 ahead of cbins, [0, cbins]
    t = cbins
    cbins = np.zeros((cbins.size + 1, 1))
    cbins[1:] = t

    # get the ratio and interpolate it
    disp = (disp - cbins[s]) * bins[s]
    disp = map[s, 0:3] * np.tile(1 - disp, (1, 3)) + map[s + 1, 0:3] * np.tile(disp, (1, 3))

    return disp


def disp_to_color(disp, max_disp=None):
    """
    Transfer disparity map to color map
    Args:
        disp (numpy.array): disparity map in (Height, Width) layout, value range [0, 255]
        max_disp (int): max disparity, optionally specifies the scaling factor
    Returns:
        disparity color map (numpy.array): disparity map in (Height, Width, 3) layout,
            range [0,255]
    """
    # grab the disp shape(Height, Width)
    h, w = disp.shape

    # if max_disp not provided, set as the max value in disp
    if max_disp is None:
        max_disp = np.max(disp)

    # scale the disp to [0,1] by max_disp
    disp = disp / max_disp

    # reshape the disparity to [Height*Width, 1]
    disp = disp.reshape((h * w, 1))

    # convert to color map, with shape [Height*Width, 3]
    disp = disp_map(disp)

    # convert to RGB-mode
    disp = disp.reshape((h, w, 3))
    disp = disp * 255.0

    return disp


def tensor_to_color(disp_tensor, max_disp=192):
    """
    The main target is to convert the tensor to image format
      so that we can load it into tensor-board.add_image()
    Args:
        disp_tensor (Tensor): disparity map
            in (BatchSize, Channel, Height, Width) or (BatchSize, Height, Width) layout
        max_disp (int): the max disparity value
    Returns:
        tensor_color (numpy.array): the converted disparity color map
            in (3, Height, Width) layout, value range [0,1]
    """
    if disp_tensor.ndimension() == 4:
        disp_tensor = disp_tensor[0, 0, :, :].detach().cpu()
    elif disp_tensor.ndimension() == 3:
        disp_tensor = disp_tensor[0, :, :].detach().cpu()
    else:
        disp_tensor = disp_tensor.detach().cpu()

    disp = disp_tensor.numpy()

    disp_color = disp_to_color(disp, max_disp) / 255.0
    disp_color = disp_color.transpose((2, 0, 1))

    return disp_color


def disp_err_to_color(disp_est, disp_gt):
    """
    Calculate the error map between disparity estimation and disparity ground-truth
    hot color -> big error, cold color -> small error
    Args:
        disp_est (numpy.array): estimated disparity map
            in (Height, Width) layout, range [0,255]
        disp_gt (numpy.array): ground truth disparity map
            in (Height, Width) layout, range [0,255]
    Returns:
        disp_err (numpy.array): disparity error map
            in (Height, Width, 3) layout, range [0,255]
    """
    """ matlab
    function D_err = disp_error_image (D_gt,D_est,tau,dilate_radius)
    if nargin==3
      dilate_radius = 1;
    end
    [E,D_val] = disp_error_map (D_gt,D_est);
    E = min(E/tau(1),(E./abs(D_gt))/tau(2));
    cols = error_colormap();
    D_err = zeros([size(D_gt) 3]);
    for i=1:size(cols,1)
      [v,u] = find(D_val > 0 & E >= cols(i,1) & E <= cols(i,2));
      D_err(sub2ind(size(D_err),v,u,1*ones(length(v),1))) = cols(i,3);
      D_err(sub2ind(size(D_err),v,u,2*ones(length(v),1))) = cols(i,4);
      D_err(sub2ind(size(D_err),v,u,3*ones(length(v),1))) = cols(i,5);
    end
    D_err = imdilate(D_err,strel('disk',dilate_radius));
    """
    # error color map with interval (0, 0.1875, 0.375, 0.75, 1.5, 3, 6, 12, 24, 48, inf)/3.0
    # different interval corresponds to different 3-channel projection
    cols = np.array(
        [
            [0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
            [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
            [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
            [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
            [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
            [3 / 3.0, 6 / 3.0, 254, 224, 144],
            [6 / 3.0, 12 / 3.0, 253, 174, 97],
            [12 / 3.0, 24 / 3.0, 244, 109, 67],
            [24 / 3.0, 48 / 3.0, 215, 48, 39],
            [48 / 3.0, float("inf"), 165, 0, 38]
        ]
    )

    # get the error (<3px or <5%) map
    tau = [3.0, 0.05]
    E = np.abs(disp_est - disp_gt)

    not_empty = disp_gt > 0.0
    tmp = np.zeros_like(disp_gt)
    tmp[not_empty] = E[not_empty] / disp_gt[not_empty] / tau[1]
    E = np.minimum(E / tau[0], tmp)

    h, w = disp_gt.shape
    err_im = np.zeros(shape=(h, w, 3)).astype(np.uint8)
    for col in cols:
        y_x = not_empty & (E >= col[0]) & (E <= col[1])
        err_im[y_x] = col[2:]

    return err_im


def group_color(est_disp, gt_disp=None, left_image=None, right_image=None, save_path=None):
    """
    Combine the Left Image, Disparity Estimation, Disparity Ground-Truth, Disparity Error into one column
    Args:
        est_disp (numpy.array): estimated disparity map
            in (Height, Width) layout, values range between [0, 255]
        gt_disp (numpy.array): disparity ground-truth map
            in (Height, Width) layout, values range between [0, 255]
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
    # concatenate Disparity Estimation and Ground-Truth in axis=0, and convert it to color
    if gt_disp is not None:
        group_image = np.concatenate((est_disp, gt_disp), 0)
        group_image = disp_to_color(group_image) / 255.0

        # add error map
        err_disp = disp_err_to_color(est_disp, gt_disp) / 255.0
        group_image = np.concatenate((group_image, err_disp), 0).clip(0., 1.)
    else:
        group_image = disp_to_color(est_disp) / 255.0

    if right_image is not None:
        right_image = np.array(right_image, np.float32) / 255.0
        # concatenate maps in order as [right_image, est_disp, gt_disp, err_disp]
        group_image = np.concatenate((right_image, group_image), 0).clip(0., 1.)

    if left_image is not None:
        left_image = np.array(left_image, np.float32) / 255.0
        # concatenate maps in order as [left_image, right_image, est_disp, gt_disp, ErrDisp]
        group_image = np.concatenate((left_image, group_image), 0).clip(0., 1.)

    if save_path is not None:
        plt.imsave(save_path, group_image, cmap=plt.cm.hot)

    return group_image
