import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from imageio import imread
import os
import os.path as osp
import sys
import mmcv

import random

def mask2color(x):
    x = np.array(x)
    color = np.round(np.array(plt.cm.jet(x)[:3]) * 255)
    return color

def kernel_inv_map(vis_attr, target_point, map_h, map_w):
    filter_size = vis_attr['filter_size']
    pos_shift = [vis_attr['dilation'] * i - vis_attr['pad'] for i in range(filter_size)]
    source_point = []
    for idx in range(vis_attr['filter_size'] ** 2):
        cur_source_point = np.array([target_point[0] + pos_shift[idx // filter_size],
                                     target_point[1] + pos_shift[idx % filter_size]])
        if cur_source_point[0] < 0 or cur_source_point[1] < 0 \
                or cur_source_point[0] > map_h - 1 or cur_source_point[1] > map_w - 1:
            continue
        source_point.append(cur_source_point.astype('f'))
    return source_point


def offset_inv_map(source_points, offset):
    for idx, _ in enumerate(source_points):
        source_points[idx][0] += offset[2 * idx]
        source_points[idx][1] += offset[2 * idx + 1]
    return source_points


def get_bottom_position(vis_attr, top_points, all_offset):
    map_h = all_offset[0].shape[2]
    map_w = all_offset[0].shape[3]

    for level in range(vis_attr['plot_level'] - 1, vis_attr['plot_level']):
        source_points = []
        for idx, cur_top_point in enumerate(top_points):
            cur_top_point = np.round(cur_top_point)
            if cur_top_point[0] < 0 or cur_top_point[1] < 0 \
                    or cur_top_point[0] > map_h - 1 or cur_top_point[1] > map_w - 1:
                continue
            cur_source_point = kernel_inv_map(vis_attr, cur_top_point, map_h, map_w)
            cur_offset = np.squeeze(all_offset[level][:, :, int(cur_top_point[0]), int(cur_top_point[1])])
            cur_source_point = offset_inv_map(cur_source_point, cur_offset)
            source_points = source_points + cur_source_point
        top_points = source_points
    return source_points


def plot_according_to_point(vis_attr, im, source_points, map_h, map_w, color):
    plot_area = vis_attr['plot_area']
    for idx, cur_source_point in enumerate(source_points):
        y = np.round((cur_source_point[0] + 0.5) * im.shape[0] / map_h).astype('i')
        x = np.round((cur_source_point[1] + 0.5) * im.shape[1] / map_w).astype('i')

        if x < 0 or y < 0 or x > im.shape[1] - 1 or y > im.shape[0] - 1:
            continue
        y = min(y, im.shape[0] - vis_attr['plot_area'] - 1)
        x = min(x, im.shape[1] - vis_attr['plot_area'] - 1)
        y = max(y, vis_attr['plot_area'])
        x = max(x, vis_attr['plot_area'])
        im[y - plot_area:y + plot_area + 1, x - plot_area:x + plot_area + 1, :] = np.tile(
            np.reshape(color[idx], (1, 1, 3)), (2 * plot_area + 1, 2 * plot_area + 1, 1)
        )
    return im


def show_dconv2_offset(im, all_offset, points, all_mask=None, step=[2, 2], filter_size=3,
                      dilation=2, pad=2, plot_area=2, plot_level=3, color=[0, 255, 0]):
    vis_attr = {'filter_size': filter_size, 'dilation': dilation, 'pad': pad,
                'plot_area': plot_area, 'plot_level': plot_level}

    map_h = all_offset[0].shape[2]
    map_w = all_offset[0].shape[3]

    step_h = step[0]
    step_w = step[1]
    start_h = np.round(step_h / 2).astype(np.uint32)
    start_w = np.round(step_w / 2).astype(np.uint32)

    final_im = im.copy()
    plt.figure()
    for point in points:
        im_h, im_w = point

        target_point = np.array([im_h, im_w])
        source_y = np.round(target_point[0] * im.shape[0] / map_h).astype(np.uint32)
        source_x = np.round(target_point[1] * im.shape[1] / map_w).astype(np.uint32)
        if source_y < plot_area or source_x < plot_area \
                or source_y >= im.shape[0] - plot_area or source_x >= im.shape[1] - plot_area:
            continue

        cur_im = np.copy(im)
        source_points = get_bottom_position(vis_attr, [target_point], all_offset)
        # get source points mask
        source_point_mask = all_mask[vis_attr['plot_level']][:, :, target_point[0], target_point[1]]
        source_point_mask = np.squeeze(np.array(source_point_mask))
        mask_color = [mask2color(m) for m in source_point_mask]
        cur_im = plot_according_to_point(vis_attr, cur_im, source_points, map_h, map_w, color=mask_color)
        cur_im[source_y - plot_area:source_y + plot_area + 1, source_x - plot_area:source_x + plot_area + 1, :] = \
            np.tile(np.reshape([0, 255, 0], (1, 1, 3)), (2 * plot_area + 1, 2 * plot_area + 1, 1))

        final_im = plot_according_to_point(vis_attr, final_im, source_points, map_h, map_w, color=mask_color)
        final_im[source_y - plot_area:source_y + plot_area + 1, source_x - plot_area:source_x + plot_area + 1, :] = \
            np.tile(np.reshape([0, 255, 0], (1, 1, 3)), (2 * plot_area + 1, 2 * plot_area + 1, 1))

        plt.axis("off")
        plt.imshow(cur_im)
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
    return final_im

def crop(im, p, r):
    hl, hh = max(p[0] - r, 0), min(p[0] + r, im.shape[0])
    wl, wh = max(p[1] - r, 0), min(p[1] + r, im.shape[1])
    return im[hl:hh, wl:wh, :]


def pltShow(color_disps, save_path=None):
    if isinstance(color_disps, (tuple, list)):
        color_disps = np.concatenate(color_disps, 1)
    if save_path is not None:
        plt.imsave(save_path, color_disps)
    plt.imshow(color_disps)
    plt.show()



if __name__ == '__main__':
    dmb_root = '/home/zhixiang/youmin/projects/depth/public/DenseMatchingBenchmark'
    data_root = '/home/zhixiang/youmin/out/data/visualization_data/SceneFlow/'
    exp_root = '/home/zhixiang/youmin/out/exps/MonoStereo/scene_flow_correlation_C8_deform_radiusFP'
    log_root = '/home/zhixiang/youmin/out/exps/MonoStereo/sampleExp'
    sys.path.insert(0, dmb_root)

    device = torch.device('cuda:0')
    config = osp.join(dmb_root, 'configs/MonoStereo/scene_flow.py')
    checkpoint = osp.join(exp_root, 'epoch_36.pth')
    batchesDict = [
        {'left_image_path': osp.join(data_root, 'images/left/0.png'),
         'right_image_path': osp.join(data_root, 'images/right/0.png'),
         'left_disp_map_path': osp.join(data_root, 'disparity/left/0.pfm'),
         'right_disp_map_path': osp.join(data_root, 'disparity/right/0.pfm'),
         },
    ]

    results = mmcv.load('/home/zhixiang/youmin/out/exps/MonoStereo/sampleExp/0/result.pkl')
    Result = results['Result']
    oriData = results['OriginalData']
    print(Result.keys())
    disps = Result['disps']
    costs = Result['costs']
    print(len(disps))
    print(len(costs))
    print(disps[0].shape)
    print(costs[0].shape)
    offsets = Result['offsets']
    masks = Result['masks']
    print(len(offsets))
    print(len(masks))
    print(offsets[0].shape)
    print(masks[0].shape)
    for i in range(len(offsets)):
        print('{}: min-> {}, max-> {}'.format(i, offsets[i].min(), offsets[i].max()))

    from dmb.visualization.stereo.show_result import ShowDisp
    show_tool = ShowDisp()
    gray_disps = []
    color_disps = []
    for d in disps:
        grd, cod = show_tool.get_gray_and_color_disp(d, max_disp=None)
        gray_disps.append(grd)
        color_disps.append(cod)

    # pltShow(color_disps)

    points = [[100, 100], [300, 420], [500, 400], [500, 620], [100, 620], [300, 575]]
    # levels = [2, 4, 6]
    levels = [1, 3, 5]
    im = []
    for i in levels:
        #     im.append(show_dconv_offset(oriData['leftImage']/255, offsets, points, all_mask=masks, step=[1, 1], filter_size=5,
        #               dilation=1, pad=2, plot_area=2, plot_level=i, color=(0, 255, 255)))
        im.append(show_dconv2_offset(color_disps[2], offsets, points, all_mask=masks, step=[1, 1], filter_size=5,
                                    dilation=1, pad=2, plot_area=2, plot_level=i, color=(0, 255, 255)))
    pltShow(im)
    r = 80
    for i in range(len(points)):
        crop_im = []
        for j in range(len(im)):
            crop_im.append(crop(im[j], points[i], r))
        pltShow(crop_im)