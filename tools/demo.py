import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import matplotlib.pyplot as plt
import mmcv

from dmb.apis.inference import  init_model, inference_stereo, is_image_file
from dmb.visualization.stereo.vis import group_color

def visualize_disp(result_pkl):
    ori_data = result_pkl['OriginalData']
    net_result = result_pkl['Result']
    if 'disps' in net_result:
        disps = net_result['disps']
        best_disp = disps[0][0, 0, :, :].cpu().numpy()
    else:
        return
    plt.imshow(group_color(best_disp, ori_data['leftDisp'], ori_data['leftImage'], ori_data['rightImage']), cmap='hot')
    plt.show()


if __name__ == '__main__':
    print("Start Inference Stereo ... ")

    parser = argparse.ArgumentParser("DenseMatchingBenchmark Inference")

    parser.add_argument(
        "--config-path",
        type=str,
        help="config file path, e.g., ../configs/AcfNet/scene_flow_adaptive.py",
        required=True,
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="path to checkpoint, checkpoint download link often given in ../configs/Model/ResultOfModel.md, "
             "e.g., for AcfNet, you can find download link in ../configs/AcfNet/ResultOfAcfNet.md",
        required=True,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        help="data root contains directories including: "
             "$(data-root)/images/left/:     (dir for left image)"
             "$(data-root)/images/right/:    (dir for right image)"
             "$(data-root)/disparity/left/:  (dir for disparity map of left image), optional"
             "$(data-root)/disparity/right/: (dir for disparity map of right image), optional",
        default='./demo_data/',
    )

    parser.add_argument(
        "--device",
        type=str,
        help="device for running, e.g., cpu, cuda:0",
        default="cuda:0"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        help="directory path for logging",
        default='./output/'
    )

    parser.add_argument(
        "--pad-to-shape",
        nargs="+",
        type=int,
        help="image shape after padding for inference, e.g., [544, 960],"
             "after inference, result will crop to original image size",
        default=None,
    )

    parser.add_argument(
        "--crop-shape",
        nargs="+",
        type=int,
        help="image shape after cropping for inference, e.g., [512, 960]",
        default=None,
    )

    parser.add_argument(
        "--scale-factor",
        type=float,
        help="the scale of image upsample/downsample you want to inference, e.g., 2.0 upsample 2x, 0.5 downsample to 0.5x",
        default=1.0,
    )

    parser.add_argument(
        "--disp-div-factor",
        type=float,
        help="if disparity map given, after reading the disparity map, often have to divide a scale to get the real disparity value, e.g. 256 in KITTI",
        default=1.0,
    )

    args = parser.parse_args()

    config_path = args.config_path
    os.path.isfile(config_path)
    checkpoint_path = args.checkpoint_path
    os.path.isfile(checkpoint_path)

    
    print("Start Preparing Data ... ")
    data_root = args.data_root
    os.path.exists(data_root)
    imageNames = os.listdir(os.path.join(data_root, 'images/left/'))
    imageNames = [name for name in imageNames if is_image_file(name)]
    imageNames.sort()
    assert len(imageNames) > 1, "No images found in {}".format(os.path.join(data_root, 'images/left/'))
    batchesDict = []
    disparity_suffix = None
    if os.path.isdir(os.path.join(data_root, 'disparity/left')):
        dispNames = os.listdir(os.path.join(data_root, 'disparity/left'))
        disparity_suffix = {name.split('.')[-1] for name in dispNames}
    for imageName in imageNames:
        left_image_path = os.path.join(data_root, 'images/left/', imageName)
        right_image_path = os.path.join(data_root, 'images/right/', imageName)
        left_disp_map_path = None
        right_disp_map_path = None
        if disparity_suffix is not None:
            for suf in disparity_suffix:
                path = os.path.join(data_root, 'disparity/left', imageName.split('.')[0]+'.'+suf)
                if os.path.isfile(path):
                    left_disp_map_path = path
                    right_disp_map_path = path.replace('disparity/left', 'disparity/right')
                    break
        batchesDict.append({
            'left_image_path': left_image_path,
            'right_image_path': right_image_path,
            'left_disp_map_path': left_disp_map_path,
            'right_disp_map_path': right_disp_map_path,
        })
    print("Total {} images found".format(len(batchesDict)))


    device = args.device
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    print("Result will save to ", log_dir)

    pad_to_shape = args.pad_to_shape
    if pad_to_shape is not None:
        print("Image will pad to shape: ", pad_to_shape)

    crop_shape = args.crop_shape
    if crop_shape is not None:
        print("Image will crop to shape: ", crop_shape)

    scale_factor = args.scale_factor
    if scale_factor > 1.0:
        print("Image will upsample: {:.2f} ".format(scale_factor))
    elif scale_factor < 1.0:
        print("Image will downsample: {:.2f} ".format(1.0/scale_factor))

    disp_div_factor = args.disp_div_factor
    print("If disparity map given, it will be divided by {:.2f} to get the real disparity value".format(disp_div_factor))

    print("Initial Model ... ")
    model = init_model(config_path, checkpoint_path, device)
    print("Model initialed!")

    print("Start Inference ... ")
    inference_stereo(
        model,
        batchesDict,
        log_dir,
        pad_to_shape,
        crop_shape,
        scale_factor,
        disp_div_factor,
        device,
    )
    print("Inference Done!")

    print("Start Visualization ... ")
    for batch in batchesDict:
        pkl_path = os.path.join(log_dir, batch['left_image_path'].split('/')[-1].split('.')[0], 'result.pkl')
        print("Visualize ", pkl_path)
        result_pkl = mmcv.load(pkl_path)
        visualize_disp(result_pkl)

    print("Done!")




