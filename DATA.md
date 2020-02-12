### Prepare Scene Flow and KITTI dataset.

It is recommended to symlink the dataset root to `$DenseMatchingBenchmark/datasets/`.

```
├── KITTI-2012
│   └── data_stereo_flow
│       ├── testing
│       └── training
├── KITTI-2015
│   ├── calib
│   ├── devkit
│   ├── testing
│   │   ├── image_2
│   │   └── image_3
│   └── training
│       ├── disp_noc_0
│       ├── disp_noc_1
│       ├── image_2
│       └── image_3
└── SceneFlow
    ├── calib
    ├── driving
    │   ├── disparity
    │   ├── frames_cleanpass
    │   └── frames_finalpass
    ├── flyingthings3d
    │   ├── disparity
    │   ├── frames_cleanpass
    │   └── frames_finalpass
    └── Monkaa
        ├── disparity
        ├── frames_cleanpass
        └── frames_finalpass


```

### Prepare visualization dataset.

We enable evaluation and visualization for each epoch. Especially, the visualization means visualize the estimated results.

It is recommended to download the visualization data we prepared, btw, you can also prepare by yourself.

####How To Use

To use, you just have to make the param 'data=dict(vis=...)' in config file valid.

####Down Link
The down-link for visualization data including:
1. Baidu YunPan: https://pan.baidu.com/s/1oH1cAIwKysCKe5RuFP_zYQ  password: wyju
2. Google Drive: on the way.

```
visualization_data
├── KITTI-2015
│   ├── annotations
│   ├── calib
│   ├── disparity
│   ├── genVisKITTI2015AnnoFile.py
│   ├── genVisKITTIVOAnnoFile.py
│   ├── images
│   └── velodyne_points
└── SceneFlow
    ├── __init__.py
    ├── annotations
    ├── disparity
    ├── genExtraSFAnnoFile.py
    ├── images
    ├── occ
    └── readme.txt
```

