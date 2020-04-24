# Result of DeepPruner

## Model Info

```
@inproceedings{Duggal2019ICCV,  
title = {DeepPruner: Learning Efficient Stereo Matching  via Differentiable PatchMatch},  
author = {Shivam Duggal and Shenlong Wang and Wei-Chiu Ma and Rui Hu and Raquel Urtasun},  
booktitle = {ICCV},  
year = {2019}
}
```

* Note: Test on GTX1080Ti, with resolution 384x1280.

|    Model Name         |   FLOPS   | Parameters | FPS  | Time(ms) |
|:---------------------:|:---------:|:----------:|:----:|:--------:|
|    DeepPruner-4x      | 472.125G  |   7.390M   |  3.42|  292.4   |
|    DeepPruner-8x      | 194.181G  |   7.470M   |  7.67|  130.4   |



## Experiment


**hints**

* batch size: n * m, n GPUs m batch/GPU
* pass: clean pass or final pass of Scene Flow dataset, default clean pass
* weight init: initialize the convolution/bn layer while training from scratch, default no initialization
* synced bn: weather use synced bn provided by apex, default False
* float16: weather use mixture precision training with level 01 provided by apex, default False
* scale loss: the loss scale factor when using apex
* time: time consuming including the training and evaluating time, in format: x h(hour) y m(minute)
* EPE: end-point-error for SceneFlow
* D1(all): 3PE(px) & 5% for KITTI 2015


### SceneFlow

RMSProp, lr(15 epochs) schedule: 1-10 with lr\*1


|    Model Name         |  lr   |batch size |weight init| synced bn | float16   |loss scale | EPE(px)| time  | BaiDuYun | GoogleDrive |
|:---------------------:|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:------:|:-----:|:--------:|:-----------:|
|    DeepPruner-4x      | 0.001 | 4*2       | ✗         |  ✓        | ✗         | ✗         | 



### KITTI-2015

|  lr   |batch size | synced bn |loss scale | 3PE(px) & 5% | 
|:-----:|:---------:|:---------:|:---------:|:------------:|
| 0.001 | 4*3       |  ✓        | ✗         | 
