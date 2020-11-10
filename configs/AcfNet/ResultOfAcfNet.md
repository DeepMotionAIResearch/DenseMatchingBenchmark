# Result of AcfNet

## Model Info

* Note: Test on GTX1080Ti, with resolution 384x1248.

|    Model Name         |   FLOPS   | Parameters | FPS  | Time(ms) |
|:---------------------:|:---------:|:----------:|:----:|:--------:|
|  AcfNet(uniform)      | 1080.0G   |  5.227M    | 1.66 |  600.8   |
|  AcfNet(adaptive)     | 1239.0G   |  5.559M    | 1.38 |  723.1   |


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
* xPE (1PE, 2PE, 3PE, 5PE): pixel error where (GroundTruth - Estimation) <= x
* D1(all): 3PE(px) & 5% for KITTI 2015


### SceneFlow


#### 10 epoch

RMSProp, lr(10 epochs) schedule: 1-10 with lr\*1


|   model name   |  lr   |batch size |weight init| synced bn | float16   |loss scale | EPE(px)| time   | BaiDuYun | GoogleDrive |
|:--------------:|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:------:|:------:|:--------:|:-----------:|
|    adaptive    | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 0.8308 | 68h18m | [link][3], pw: qxxr | [link][4]|
|    uniform     | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 0.8511 | 26h50m | [link][1], pw: 9s4e | [link][2]|


##### Disparity Predictor Ablation

If we alternate the disparity predictor from `FasterSoftArgmin` to `LocalSoftArgmin` only for reference


|   model name   | predictor |    1PE    |    2PE    |    3PE    |    5PE    | EPE(px)   |
|:--------------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|    adaptive    |   Faster  |   7.905    |   5.125   |  3.991    |   2.873   |   0.8308  |
|    adaptive    |   Local   |
|    uniform     |   Faster  |   8.626   |   5.544   |   4.291   |   3.061   |   0.8511  |
|    uniform     |   Local   |   5.983   |   3.620   |   2.838   |   2.164   |   0.8216  |


**Analysis**

1. Little difference for `EPE`, but make significant effect on `xPE`.
 
2. Therefore, to get better result on KITTI, alternate the disparity predictor from `FasterSoftArgmin` to `LocalSoftArgmin`

3. `LocalSoftArgmin` only works when cost volume supervised with uni-modal distribution, worse result for PSMNet


#### 20 epoch

RMSProp, lr(20 epochs) schedule: 1-20 with lr\*1


|   model name   |  lr   |batch size |weight init| synced bn | float16   |loss scale | EPE(px)| time   | BaiDuYun | GoogleDrive |
|:--------------:|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:------:|:------:|:--------:|:-----------:|
|    adaptive    | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 0.7172 | 134h31m| [link][3], pw: qxxr | [link][4]|
|    uniform     | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 0.7440 | 56h53m | [link][1], pw: 9s4e | [link][2]|


##### Disparity Predictor Ablation

If we alternate the disparity predictor from `FasterSoftArgmin` to `LocalSoftArgmin` only for reference


|   model name   | predictor |    1PE    |    2PE    |    3PE    |    5PE    | EPE(px)   |
|:--------------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|    adaptive    |   Faster  |   6.918   |   4.480   |   3.485   |   2.498   |   0.7172  |
|    adaptive    |   Local   |
|    uniform     |   Faster  |   7.647   |   4.917   |   4.381   |   2.693   |   0.7440  |
|    uniform     |   Local   |   5.338   |   3.232   |   2.536   |   1.927   |   0.7161  |


### KITTI-2015


|   model name   |  lr   |batch size |weight init| synced bn | float16   |loss scale | D1(all) |  time  | BaiDuYun | GoogleDrive |
|:--------------:|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:-------:|:------:|:--------:|:-----------:|
|    adaptive    | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 
|    uniform     | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         |


## How TO

Alternate the disparity predictor from `FasterSoftArgmin` to `LocalSoftArgmin`

In config file, change
```python
disp_predictor=dict(
        # default FasterSoftArgmin
        type="FASTER",
        # the maximum disparity of disparity search range
        max_disp=max_disp,
        # the start disparity of disparity search range
        start_disp=0,
        # the step between near disparity sample
        dilation=1,
        # the temperature coefficient of soft argmin
        alpha=1.0,
        # whether normalize the estimated cost volume
        normalize=True,
    ),
```

to
```python
disp_predictor=dict(
        # LocalSoftArgmin
        type="LOCAL",
        # the maximum disparity of disparity search range
        max_disp=max_disp,
        # the radius of window when local sampling
        radius=3,
        # the start disparity of disparity search range
        start_disp=0,
        # the step between near disparity sample
        dilation=1,
        # the step between near disparity index when local sampling
        radius_dilation=1,
        # the temperature coefficient of soft argmin
        alpha=1.0,
        # whether normalize the estimated cost volume
        normalize=True,
    ),

```



[1]: https://pan.baidu.com/s/11sR2mUEhCyp06g7LXsFG2g
[2]: https://drive.google.com/open?id=1VwOrfEPfbdrzYvie2bVqUS1-8k_5_Yt1
[3]: https://pan.baidu.com/s/1jINm_AYzG9f89ml2Dire0A
[4]: https://drive.google.com/open?id=1sLHrE76SFRfEzu2YK3XwF8QqzBDVdPDx
