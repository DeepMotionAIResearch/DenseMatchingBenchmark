##Model Info

* Note: Test on GTX1080Ti, with resolution 540x960.

|    Model Name         |   FLOPS   | Parameters | FPS  | Time(ms) |
|:---------------------:|:---------:|:----------:|:----:|:--------:|
| StereoNet-8X-2stage   | 78.512G   |  399.066K  | 19.17|  52.2    |
| StereoNet-8X-4stage   | 186.719G  |  624.860K  |  8.54|  117.0   |



##Experiment


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
| StereoNet-8X-2stage   | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 



### KITTI-2015

|  lr   |batch size | synced bn |loss scale | 3PE(px) & 5% | 
|:-----:|:---------:|:---------:|:---------:|:------------:|
| 0.001 | 4*3       |  ✓        | ✗         | 
