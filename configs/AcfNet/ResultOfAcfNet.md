##Model Info

* Note: Test on GTX1080Ti, with resolution 540x960.

|    Model Name         |   FLOPS   | Parameters | FPS  | Time(ms) |
|:---------------------:|:---------:|:----------:|:----:|:--------:|
|  AcfNet(uniform)      | 1080.0G   |  5.227M    | 1.66 |  600.8   |
|  AcfNet(adaptive)     | 1239.0G   |  5.559M    | 1.38 |  723.1   |



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


#### 10 epoch

RMSProp, lr(10 epochs) schedule: 1-10 with lr\*1


|   model name   |  lr   |batch size |weight init| synced bn | float16   |loss scale | EPE(px)| time   | BaiDuYun | GoogleDrive |
|:--------------:|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:------:|:------:|:--------:|:-----------:|
|    adaptive    | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         |
|    uniform     | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 0.8372 | 22h13m |


#### 20 epoch

RMSProp, lr(20 epochs) schedule: 1-20 with lr\*1


|   model name   |  lr   |batch size |weight init| synced bn | float16   |loss scale | EPE(px)| time   | BaiDuYun | GoogleDrive |
|:--------------:|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:------:|:------:|:--------:|:-----------:|
|    adaptive    | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 
|    uniform     | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 



### KITTI-2015


|   model name   |  lr   |batch size |weight init| synced bn | float16   |loss scale | D1(all) |  time  | BaiDuYun | GoogleDrive |
|:--------------:|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:-------:|:------:|:--------:|:-----------:|
|    adaptive    | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 
|    uniform     | 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 
