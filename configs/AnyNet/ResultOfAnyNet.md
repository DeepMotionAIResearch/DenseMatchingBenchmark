# Result of AnyNet

## Model Info

```
@article{wang2018anytime,
  title={Anytime Stereo Image Depth Estimation on Mobile Devices},
  author={Wang, Yan and Lai, Zihang and Huang, Gao and Wang, Brian H. and Van Der Maaten, Laurens and Campbell, Mark and Weinberger, Kilian Q},
  journal={arXiv preprint arXiv:1810.11408},
  year={2018}
}
```

* Note: Test on GTX1080Ti, with resolution 384x1248.

|    Model Name         |   FLOPS   | Parameters | FPS  | Time(ms) |
|:---------------------:|:---------:|:----------:|:----:|:--------:|
|       AnyNet          |  1.476G   |  46.987K   | 



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

RMSProp, lr(20 epochs) schedule: 1-20 with lr\*1

- Inference with 1 GPU takes long time
- Although training for 20 epochs, but we find epoch=12 get the best result

|  lr   |batch size |weight init| synced bn | float16   |loss scale | EPE(px)|  time  | BaiDuYun | GoogleDrive |
|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:------:|:------:|:--------:|:-----------:|
| 0.0005| 1*6       | ✗         |  ✓        | ✗         | ✗         | 3.190  | 14h12m | [link][1], pw: dtff| [link][2] |



### KITTI-2015

|  lr   |batch size |weight init| synced bn | float16   |loss scale | D1(all)  |  time  | BaiDuYun | GoogleDrive |
|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:--------:|:------:|:--------:|:-----------:|
| 0.001 | 1*6       | ✗         |  ✓        | ✗         | ✗         | 



[1]: https://pan.baidu.com/s/10bP0TXCXHcdIg49Fv13H7Q
[2]: https://drive.google.com/open?id=1_5hBOfKwg_TnMFvZr4qEkU0bEwRoRlxL
