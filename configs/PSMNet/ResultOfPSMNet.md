# Result of PSMNet

```
@inproceedings{chang2018pyramid,
  title={Pyramid Stereo Matching Network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5410--5418},
  year={2018}
}
```

## Model Info

* Note: Test on GTX1080Ti, with resolution 384x1248.

|    Model Name         |   FLOPS   | Parameters | FPS  | Time(ms) |
|:---------------------:|:---------:|:----------:|:----:|:--------:|
|       PSMNet          | 938.186G  |  5.225M    | 1.67 |  599.2   |



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

RMSProp, lr(10 epochs) schedule: 1-10 with lr\*1


|  lr   |batch size |weight init| synced bn | float16   |loss scale | EPE(px)|  time  | BaiDuYun | GoogleDrive |
|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:------:|:------:|:--------:|:-----------:|
| 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 1.112  | 22h44m | [link][1], pw: 0kxt| [link][3] |



### KITTI-2015

|  lr   |batch size |weight init| synced bn | float16   |loss scale | D1(all)  |  time  | BaiDuYun | GoogleDrive |
|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:--------:|:------:|:--------:|:-----------:|
| 0.001 | 4*3       | ✗         |  ✓        | ✗         | ✗         | 2.33     | 15h15m | [link][2], pw: odt8| [link][4] |



[1]: https://pan.baidu.com/s/1e693uEuNK6uAg3OZstDJVQ
[2]: https://pan.baidu.com/s/1XnrtztXY9og3-JtBrLEGyA
[3]: https://drive.google.com/open?id=1aPJiGkt9P2Lt0UCcM817YjONV2DRDEBH
[4]: https://drive.google.com/drive/folders/1T__OTsViq5tkstm7jKV6p9wSs96EYUGw?usp=sharing
