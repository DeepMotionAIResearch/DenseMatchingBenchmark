# Result of StereoNet

## Model Info

```
@inproceedings{khamis2018stereonet,
  title={Stereonet: Guided hierarchical refinement for real-time edge-aware depth prediction},
  author={Khamis, Sameh and Fanello, Sean and Rhemann, Christoph and Kowdle, Adarsh and Valentin, Julien and Izadi, Shahram},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={573--590},
  year={2018}
}

@inproceedings{zhang2018activestereonet,
  title={Activestereonet: End-to-end self-supervised learning for active stereo systems},
  author={Zhang, Yinda and Khamis, Sameh and Rhemann, Christoph and Valentin, Julien and Kowdle, Adarsh and Tankovich, Vladimir and Schoenberg, Michael and Izadi, Shahram and Funkhouser, Thomas and Fanello, Sean},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={784--801},
  year={2018}
}

```

* Note: Test on GTX1080Ti, with resolution 384x1248.

|    Model Name         |   FLOPS   | Parameters | FPS  | Time(ms) |
|:---------------------:|:---------:|:----------:|:----:|:--------:|
| StereoNet-8X-2stage   | 78.512G   |  399.066K  | 19.17|  52.2    |
| StereoNet-8X-4stage   | 186.719G  |  624.860K  |  8.54|  117.0   |



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

RMSProp, lr(11 epochs) schedule: 1-11 with lr\*1

- Inference with 1 GPU takes long time

|    Model Name         |  lr   |batch size |weight init| synced bn | float16   |loss scale | EPE(px)| time   | BaiDuYun | GoogleDrive |
|:---------------------:|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:------:|:------:|:--------:|:-----------:|
| StereoNet-8X-2stage   | 0.001 | 1*4       | ✗         |  ✓        | ✗         | ✗         | 1.533  | 40h56m |[link][1], pw: rza0 | [link][2]|
| StereoNet-8X-4stage   | 0.001 | 1*4       | ✗         |  ✓        | ✗         | ✗         | 1.329  | 143h45m|[link][3], pw: gpjm | [link][4]|



### KITTI-2015

|  lr   |batch size | synced bn |loss scale | 3PE(px) & 5% | 
|:-----:|:---------:|:---------:|:---------:|:------------:|
| 0.001 | 1*4       |  ✓        | ✗         | 


[1]: https://pan.baidu.com/s/1cuvjEETJUnpnxy_pFqiTRw
[2]: https://drive.google.com/open?id=1cuXzQDfQ28a9gmSJichaIGYsEITGp_Qh
[3]: https://pan.baidu.com/s/13DOhuuvqvNL9ksg5_85GEw
[4]: https://drive.google.com/open?id=10TYF5SqN26-GsVIf2ytXALbNMBgOLH_1
