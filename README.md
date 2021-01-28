# DenseMatchingBenchmark
This project aims at providing the necessary building blocks for easily
creating dense matching models, e.g. Stereo Matching, Scene Flow using **PyTorch 1.2** or higher.

## Introduction
Our architecture is based on two wellknown detection framework: [mmdetection](https://github.com/open-mmlab/mmdetection) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). By integating their major features, our architecture is suitable for dense matching, and achieves robust performance！

### Major features

- **Modular Design**

  We decompose the matching framework into different components and one can easily construct a customized matching framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary matching frameworks, *e.g.* AcfNet, GANet, GwcNet, PSMNet, GC-Net etc.

- **visualization**
    
  The toolbox provides various visualization, e.g., cost volume / distribution visualization, colorful disparity map and so on.

- **State of the art**

  The toolbox stems from the codebase developed by AcfNet team who ranked 1st at [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) and 3rd at [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) in 2019, and we keep pushing it forward.


### Highlights
- **Multi-GPU training and inference**
- **Mixed precision training:** trains faster with less GPU memory on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores).
- **Batched inference:** can perform inference using multiple images per batch per GPU


## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Get Started

Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of DenseMatchingBenchmark.

## TODO

- [x] inference
- [x] visulization tool(for cost volume)
- [ ] unsupervised mono-depth
- [ ] unsupervised stereo


## Experiment Results

All our reimplemented methods will provide checkpoint(including all training details, e.g., TensorBoard Record) in corresponding config file `ResultOf{ModelName}`

(*x*): means the result in original paper



|        Model       |   FLOPS   | Parameters | FPS  | Time(ms) | [SceneFlow (EPE)][3] | [KITTI 2012][2] | [KITTI 2015 (D1-all)][1] |
|:------------------:|:---------:|:----------:|:----:|:--------:|:---------------:|:----------:|:-------------------:|
|       PSMNet       | 938.186G  |  5.225M    | 1.67 |  599.2   | 1.112 (*1.090*) |            | 2.33  (*2.32*)|
|       AcfNet       | 1239.0G   |  5.559M    | 1.38 |  723.1   | 0.831 (*0.867*) |
|StereoNet-8x-single | 78.512G   |  399.066K  | 19.17|  52.2    | 1.533 (*1.525*) |
|      DeepPruner    |
|       AnyNet       |  1.476G   |  46.987K   |      |          | 3.190 (*~3.2*)  |

## Changelog

Update Time: 24/04/2020

1. Integrate Optical Flow task into our architecture
2. Inference a model is avaliable
3. Unify the format of module's name

Update Time: 10/11/2020

1. Fix a [bug](https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark/blob/master/dmb/modeling/stereo/cmn/cmn.py#L80) in the loss of confidence estimation network, and retrain the AcfNet-Adaptive Network, checkpoints are uploaded [here](https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark/blob/master/configs/AcfNet/ResultOfAcfNet.md)

Update Time: 28/01/2021

1. Add [demo.py](tools/demo.py) to run a demo.
2. Support visualization for cost distribution, run [view_cost.py](tools/view_cost.py) to take a look. We implement an UI for friendly interaction.
3. For detailed HowToUse, please refer to [GETTING_STARTED.md](GETTING_STARTED.md)

## Contributing

We appreciate all contributions to improve DenseMatchingBenchmark. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

DenseMatchingBenchmark is an open source project that is contributed by researchers and engineers from various colledges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new matching algorithm.


## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
    @article{zhang2020adaptive,
      title={Adaptive Unimodal Cost Volume Filtering for Deep Stereo Matching},
      author={Zhang, Youmin and Chen, Yimin and Bai, Xiao and Yu, Suihanjin and Yu, Kun and Li, Zhiwei and Yang, Kuiyuan},
      journal={AAAI},
      year={2020}
    }

```


## Contact

This repo is currently maintained by Youmin Zhang([@youmi-zym](http://github.com/youmi-zym))
 and Yimin Chen ([@Minwell-dht](http://github.com/Minwellcym))
 
## License
DenseMatchingBenchmark is released under the MIT license. See [LICENSE](LICENSE) for additional details.


[1]: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
[2]: http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo
[3]: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html


