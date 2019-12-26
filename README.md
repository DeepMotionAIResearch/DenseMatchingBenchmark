# DenseMatchingBenchmark
This project aims at providing the necessary building blocks for easily
creating dense matching models, e.g. Stereo Matching, Scene Flow using **PyTorch 1.1** or higher.

## Introduction
Our architecture is based on two wellknown detection framework: [mmdetection](https://github.com/open-mmlab/mmdetection) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). By integating their major features, our architecture is suitable for dense matching, and achieves robust performance！

### Major features

- **Modular Design**

  We decompose the matching framework into different components and one can easily construct a customized matching framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary matching frameworks, *e.g.* AcfNet, GANet, GwcNet, PSMNet, GC-Net etc.

- **visualization**

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

- [ ] inference
- [ ] visulization tool(for cost volume)
- [ ] polish archive
- [ ] unsupervised mono-depth
- [ ] unsupervised stereo
- [ ] real-time stereo model

## Experiment Results


## Contributing

We appreciate all contributions to improve DenseMatchingBenchmark. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

DenseMatchingBenchmark is an open source project that is contributed by researchers and engineers from various colledges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new matching algortihm.


## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
    @article{zhang2019adaptive,
      title={Adaptive Unimodal Cost Volume Filtering for Deep Stereo Matching},
      author={Zhang, Youmin and Chen, Yimin and Bai, Xiao and Zhou, Jun and Yu, Kun and Li, Zhiwei and Yang, Kuiyuan},
      journal={arXiv preprint arXiv:1909.03751},
      year={2019}
    }

```


## Contact

This repo is currently maintained by Youmin Zhang([@youmi-zym](http://github.com/youmi-zym))
 and Yimin Chen ([@Minwell-dht](http://github.com/Minwellcym))
## License
DenseMatchingBenchmark is released under the MIT license. See [LICENSE](LICENSE) for additional details.
