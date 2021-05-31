## Installation

### Requirements:
- PyTorch1.7.1+, Python3.8+, Cuda11.0+
- mmcv==1.3.5

We have verified with Pytorch==1.7.1, the code can train/val/test normally. But with limited GPU resources, experiments are not runned again.

So, if you find worse result, you can checkout to 177c56ca1952f54d28e6073afa2c16981113a2af
```bash
cd $DenseMatchingBenchmark
git checkout 177c56ca1952f54d28e6073afa2c16981113a2af
```

### Option 1: Step-by-step installation

a. Create a conda virtual environment and activate it. Then install required packages.


```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name dense_matching_benchmark python=3.8
conda activate dense_matching_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install pip

# install required packages from requirements.txt
pip install -r requirements.txt
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/).

c0. Install detectron2
```bash
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
```

c1. Install apex
```bash
# optional step:
export CUDA_HOME=/usr/local/cuda-x.x/
# where x.x corresponds to your CUDA version used to install pytorch

git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
*Potential ISSUE*: [complie error](https://github.com/NVIDIA/apex/issues/802#issuecomment-618699214) 
```
# optional step: 
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
```

c2. Install [spatial-correlation-sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension)
```bash
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension
git checkout fix_1.7
cd Pytorch-Correlation-extension
python setup.py install
```


d. Clone the DenseMatchingBenchmark repository.

```bash
git clone https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark.git
cd DenseMatchingBenchmark
```

e. Install DenseMatchingBenchmark(other dependencies will be installed optionally).
```bash
# libs include: dmb, spn

# the $1 can be: 'all', 'dmb', 'spn'
# => install all libs or specific lib, e.g. dmb

# the $2 can be: 'install'
# => if 'install' given, the libs will be installed into site-packages
# => if not given, the libs will be install with symbolic links,
# => so that you can modify the files if you want and won't need to re-build it

bash INSTALL.sh $1 $2

# recommend install instruction:

bash INSTALL.sh all

```

### Prepare data

Data prepare please refer to [DATA.md](DATA.md)



### Notice
You can run `python(3) setup.py develop` or `pip install -e .` to install DenseMatchingBenchmark if you want to make modifications to it frequently.

If there are more than one DenseMatchingBenchmark on your machine, and you want to use them alternatively.
Please insert the following code to the main file
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```
or run the following command in the terminal of corresponding folder.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```



