# Getting Started

This page provides basic tutorials about the usage of DenseMatchingBenchmark.
For installation instructions, please see [INSTALL.md](INSTALL.md).


## Train a model

DenseMatchingBenchmark implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

* The log file will conclude: environmental info, config file content, argument parse info, training info (printed loss etc.)
* The tensorboardX will log scalar and image info.

**Important**: The default learning rate in config files is for 8 GPUs and 2 img/gpu (batch size = 8*2 = 16).
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.01 for 4 GPUs * 2 img/gpu and lr=0.08 for 16 GPUs * 4 img/gpu.

### Train with a single GPU

```shell
python tools/train.py PATH_TO_CFG_FILE --gpus 1 --launcher none --validate
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
python -m torch.distributed.launch --master_port $PORT --nproc_per_node=$NGPUS \
        tools/train.py $PATH_TO_CONFIG_FILE --launcher pytorch --validate --gpus $NGPUS
```

For example, using 4 GPUs for training:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 5000 --nproc_per_node=4 \
        tools/train.py configs/PSMNet/scene_flow.py --launcher pytorch --validate --gpus 4
```

Note:

- `--master_port`: the port used when training, please set different $PORT for each distributed training.

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 1, which can be modified like [this](configs/PSMNet/)) epochs during the training.
- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--CUDA_VISIBLE_DEVICES`: Specify the particular GPUs ID, for detailed usage please google.

Difference between `resume_from` and `load_from`:
`resume_from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load_from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.


## Test a model


### Test with multiple GPUs

Aim:

1. Evaluation on public datasets (e.g., Scene Flow, KITTI) and visualization of the results.
2. Submission to benchmark. For KITTI, the visualization saved in the `out_dir` including the `disp_0` which can be directly used for submission.
3. Just want to see the visualized results of your own dataset.

```shell
python -m torch.distributed.launch --master_port $PORT --nproc_per_node=$NGPUS \
        tools/test.py $PATH_TO_CONFIG_FILE --launcher pytorch --validate --gpus $NGPUS --show $SHOW
```

For example, using 4 GPUs for testing:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 5000 --nproc_per_node=4 \
        tools/test.py configs/PSMNet/scene_flow.py --launcher pytorch --validate --gpus 4 --show False
```

Note:

- `--master_port`: the port used when testing, please set different $PORT for each distributed testing.

Optional arguments are:

- `--checkpoint ${CHECKPOINT_FILE}`: Override the checkpoint file specified in config file. This checkpoint file is the model weight you want to load for testing.
- `--out_dir ${WORK_DIR}`: Override the output directory specified in the config file.
- `--validate`(**strongly recommended**): Perform evaluation during the testing.
- `--show ${BOOL}`(**strongly recommended**): Whether save the outputted results to local or not. For huge dataset (e.g., Scene Flow), it is not recommended. Because it will takes vast storage. But for small dataset (e.g., KITTI), it is recommended.
- `--CUDA_VISIBLE_DEVICES`: Specify the particular GPUs ID, for detailed usage please google.


## Inference a model

The [Inference API](dmb/apis/inference.py) has been provided. Supporting resize, pad or crop the images for inference. 

### Run a demo

For undertanding our benchmark more conveniently, we provide a demo.py in tools subdirectory. To run it, you first have to download a checkpoint, e.g., you can download our AcfNet checkpoint from [here](https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark/blob/master/configs/AcfNet/ResultOfAcfNet.md#sceneflow).
```shell
cd tools
chmod +x demo.sh
# please reconfigure some args in deme.sh to satisfy your system
./demo.sh

```

**Optional**:
If you also want to visualize the cost distribution of some network, such as AcfNet, you can also run a UI for friendly interaction.
```
cd tools
python view_cost.py
```



## Others

### Tricks for debug

To be honest, the huge framework is kind of hard to debug. Here, we give some tricks for debug.

The direct idea is to prepare a small dataset, and set it as train/eval/vis/test data. Then, it's really quick for model to run a full epoch. Therefore, it's very convenient for you to check the whole running state and find bugs.

As we have prepare a small dataset -- visualization data (please refer to details in [DATA.md](DATA.md)) for you, you can just use it for debug.

**But**: a small dataset with several images will throw a error if you use multi-gpu for debug, so, **only one gpu** is recommended!

If you have any convenient debug tricks, please let me know.

### Use my own datasets

You can prepare own datasets as the visualization data (please refer to details in [DATA.md](DATA.md)).


### Develop new components

