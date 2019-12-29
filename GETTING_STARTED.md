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

**\*Important\***: The default learning rate in config files is for 8 GPUs and 2 img/gpu (batch size = 8*2 = 16).
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.01 for 4 GPUs * 2 img/gpu and lr=0.08 for 16 GPUs * 4 img/gpu.

### Train with a single GPU

```shell
python tools/train.py PATH_TO_CFG_FILE --gpus 1 --launcher none --validate
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS \
        tools/train.py PATH_TO_CFG_FILE --launcher pytorch --validate --gpus NUM_GPUS
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 1, which can be modified like [this](configs/PSMNet/)) epochs during the training.
- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.

Difference between `resume_from` and `load_from`:
`resume_from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load_from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.


## How-to

### Use my own datasets


### Develop new components
