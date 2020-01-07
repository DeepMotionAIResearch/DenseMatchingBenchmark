import os.path as osp

# model settings
max_disp = 192
model = dict(
    meta_architecture="GeneralizedStereoModel",
    max_disp=max_disp,  # max disparity
    batch_norm=True,  # the model whether or not to use BatchNorm
    backbone=dict(
        conv_body="PSMNet",
        in_planes=3,  # the in planes of feature extraction backbone
        scale=4,  # down-sample scale of the final feature map
    ),
    cost_processor=dict(
        cat_func="default",
        cost_aggregator=dict(
            type="ACF",
            in_planes=64,  # the in planes of cost aggregation sub network
        ),
    ),
    cmn=dict(
        num=3,
        # variance = coefficient * ( 1 - confidence ) + init_value
        # confidence estimation network coefficient
        alpha=1.0,
        # the lower bound of variance of distribution
        beta=1.0,
        losses=dict(
            nll_loss=dict(
                weight=1.0,
                # weights for different scale loss
                weights=(1.0, 0.7, 0.5),
            ),
        ),
    ),
    disp_predictor=dict(
        mode="default",
        alpha=1.0,  # the temperature coefficient of soft argmin
    ),
    losses=dict(
        l1_loss=dict(
            weight=1.0,
            # weights for different scale loss
            weights=(1.0, 0.7, 0.5),
        ),
        focal_loss=dict(
            weight=1.0,
            # weights for different scale loss
            weights=(1.0, 0.7, 0.5),
            coefficient=5.0,
        )
    ),
    eval=dict(
        lower_bound=0,  # evaluate the disparity map within (lower_bound, upper_bound)
        upper_bound=max_disp,
        eval_occlusion=True,  # evaluate the disparity map in occlusion area and not occlusion
        is_cost_return=False,  # return the cost volume after regularization for visualization
        is_cost_to_cpu=True,  # whether move the cost volume from cuda to cpu
    ),
)

# dataset settings
dataset_type = 'SceneFlow'
data_root = 'datasets/{}/'.format(dataset_type)
annfile_root = osp.join(data_root, 'annotations')

data = dict(
    # if disparity of datasets is sparse, default dataset is SceneFLow
    sparse=False,
    imgs_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_train.json'),
        input_shape=[256, 512],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_right_disp=False,
    ),
    eval=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_test.json'),
        input_shape=[544, 960],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_right_disp=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_test.json'),
        input_shape=[544, 960],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_right_disp=False,
    ),
)

optimizer = dict(type='RMSprop', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20]
)
checkpoint_config = dict(interval=1)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

apex = dict(  # https://nvidia.github.io/apex/amp.html
    synced_bn=True,  # whether to use apex.synced_bn
    use_mixed_precision=False,  # whether to use apex for mixed precision training
    type="float16",  # the model weight type: float16 or float32
    loss_scale=16,  # the factor when apex scales the loss value
)

total_epochs = 20

num_gpu = 4
device_ids = range(num_gpu)
dist_params = dict(backend='nccl')

log_level = 'INFO'
validate = True
load_from = None
resume_from = None

workflow = [('train', 1)]
work_dir = '/data/exps/stereo/acfnet-sf'

# For test
checkpoint = osp.join(work_dir, 'epoch_10.pth')
out_dir = osp.join(work_dir, 'epoch_10')
