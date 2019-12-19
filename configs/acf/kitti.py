import os.path as osp

# model settings
max_disp = 192
model = dict(
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
            type="PSM",
            in_planes=64,  # the in planes of cost aggregation sub network
        ),
    ),
    # variance = coefficient * ( 1 - confidence ) + init_value
    cmn=dict(
        num=1,
        alpha=1.0,  # confidence estimation network coefficient
        beta=1.0,  # the lower bound of variance of distribution
        losses=dict(
            nll_loss=dict(
                weights=(0.1,),
                weight=1.0,
            ),
        ),
    ),
    disp_predictor=dict(
        mode="default",
        alpha=1.0,  # the temperature coefficient of soft argmin
    ),
    losses=dict(
        l1_loss=dict(
            weights=(0.1,),  # weights for different scale loss
            weight=1.0,
        ),
        focal_loss=dict(
            weights=(0.1,),
            weight=1.0,
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
data_root = osp.join('/node01_data5/StereoMatching/', dataset_type + '_ToBeDel')
# data_root = osp.join('/', dataset_type)
annfile_root = osp.join('/node01/jobs/io/out/youmin/annotations/', dataset_type)

extra_data_root = osp.join('./extra_data/', dataset_type)
extra_annfile_root = osp.join('./extra_data/', dataset_type, 'annotations')

data = dict(
    sparse=False,  # if disparity of datasets is sparse, default dataset is SceneFLow
    imgs_per_gpu=3,
    workers_per_gpu=8,
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
    vis=dict(
        type=dataset_type,
        data_root=extra_data_root,
        annfile=osp.join(extra_annfile_root, 'extra_test.json'),
        input_shape=[544, 960],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
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
num_gpu = 1
device_ids = range(num_gpu)
dist_params = dict(backend='nccl')
log_level = 'INFO'
validate = True
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = ''

# For test
checkpoint = osp.join(work_dir, 'epoch_10.pth')
out_dir = osp.join(work_dir, 'epoch_10')
