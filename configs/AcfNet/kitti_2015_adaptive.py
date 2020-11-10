import os.path as osp

# the task of the model for, including 'stereo' and 'flow', default 'stereo'
task = 'stereo'

# model settings
max_disp = 192
model = dict(
    meta_architecture="GeneralizedStereoModel",
    max_disp=max_disp,  # max disparity
    batch_norm=True,  # the model whether or not to use BatchNorm
    backbone=dict(
        type="PSMNet",
        in_planes=3,  # the in planes of feature extraction backbone
    ),
    cost_processor=dict(
        # Use the concatenation of left and right feature to form cost volume, then aggregation
        type='Concatenation',
        cost_computation=dict(
            # default cat_fms
            type="default",
            # the maximum disparity of disparity search range under the resolution of feature
            max_disp=int(max_disp // 4),
            # the start disparity of disparity search range
            start_disp=0,
            # the step between near disparity sample
            dilation=1,
        ),
        cost_aggregator=dict(
            type="AcfNet",
            # the maximum disparity of disparity search range
            max_disp = max_disp,
            # the in planes of cost aggregation sub network
            in_planes=64,
        ),
    ),
    cmn=dict(
        # the number of replicated confidence measure network
        num=3,
        # variance = alpha * ( 1 - confidence ) + beta
        # confidence estimation network coefficient
        alpha=1.0,
        # the lower bound of variance of distribution
        beta=1.0,
        # the in planes of confidence measure network
        in_planes=max_disp,
        losses=dict(
            nll_loss=dict(
                # the maximum disparity of disparity search range
                max_disp=max_disp,
                # weight for confidence loss with regard to other loss type
                weight=8.0,
                # weights for different scale loss
                weights=(1.0, 0.7, 0.5),
            ),
        ),
    ),
    disp_predictor=dict(
        # default FasterSoftArgmin
        type="FASTER",
        # the maximum disparity of disparity search range
        max_disp=max_disp,
        # the start disparity of disparity search range
        start_disp=0,
        # the step between near disparity sample
        dilation=1,
        # the temperature coefficient of soft argmin
        alpha=1.0,
        # whether normalize the estimated cost volume
        normalize=True,
    ),
    losses=dict(
        l1_loss=dict(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # weight for l1_loss with regard to other loss type
            weight=0.1,
            # weights for different scale loss
            weights=(1.0, 0.7, 0.5),
        ),
        focal_loss=dict(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # the start disparity of disparity search range
            start_disp=0,
            # the step between near disparity sample
            dilation=1,
            # weight for stereo focal loss with regard to other loss type
            weight=1.0,
            # weights for different scale loss
            weights=(1.0, 0.7, 0.5),
            # stereo focal loss focal coefficient
            coefficient=5.0,
            # the variance of uni-modal distribution
            variance=None,  # if not given, the variance will be estimated by network
        )
    ),
    eval=dict(
        # evaluate the disparity map within (lower_bound, upper_bound)
        lower_bound=0,
        upper_bound=max_disp,
        # evaluate the disparity map in occlusion area and not occlusion
        eval_occlusion=True,
        # return the cost volume after regularization for visualization
        is_cost_return=False,
        # whether move the cost volume from cuda to cpu
        is_cost_to_cpu=True,
    ),
)

# dataset settings
dataset_type = 'KITTI-2015'
# data_root = 'datasets/{}/'.format(dataset_type)
# annfile_root = osp.join(data_root, 'annotations')

# root = '/home/youmin/'
root = '/node01/jobs/io/out/youmin/'

data_root = osp.join(root, 'data/StereoMatching/', dataset_type)
annfile_root = osp.join(root, 'data/annotations/', dataset_type)

# If you don't want to visualize the results, just uncomment the vis data
# For download and usage in debug, please refer to DATA.md and GETTING_STATED.md respectively.
vis_data_root = osp.join(root, 'data/visualization_data/', dataset_type)
vis_annfile_root = osp.join(vis_data_root, 'annotations')


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
data = dict(
    # if disparity of datasets is sparse, e.g., SceneFLow is not sparse, but KITTI is sparse
    sparse=True,
    imgs_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'full_train.json'),
        input_shape=[256, 768],
        use_right_disp=False,
        **img_norm_cfg,
    ),
    eval=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'full_eval.json'),
        input_shape=[384, 1248],
        use_right_disp=False,
        **img_norm_cfg,
    ),
    # If you don't want to visualize the results, just uncomment the vis data
    vis=dict(
        type=dataset_type,
        data_root=vis_data_root,
        annfile=osp.join(vis_annfile_root, 'vis_test.json'),
        input_shape=[384, 1248],
        **img_norm_cfg,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'full_test.json'),
        input_shape=[384, 1248],
        use_right_disp=False,
        **img_norm_cfg,
    ),
)

optimizer = dict(type='RMSprop', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=100,
    warmup_ratio=1.0,
    gamma=1/3,
    step=[100, 300, 600]
)
checkpoint_config = dict(
    interval=25
)

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

# https://nvidia.github.io/apex/amp.html
apex = dict(
    # whether to use apex.synced_bn
    synced_bn=True,
    # whether to use apex for mixed precision training
    use_mixed_precision=False,
    # the model weight type: float16 or float32
    type="float16",
    # the factor when apex scales the loss value
    loss_scale=16,
)

total_epochs = 600
# every n epoch evaluate
validate_interval = 25

# each model will return several disparity maps, but not all of them need to be evaluated
# here, by giving indexes, the framework will evaluate the corresponding disparity map
eval_disparity_id = [0, 1, 2]

gpus = 4
dist_params = dict(backend='nccl')
log_level = 'INFO'
validate = True
load_from = osp.join(root, 'exps/AcfNet/scene_flow_adaptive/epoch_20.pth')
resume_from = None

workflow = [('train', 1)]
work_dir = osp.join(root, 'exps/AcfNet/kitti_2015_adaptive')

# For test
checkpoint = osp.join(work_dir, 'epoch_600.pth')
out_dir = osp.join(work_dir, 'epoch_600')
