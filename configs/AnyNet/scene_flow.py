# ------------------------------------------------------------------------------
# This is an reimplementation of the version written by Wang, Yan
# Original code: https://github.com/mileyan/AnyNet
# ------------------------------------------------------------------------------
import os.path as osp

# the task of the model for, including 'stereo' and 'flow', default 'stereo'
task = 'stereo'

# model settings
max_disp = 192
C = 1

model = dict(
    meta_architecture="AnyNet",
    # max disparity
    max_disp=max_disp,
    # the model whether or not to use BatchNorm
    batch_norm=True,
    stage=['init_guess', 'warp_level_8', 'warp_level_4'],
    backbone=dict(
        type="AnyNet",
        # the in planes of feature extraction backbone
        in_planes=3,
        # the base channels of convolution layer in AnyNet
        C=C,
        # the number of block
        block_num=2,
    ),
    cost_processor=dict(
        # Use the concatenation of left and right feature to form cost volume, then aggregation
        type='AnyNet',
        cost_computation=dict(
            # fast_cat_fms
            type="fast_mode",
            # the maximum disparity of disparity search range under the resolution of feature
            max_disp=dict(
                init_guess=int(max_disp // 16),
                warp_level_8=5,
                warp_level_4=5,
            ),
            # the start disparity of disparity search range
            start_disp=dict(
                init_guess=0,
                warp_level_8=-2,
                warp_level_4=-2,
            ),
            # the step between near disparity sample
            dilation=dict(
                init_guess=1,
                warp_level_8=1,
                warp_level_4=1,
            ),
        ),
        cost_aggregator=dict(
            type="AnyNet",
            # the in planes of cost aggregation sub network
            in_planes=dict(
                init_guess=8*C,
                warp_level_8=4*C,
                warp_level_4=2*C,
            ),
            # the channels of middle 3d convolution layer
            agg_planes=dict(
                init_guess=16,
                warp_level_8=4,
                warp_level_4=4,
            ),
            # the number of middle 3d convolution layer
            num=4,

        ),
    ),
    disp_predictor=dict(
        # default FasterSoftArgmin
        type="FASTER",
        # the maximum disparity of disparity search range
        max_disp=dict(
            init_guess=int(max_disp // 16),
            warp_level_8=5,
            warp_level_4=5,
        ),
        # the start disparity of disparity search range
        start_disp=dict(
            init_guess=0,
            warp_level_8=-2,
            warp_level_4=-2,
        ),
        # the step between near disparity sample
        dilation=dict(
            init_guess=1,
            warp_level_8=1,
            warp_level_4=1,
        ),
        # the temperature coefficient of soft argmin
        alpha=1.0,
        # whether normalize the estimated cost volume
        normalize=True,
    ),
    disp_refinement=dict(
        type='AnyNet',
        # the in planes of disparity refinement sub network
        in_planes=3,
        # the planes of convolution layer for spn
        spn_planes=8,
    ),
    losses=dict(
        l1_loss=dict(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # weights for different scale loss
            weights=(1.0, 1.0, 0.5, 0.25),
            # weight for l1 loss with regard to other loss type
            weight=1.0,
        ),
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
dataset_type = 'SceneFlow'
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
    # whether disparity of datasets is sparse, e.g., SceneFLow is not sparse, but KITTI is sparse
    sparse=False,
    imgs_per_gpu=6,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_train.json'),
        input_shape=[512, 960],
        use_right_disp=False,
        **img_norm_cfg,
    ),
    eval=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_test.json'),
        input_shape=[544, 960],
        use_right_disp=False,
        **img_norm_cfg,
    ),
    # If you don't want to visualize the results, just uncomment the vis data
    vis=dict(
        type=dataset_type,
        data_root=vis_data_root,
        annfile=osp.join(vis_annfile_root, 'vis_test.json'),
        input_shape=[544, 960],
        **img_norm_cfg,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'cleanpass_test.json'),
        input_shape=[544, 960],
        use_right_disp=False,
        **img_norm_cfg,
    ),
)

optimizer = dict(type='RMSprop', lr=5e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20]
)
checkpoint_config = dict(
    interval=2
)
# every n epoch evaluate
validate_interval = 2

log_config = dict(
    interval=10,
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

total_epochs = 20

# each model will return several disparity maps, but not all of them need to be evaluated
# here, by giving indexes, the framework will evaluate the corresponding disparity map
eval_disparity_id = [0, 1, 2, 3]

gpus = 1
dist_params = dict(backend='nccl')
log_level = 'INFO'
validate = True
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = osp.join(root, 'exps/AnyNet/scene_flow')

# For test
checkpoint = osp.join(work_dir, 'epoch_10.pth')
out_dir = osp.join(work_dir, 'epoch_10')
