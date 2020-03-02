import os.path as osp
# max search radium for each level
max_displacement = 4
# the task of the model for
task = 'flow'

model = dict(
    meta_architecture='PWCNet',
    # the model whether or not to use BatchNorm
    batch_norm=False,
    # the stage in coarse-to-fine
    stage=['init_guess', 'warp_level_32', 'warp_level_16', 'warp_level_8', 'warp_level_4'],
    backbone=dict(
        type="HMNet",
        # the in planes of feature extraction backbone
        in_planes=3,
    ),
    cost_processor=dict(
        # Use the concatenation of left and right feature to form cost volume, then aggregation
        type='HMNet',
        # enable coarse-to-fine
        coarse_to_fine=True,
        # enable global matching
        global_matching=True,
        # enable residual refinement for each level
        residual=True,
        cost_computation=dict(
            # correlation
            type="correlation",
            # maximum displacement when searching matching pixel
            max_displacement=max_displacement,
        ),
        cost_aggregator=dict(
            type="HMNet",
            # whether use dense connections
            dense=True,
            # whether server each stage as residual refinement
            residual=True,
            # the channels of intermediate convolution layers
            agg_planes_list = [128, 128, 96, 64, 32],
            # the in planes of cost aggregation network in coarse-to-fine branch
            in_planes=dict(
            # left image planes + global cost volume planes + correlation-based cost volume planes + others
                init_guess=196+128+(max_displacement*2+1)**2,
                warp_level_32=128+128+4+(max_displacement*2+1)**2,
                warp_level_16=96+128+4+(max_displacement*2+1)**2,
                warp_level_8=64+128+4+(max_displacement*2+1)**2,
                warp_level_4=32+128+4+(max_displacement*2+1)**2,
            ),
            # the out planes of cost aggregation network in coarse-to-fine branch
            out_planes=512,
            # the in planes of cost aggregation network in global information branch
            global_in_planes=dict(
                init_guess=196,
                warp_level_32=128+4,
                warp_level_16=96+4,
                warp_level_8=64+4,
                warp_level_4=32+4,
            ),
            # the out planes of cost aggregation network in global information branch
            global_out_planes=128,
        ),
    ),
    flow_predictor=dict(
        type="HMNet",
        # in channels of cost volume
        in_planes=512,
    ),
    flow_refinement=dict(
        type='HMNet',
        # the in planes of disparity refinement sub network
        in_planes=512,
    ),
    losses=dict(
        p_norm_loss=dict(
            # p norm
            p=2.0,
            # a balance factor with absolute difference between estimated and ground truth flow
            epsilon=0.0,
            # weights for different scale loss
            weights=(0.125, 0.125, 0.25, 0.5, 1.0),
            # weight for l1 loss with regard to other loss type
            weight=1.0,
        ),
    ),
    eval=dict(
        # evaluate the disparity map in occlusion area and not occlusion
        eval_occlusion=True,
    ),
)


# dataset settings
dataset_type = 'FlyingChairs'

root = '/home/youmin/'
# root = '/node01/jobs/io/out/youmin/'

data_root = osp.join(root, 'data/OpticalFlow/', dataset_type)
annfile_root = osp.join(root, 'data/annotations/', dataset_type)

# If you don't want to visualize the results, just uncomment the vis data
# For download and usage in debug, please refer to DATA.md and GETTING_STATED.md respectively.
vis_data_root = osp.join(root, 'data/visualization_data/', dataset_type)
vis_annfile_root = osp.join(vis_data_root, 'annotations')


data = dict(
    # whether flow of dataset is sparse, e.g., SceneFLow is not sparse, but KITTI is sparse
    sparse=False,
    imgs_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'train.json'),
        input_shape=[320, 448],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    eval=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'eval.json'),
        input_shape=[384, 512],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    # If you don't want to visualize the results, just uncomment the vis data
    vis=dict(
        type=dataset_type,
        data_root=vis_data_root,
        annfile=osp.join(vis_annfile_root, 'vis_test.json'),
        input_shape=[384, 512],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        annfile=osp.join(annfile_root, 'test.json'),
        input_shape=[384, 512],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
)

optimizer = dict(type='RMSprop', lr=1e-3)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='Step',
    by_epoch=True,  # if False, by iteration; if True, by epoch
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    gamma=1.0 / 2,
    step=[40, 60, 80, 100]  # StepLrUpdate
)
checkpoint_config = dict(
    interval=5
)
# every n epoch evaluate
validate_interval = 5

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
    synced_bn=False,
    # whether to use apex for mixed precision training
    use_mixed_precision=False,
    # the model weight type: float16 or float32
    type="float16",
    # the factor when apex scales the loss value
    loss_scale=16,
)

total_epochs = 100

# each model will return several flow maps, but not all of them need to be evaluated
# here, by giving indexes, the framework will evaluate the corresponding flow map
eval_flow_id = [0, 1, 2, 3, 4]

gpus = 8
dist_params = dict(backend='nccl')
log_level = 'INFO'
validate = True
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = osp.join(root, 'exps/PWCNet/flying_chairs')

# For test
checkpoint = osp.join(work_dir, 'epoch_100.pth')
out_dir = osp.join(work_dir, 'epoch_100')



