import os.path as osp

# model settings
max_disp = 192
C = 8

model = dict(
    meta_architecture="MonoStereo",
    # max disparity
    max_disp=max_disp,
    # the model whether or not to use BatchNorm
    batch_norm=True,
    backbone=dict(
        type="MonoStereo",
        # the in planes of feature extraction backbone
        in_planes=3,
        # the base channels of convolution layer in AnyNet
        C=C,
    ),
    disp_sampler=dict(
        type='MONOSTEREO',
        # max disparity
        max_disp=192,
        # the down-sample scale of the input feature map
        scale=4,
        # the number of diaparity samples
        disparity_sample_number=8,
        # the in planes of extracted feature
        in_planes=2*C,
        # the base channels of convolution layer in this network
        C=C,
    ),
    cost_processor=dict(
        type='CAT',
        cost_computation = dict(
            # default fast_cat_fms
            type="fast_mode",
        ),
        cost_aggregator=dict(
            type="MONOSTEREO",
            # the in planes of cost aggregation sub network,
            in_planes=4*C,
            # the base channels of convolution layer in this network
            C=C,
            # the number of diaparity samples
            disparity_sample_number=8,
        ),
    ),
    cmn=dict(
        # the number of replicated confidence measure network
        num=1,
        # variance = alpha * ( 1 - confidence ) + beta
        # confidence estimation network coefficient
        alpha=1.0,
        # the lower bound of variance of distribution
        beta=1.0,
        # the in planes of confidence measure network
        in_planes=max_disp // 4,
        losses=dict(
            nll_loss=dict(
                # the maximum disparity of disparity search range
                max_disp=max_disp,
                # the start disparity of disparity search range
                start_disp=0,
                # weight for confidence loss with regard to other loss type
                weight=120.0,
                # weights for different scale loss
                weights=(1.0, ),
            ),
        ),
    ),
    disp_predictor=dict(
        # default SoftArgmin
        type='DEFAULT',
        # the temperature coefficient of soft argmin
        alpha=1.0,
        # whether normalize the estimated cost volume
        normalize=True,

    ),
    disp_refinement=dict(
        type='MONOSTEREO',
        # the in planes of disparity refinement sub network
        in_planes=3,
        # the number of edge aware refinement module
        num=1,
        # the base channels of convolution layer in this network
        C=C,
    ),
    losses=dict(
        l1_loss=dict(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # weights for different scale loss
            weights=(1.0, 0.7, 0.5),
            # weight for l1 loss with regard to other loss type
            weight=1.0,
        ),
        focal_loss=dict(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # the start disparity of disparity search range
            start_disp=0,
            # weight for stereo focal loss with regard to other loss type
            weight=5.0,
            # weights for different scale loss
            weights=(1.0, ),
            # stereo focal loss focal coefficient
            coefficient=5.0,
            # the variance of uni-modal distribution
            variance=None, # 1.2,  # if not given, the variance will be estimated by network
        ),
        relative_loss=dict(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # the start disparity of disparity search range
            start_disp=0,
            # weight for stereo focal loss with regard to other loss type
            weight=10.0,
            # weights for different scale loss
            weights=(1.0, 1.0),
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
dataset_type = 'SceneFlow'

# root = '/home/youmin/'
root = '/node01/jobs/io/out/youmin/'

# data_root = osp.join(root, 'data/StereoMatching/', dataset_type)
data_root = '/SceneFlow'
annfile_root = osp.join(root, 'data/annotations/', dataset_type)

# If you don't want to visualize the results, just uncomment the vis data
# For download and usage in debug, please refer to DATA.md and GETTING_STATED.md respectively.
vis_data_root = osp.join(root, 'data/visualization_data/', dataset_type)
vis_annfile_root = osp.join(vis_data_root, 'annotations')


data = dict(
    # whether disparity of datasets is sparse, e.g., SceneFLow is not sparse, but KITTI is sparse
    sparse=False,
    imgs_per_gpu=4,
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
    # If you don't want to visualize the results, just uncomment the vis data
    vis=dict(
        type=dataset_type,
        data_root=vis_data_root,
        annfile=osp.join(vis_annfile_root, 'vis_test.json'),
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
    gamma=1/2,
    step=[20, 40, 60, 64],
)

checkpoint_config = dict(
    interval=4
)
# every n epoch evaluate
validate_interval = 4

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

total_epochs = 64

# each model will return several disparity maps, but not all of them need to be evaluated
# here, by giving indexes, the framework will evaluate the corresponding disparity map
eval_disparity_id = [0, 1, 2, 3, 4]

gpus = 4
dist_params = dict(backend='nccl')
log_level = 'INFO'
validate = True
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = osp.join(root, 'exps/MonoStereo/scene_flow_correlation_C8_relativeSigmoid_NoLinearConfMinMax100Loss')

# For test
checkpoint = osp.join(work_dir, 'epoch_64.pth')
out_dir = osp.join(work_dir, 'epoch_64')
