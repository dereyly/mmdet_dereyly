# model settings
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='DoubleHeadReID',
    pretrained=  # NOQA
    'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',  # NOQA
    backbone=dict(
        type='Res2Net',
        depth=50,
        scale=4,
        baseWidth=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,
        style='pytorch',
        dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        # gcb=dict(ratio=1. / 16., ),
        # stage_with_gcb=(False, True, True, True),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        ),
    rpn_head=dict(
        type='GARPNHead',
        in_channels=256,
        feat_channels=256,
        octave_base_scale=8,
        scales_per_octave=3,
        octave_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        anchor_base_sizes=None,
        anchoring_means=[.0, .0, .0, .0],
        anchoring_stds=[0.07, 0.07, 0.14, 0.14],
        target_means=(.0, .0, .0, .0),
        target_stds=[0.07, 0.07, 0.11, 0.11],
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    reg_roi_scale_factor=1.3,
    bbox_head=dict(
        type='DoubleConvFCBBoxHeadReId',
        num_convs=4,
        num_fcs=2,
        in_channels=256,
        conv_out_channels=1024,
        fc_out_channels=128,
        roi_feat_size=7,
        num_classes=81,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        ga_assigner=dict(
            type='ApproxMaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        ga_sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        center_ratio=0.2,
        ignore_ratio=0.5,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=400,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.55,
            neg_iou_thr=0.55,
            min_pos_iou=0.55,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=350,
            pos_fraction=0.35,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=320,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=1e-4, nms=dict(type='nms', iou_thr=0.6), max_per_img=200))
# dataset settings
dataset_type = 'CocoDataset'
# data_root = '/root/COCO/'
data_root = '/media/dereyly/ssd_big/ImageDB/waymo'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
     dict(
        type='Resize',
        img_scale=[(2000, 600), (2000, 1300)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomCrop', crop_size=(1184, 1184)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(5000, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=[data_root + '/train/annotations/train.json','/media/dereyly/ssd_big/ImageDB/waymo_full/annotations/val.json'],
        img_prefix=[data_root + '/train/images/','/media/dereyly/ssd_big/ImageDB/waymo_full/images/'],
        pipeline=train_pipeline),
    # val=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_val2017.json',
    #     img_prefix=data_root + 'val2017/',
    #     pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + '_val/annotations/val_small_rnd.json',
        # img_prefix=data_root + '_val/images/',
        ann_file='/media/dereyly/data_ssd/ImageDB/waymo_v2/val/annotations/val_small_rnd3.json',
        img_prefix='/media/dereyly/data_ssd/ImageDB/waymo_v2/val/images/',
        pipeline=test_pipeline))
# # optimizer
# optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type='Adam',
    lr=.0001
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    gamma=0.25,
    warmup_iters=500,
    warmup_ratio=1.0 / 32,
    step=[5, 8])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 8
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/media/dereyly/data/models/waymo/denc_reid2'
load_from = '/media/dereyly/data/models/waymo/denc_reid/epoch_2.pth'
resume_from = None
workflow = [('train', 1)]
