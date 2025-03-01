_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]
model = dict(
    type='DINO',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        num_classes=37,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

data_root = './data/medbin_0716/'
metainfo = {
    'classes': ('Bloody_objects', 'Electronic_thermometer', 'Mask', 'N95',
                'Oxygen_cylinder', 'Radioactive_objects', 'bandage', 'blade', 'capsule', 'cotton_swab', 'covid_buffer',
                'covid_buffer_box', 'covid_test_case', 'drug_packaging', 'gauze', 'glass_bottle', 'harris_uni_core', 'harris_uni_core_cap', 
                'iodine_swab', 'medical_gloves', 'medical_infusion_bag', 'mercury_thermometer', 'needle', 'paperbox', 'pill', 'plastic_medical_bag', 'plastic_medical_bottle',
                'reagent_tube', 'reagent_tube_cap', 'scalpel', 'single_channel_pipette', 'syringe', 'transferpettor_glass', 
                'transferpettor_plastic', 'tweezer_metal', 'tweezer_plastic', 'unguent'
                 ),
    'palette': [
        (13, 35, 78), (215, 72, 36), (45, 89, 123), (162, 210, 29),
        (78, 44, 200), (33, 180, 92), (245, 19, 55), (92, 255, 105),
        (201, 63, 11), (150, 38, 235), (18, 88, 72), (73, 245, 175),
        (234, 11, 95), (100, 200, 10), (55, 30, 180), (210, 160, 88),
        (17, 69, 142), (190, 33, 255), (88, 255, 77), (230, 120, 25),
        (29, 55, 220), (160, 66, 33), (77, 190, 10), (235, 88, 130),
        (125, 245, 42), (42, 17, 210), (255, 130, 88), (20, 160, 29),
        (210, 75, 210), (90, 180, 60), (33, 120, 240), (175, 50, 20),
        (55, 210, 130), (250, 85, 5), (120, 30, 170), (215, 160, 70),
        (11, 80, 245), (155, 255, 44), (68, 23, 140), (240, 95, 200)
    ]
}
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')))
test_dataloader = val_dataloader

# max_epochs = 50
# interval = 5

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'
