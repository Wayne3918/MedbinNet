_base_ = '../MMDetection/configs/queryinst/queryinst_r50_fpn_1x_coco.py'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# learning policy
max_epochs = 36
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

data_root = '../MMDetection/data/MedBin_Dataset/'
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

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r18_fpn_3x_coco/solov2_light_r18_fpn_3x_coco_20220511_083717-75fa355b.pth'
