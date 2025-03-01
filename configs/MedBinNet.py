# The new config inherits a base config to highlight the necessary modification
_base_ = '../MMDetection/configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        deepen_factor=0.125,
        widen_factor=0.25,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.')),
    neck=dict(in_channels=[64, 128, 256], out_channels=64, num_csp_blocks=1),
    bbox_head=dict(
        in_channels=64, 
        feat_channels=64,
        num_classes=37,
    ))

# Modify dataset related settings
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

max_epochs = 200
interval = 5

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'
