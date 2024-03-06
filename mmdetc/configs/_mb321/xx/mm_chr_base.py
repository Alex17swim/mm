# mm_chr_base.py
from pt_flag import agCOCO, BB_SEG, LOG_GAP_VAL, NUM_WORKERS_SEG, SAMPLES_PER_GPU_SEG, SEG_PRE_TRAIN, MM_PTH_TRAINED,\
    SEG_RESUME_TRAIN, FP_RESUME_SEG, input3
from pt_flag import IMG_NORM_CFG as img_norm_cfg
'''
from pt_flag import NUM_CLASS_SEG, MM_PTH_TRAINED, BB_SEG #, MM_PTH_TRAINED,
from mmdetection.configs.mb321.mm_chr_base import dataset_type as dataset_type
from mmdetection.configs.mb321.mm_chr_base import data as data
from mmdetection.configs.mb321.mm_chr_base import evaluation as evaluation
from mmdetection.configs.mb321.mm_chr_base import log_config as log_config
from mmdetection.configs.mb321.mm_chr_base import init_cfg as init_cfg
load_from = MM_PTH_TRAINED
## if depth != 50: input(f"?? 'defrom_' expect depth=50, not{depth}")
'''
dataset_type = agCOCO.name # !!!
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# if BB_SEG == 'mm_pr_3x':img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)  # use caffe img_norm
size_divisor = 32
_img_scale=[(400, 1333), (500, 1333), (600, 1333)]
if 'deform' in BB_SEG:
    size_divisor = 1
    _img_scale=[(400, 4200), (500, 4200), (600, 4200)]
    print(f"---set size_divisor={size_divisor} for {BB_SEG}")

if BB_SEG[len('mm_'):len('mm_swin_')] in ['swin_', 'pr_3x', 'mrcnn', 'retina']: # mask
    print(f"--- using mask for:{BB_SEG}")
    with_mask = True
    keys = ['img', 'gt_bboxes', 'gt_labels', 'gt_masks']
    metric=['bbox', 'segm']
else:
    print(f"--- ONLY using bbox for {BB_SEG}")
    with_mask = False
    keys = ['img', 'gt_bboxes', 'gt_labels']
    metric=['bbox']

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=with_mask),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment', # for swin, augmentation strategy originates from DETR / Sparse RCNN
        policies=[[dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value', keep_ratio=True)
                ],
                [dict(type='Resize', img_scale=_img_scale,
                multiscale_mode='value', keep_ratio=True),
                dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                dict(type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333),
                           (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value', override=True, keep_ratio=True)]]
        ), # AutoAugment
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=size_divisor),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=keys)    
] # end train_pipeline

# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(1333, 800), flip=False,
        transforms=[dict(type='Resize', keep_ratio=True), dict(type='RandomFlip'), dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1), dict(type='ImageToTensor', keys=['img']), dict(type='Collect', keys=['img'])
        ])
] # test_pipeline

# data_spa0 = dict(train=dict(pipeline=train_pipeline))  # pass
data = dict(
    samples_per_gpu=SAMPLES_PER_GPU_SEG, # OK on 2 with batch=69; failed on 1 for with=128
    workers_per_gpu=NUM_WORKERS_SEG, # 2
    train=dict(filter_empty_gt=False, pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)) # data
evaluation = dict(interval=LOG_GAP_VAL, metric=metric)
log_config = dict(interval=5) # gap in training to print/save log 
# checkpoint_config = dict(interval=LOG_GAP_VAL) # gap in training to save '.pth'

# use ResNet as backbone: detr50/deform50; 50/101: spa,faster/mrcnn
if 'faster_' in BB_SEG or 'mrcnn_' in BB_SEG or 'spa_' in BB_SEG or 'retina_' in BB_SEG:
    depth = 50 if '50' in BB_SEG else 101 if '101' in BB_SEG else -1
elif 'swin_' in BB_SEG:
    depth = [2, 2, 18, 2] if 'swin_s_' in BB_SEG else [2, 2, 6, 2] # swin_s_fp16
else: # 'detr', 'deform_', 'pr_'
    depth = 50
print(f"name={BB_SEG}, depth={depth}")
if SEG_PRE_TRAIN:
    # Downloading: "https://download.pytorch.org/models/resnet101-63fe2227.pth"
    # to /public/home/alex/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth
    if 'swin' in BB_SEG:
        _tm = 'tiny' if 'swin_t_' in BB_SEG else 'small' if 'swin_s_' in BB_SEG else 'xx'    
        _ck = f'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_{_tm}_patch4_window7_224.pth'
    elif 'pr_3x' in BB_SEG: _ck = 'open-mmlab://detectron2/resnet50_caffe'
    else: _ck = f'torchvision://resnet{depth}'
    init_cfg = dict(type='Pretrained', checkpoint=_ck)
else: init_cfg = None
print(f"Pretrained with init_cfg={init_cfg}")
# if SEG_RESUME_TRAIN: # set in chr4.py --> args.resume_from
#     from all_include.pt_head import find_sth_easy
#     load_from = find_sth_easy(FP_RESUME_SEG, [BB_SEG])
#     print(f"--- resume with load_from={load_from}")
# else:
load_from = MM_PTH_TRAINED

''' 2x
'mm_retina_50_2x':  "retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth",
'mm_retina_101_2x': "retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth",
faster 50:  faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
faster 101: faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth
mrcnn 50:   mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth
mrcnn 101:  mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth

detr:       detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth
defor base: deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth
defor iter: deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth
defor two:  deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth

spar\_50\_100\_1x: box AP=37.9, sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth
spar 50 100:    sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.pth
spar 50 300:    sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth
spar 101 100:   sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth
spar 101 300:   sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth

swin_t_base:    mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth
swin t crop:    mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth
swin t fp16:    mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth
swin s fp16:    mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth
'''