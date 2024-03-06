print(f" *here is mm_cfg: mb321_base_detc.py") # from centernet/centernet-update_r50-caffe_fpn_ms-1x_coco

_base_ = [
    # '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/_mb321_coco_instance.py', # coco_instance
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# import os.path as osp
from MB321.mm_flag import agMM, agDATA, agDIR, agTAG # , agLOG
# from mmengi.mmengine.config.lazy import LazyObject
# osp = LazyObject('os.path')
max_epochs, val_interval, fp_pth = agDATA.epoch_seg, agMM.gap_tra, agDIR.pth_all_seg # agDIR.out_seg
data_root, outfile_prefix = agDIR.root_seg+'/', agDIR.out_seg+f'/{agTAG.bb_seg}{agTAG.bug_seg}'
seed, metric, scale_wh, _test_dir = agMM.seed, ['bbox', 'segm'], agMM.size_seg_wh, 'test_img' # test_img, 'val_coco' #
batch_size = 2 if scale_wh[0] <= 1333 else 1
randomness = dict(seed=seed)
pretrained = agMM.pre_pth_seg
load_from = pretrained # f"/public/home/alex/Docu/Dataset/trained/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth" # None
if (batch_size > 2) and ('swin' in agTAG.bb_seg): input(f" !! batch_size={batch_size} seems to big for swin-t, expect 1 or 2 ..\n\n")

# log_level=logging.ERROR
log_level = 'INFO' # choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
# log_file = agLOG.val_seg # set this in "mm/mmengi/mmengine/runner --> line 401:self.build_logger"
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=30, out_dir=fp_pth),
    visualization=dict(type='DetVisualizationHook')
) # default_hooks

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=scale_wh, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=scale_wh, keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    # If you don't have a gt annotation, delete the pipeline
    # dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=scale_wh, keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=batch_size,
    # num_workers=num_workers, persistent_workers=pw,
    dataset=dict( # type=dataset_type,
        data_root=data_root, ann_file='train_coco/annotations.json', data_prefix=dict(img='train_coco/'), pipeline=train_pipeline))
val_dataloader = dict(batch_size=batch_size,
    dataset=dict(data_root=data_root, ann_file='val_coco/annotations.json', data_prefix=dict(img='val_coco/'), pipeline=val_pipeline))
test_dataloader = dict(batch_size=batch_size,
    dataset=dict(data_root=data_root, ann_file=f'{_test_dir}/annotations.json', data_prefix=dict(img=f'{_test_dir}/'), pipeline=test_pipeline))
# test_dataloader = dict(batch_size=batch_size,
#     dataset=dict(data_root=data_root, ann_file=None, data_prefix=dict(img='test_img/'), pipeline=test_pipeline))

val_evaluator = dict(ann_file=data_root + 'val_coco/annotations.json', metric=metric, outfile_prefix=outfile_prefix)
# test_evaluator = dict(ann_file=data_root + f'{_test_dir}/annotations.json', metric=metric, outfile_prefix=outfile_prefix)
test_evaluator = dict(_delete_=True, # val_evaluator # dict(_delete_=True)
    type='CocoMetric', ann_file=data_root + 'val_coco/annotations.json', metric=metric,
    classwise=True, format_only=True,
    backend_args=None, outfile_prefix=outfile_prefix)
train_cfg = dict(max_epochs=max_epochs, val_interval=val_interval) # type='EpochBasedTrainLoop',
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))



# model = dict(
#     type='CenterNet',
#     # use caffe img_norm
#     data_preprocessor=dict(
#         type='DetDataPreprocessor',
#         mean=[103.530, 116.280, 123.675],
#         std=[1.0, 1.0, 1.0],
#         bgr_to_rgb=False,
#         pad_size_divisor=32),
#     backbone=dict(
#         type='ResNet',
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=False),
#         norm_eval=True,
#         style='caffe',
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='open-mmlab://detectron2/resnet50_caffe')),
#     neck=dict(
#         type='FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         start_level=1,
#         add_extra_convs='on_output',
#         num_outs=5,
#         # There is a chance to get 40.3 after switching init_cfg,
#         # otherwise it is about 39.9~40.1
#         init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
#         relu_before_extra_convs=True),
#     bbox_head=dict(
#         type='CenterNetUpdateHead',
#         num_classes=80,
#         in_channels=256,
#         stacked_convs=4,
#         feat_channels=256,
#         strides=[8, 16, 32, 64, 128],
#         hm_min_radius=4,
#         hm_min_overlap=0.8,
#         more_pos_thresh=0.2,
#         more_pos_topk=9,
#         soft_weight_on_reg=False,
#         loss_cls=dict(
#             type='GaussianFocalLoss',
#             pos_weight=0.25,
#             neg_weight=0.75,
#             loss_weight=1.0),
#         loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
#     ),
#     train_cfg=None,
#     test_cfg=dict(
#         nms_pre=1000,
#         min_bbox_size=0,
#         score_thr=0.05,
#         nms=dict(type='nms', iou_threshold=0.6),
#         max_per_img=100))

# # single-scale training is about 39.3
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='RandomChoiceResize',
#         scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
#                 (1333, 768), (1333, 800)],
#         keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]

# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# # learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.00025,
#         by_epoch=False,
#         begin=0,
#         end=4000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]

# optim_wrapper = dict(
#     optimizer=dict(lr=0.01),
#     # Experiments show that there is no need to turn on clip_grad.
#     paramwise_cfg=dict(norm_decay_mult=0.))

# # NOTE: `auto_scale_lr` is for automatically scaling LR,
# # USER SHOULD NOT CHANGE ITS VALUES.
# # base_batch_size = (8 GPUs) x (2 samples per GPU)
# auto_scale_lr = dict(base_batch_size=16)
