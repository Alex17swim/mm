print(f" *here is mm_cfg: toy_mrcnn.py") # from mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    'mb321_base_detc.py'
    # '../_base_/datasets/_mb321_coco_instance.py', # coco_instance
    # '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


# print(f" *_mb321/toy_mrcnn.py\n")