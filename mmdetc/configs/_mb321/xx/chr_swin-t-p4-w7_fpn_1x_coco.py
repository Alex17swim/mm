# chr_swin-t-p4-w7_fpn_1x_coco.py
print("where is chr_swin-t-p4-w7_fpn_1x_coco.py? __file__={}".format(__file__))
from pt_flag import TM, SEG_SWIN_TYPE, NUM_CLASS_SEG, BB_SEG #, MM_PTH_TRAINED,
from mmdetection.configs.mb321.mm_chr_base import dataset_type as dataset_type
from mmdetection.configs.mb321.mm_chr_base import data as data
from mmdetection.configs.mb321.mm_chr_base import evaluation as evaluation
from mmdetection.configs.mb321.mm_chr_base import log_config as log_config
from mmdetection.configs.mb321.mm_chr_base import init_cfg as init_cfg
from mmdetection.configs.mb321.mm_chr_base import load_from as load_from
from mmdetection.configs.mb321.mm_chr_base import depth as depth
# if depth != 50: input(f"?? 'defrom_' expect depth=50, not{depth}")
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance_mb321.py',
    '../_base_/schedules/mb321_schedule.py',
    '../_base_/mb321_runtime.py'
]

# swin_s_fp16, swin_t_crop, swin_t_fp16, swin_t_base
# depths_list= [2, 2, 18, 2] if 'swin_s_fp16' == BB_SEG  else [2, 2, 6, 2]
if 'fp16' in BB_SEG:
    print(f"--- set fp16 for:{BB_SEG}")
    fp16 = dict(loss_scale=dict(init_scale=512))

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type = SEG_SWIN_TYPE, # 'SwinTransformer',
        embed_dims=96,
        depths=depth,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=init_cfg),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(bbox_head=dict(num_classes=NUM_CLASS_SEG), mask_head=dict(num_classes=NUM_CLASS_SEG) )
) # model

print(f"chr_swin-t-p4-w7_fpn_1x_coco.py at:{TM}\n _base_={_base_}\n load_from={load_from}")