import os
print(f"where is clip_our.py? __file__={__file__};abs={os.path.abspath(__file__)}")

_base_ = [
    # '../_base_/models/resnet50.py',
    'mb321_base_cls.py'
    ] # _base_

from MB321.mm_flag import agNET
bb_name = agNET.bb_cls

orig_2262 = True if '2262' in bb_name else False
print(f"* orig_2262={orig_2262}")
# schedule
lr = 5e-4 * 1024 / 512  # CLS_LR == 0.001
warmup_iters, opt_type = 200, 'AdamW'  # 5
paramwise_cfg = dict(_delete=True, norm_decay_mult=0.0, bias_decay_mult=0.0)
if orig_2262:
    paramwise_cfg = dict(pad_small_map=False, norm_decay_mult=0.0, bias_decay_mult=0.0,
                         custom_keys={'.absolute_pos_embed': dict(decay_mult=0.0),
                                      '.relative_position_bias_table': dict(decay_mult=0.0)})  # paramwise_cfg
    if orig_2262:
        _s = f" expect original swin-t(2262OurSwin) warmup_iters({warmup_iters})=20"
        if pre_train and warmup_iters != 20:
            input(f"_s\n\n")
        else:
            # opt_type = 'SGD' # 2022-12-09
            print(_s)
optimizer = dict(type=opt_type, lr=lr, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
                 paramwise_cfg=paramwise_cfg)  # for batch in each gpu is 128, 8 gpu: lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer_config = dict(grad_clip=dict(max_norm=5.0))  # _delete_=True,
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr_ratio=1e-2, warmup='linear', warmup_ratio=1e-3,
                 warmup_iters=warmup_iters, warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=max_e)
runner_type = 'FlexibleRunner'
# optimizers

# runtime
checkpoint_config = dict(interval=gap_tra)
log_config = dict(interval=gap_tra)  # train

# dataset
evaluation = dict(interval=gap_val, metric=metrics, metric_options={
    'topk': (1, topk)})  # val, # "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for ' multi-label dataset
dataset_type, size_wh = 'Ichr24', (img_h, img_h)
# img_train, img_val, img_test = MM_DATA_MB321['ichr_train'], MM_DATA_MB321['ichr_val'], MM_DATA_MB321['ichr_test']
# data = dict(
#     samples_per_gpu=SAMPLES_PER_GPU_CLS,  # 128 in swin_svt_base
#     workers_per_gpu=NUM_WORKERS_CLS,
#     train=dict(data_prefix=img_train, size_wh=size_wh),
#     val=dict(data_prefix=img_val, size_wh=size_wh),
#     test=dict(data_prefix=img_test, size_wh=size_wh, _txt=LOG_THIS_ABN)
# )  # data

# --start model
type_bb = bb_name # bb_name[len('mmc_'):]
in_channels_head = 96*2**3
head = dict(type='LinearClsHead', num_classes=num_cls, in_channels=in_channels_head,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            cal_acc=True)  # topk=(1, TOP_K), # NOT set in head, but in 'chr7_cls.py --> args'; Top-k accuracy rate, top1 and top5
backbone = {'type': 'set later'}  # Mandatory !!
neck = None  # {'type':'GlobalAveragePooling'}

if 'rn50' in type_bb.lower():  # !! must before elif 'OurSwin'
    backbone.update({'type': 'LLM_resnet50'})
    # if orig_2262:
    #     backbone.update({'drop_path_rate': 0.2})  # default 0.1
    #     head.update(loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, num_classes=num_cls, mode='original'))  # head['loss'] = {}
    #     if orig_2262 and in_channels_head != 768: input(f"!! expect swin-t(2262OurSwin) in_channels_head({in_channels_head}) = 768\n\n")

    # if '2262' not in type_bb:
    #     head=dict(type='ClsHead', loss=dict(type='CrossEntropyLoss', loss_weight=1.0), topk=(1, TOP_K),)
elif 'OurVit' in type_bb:  # OurVit
    input(f"** uncomment following code ~")
#     img_size, patch_num_in_row = IMG_H, 3 # 96=3*32; 192=3*64; 256=8*32
#     backbone.update({'type':'OurVit', 'img_size':img_size, 'patch_size':int(img_size/patch_num_in_row), 'dim':DIM_FEAT,
#         'init_cfg':[{'type':'Kaiming', 'layer':'Conv2d', 'mode':'fan_in', 'nonlinearity':'linear'}] })
#     assert backbone['img_size'] == IMG_H, f"! backbone['img_size']={backbone['img_size']}, IMG_H={IMG_H}"
#     head=dict(type='ClsHead', loss=dict(type='CrossEntropyLoss', loss_weight=1.0))
#     # schedule
#     # paramwise_cfg = dict(custom_keys={'.cls_token': dict(decay_mult=0.0), '.pos_embed': dict(decay_mult=0.0)})
#     optimizer = dict(type='AdamW', lr=0.003, weight_decay=0.3) #, paramwise_cfg=paramwise_cfg,)
#     optimizer_config = dict(grad_clip=dict(max_norm=1.0))
#     lr_config = dict(policy='CosineAnnealing', min_lr=0, warmup='linear', warmup_iters=warmup_iters, warmup_ratio=1e-4,)
else:
    raise ValueError(f"??? chr_our.py: bb_name={bb_name}")
print(f"chr_our.py: type of backbone:{backbone}")

model = dict(type='ImageClassifier', backbone=backbone, neck=neck, head=head,
             # init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.), dict(type='Constant', layer='LayerNorm', val=1., bias=0.) ],
             # train_cfg=dict(augments=[dict(type='BatchMixup', alpha=0.8, num_classes=num_cls, prob=0.5), dict(type='BatchCutMix', alpha=1.0, num_classes=num_cls, prob=0.5)])
             )  # model
if orig_2262:
    model.update(dict(
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                  dict(type='Constant', layer='LayerNorm', val=1., bias=0.)],
        train_cfg=dict(augments=[dict(type='BatchMixup', alpha=0.8, num_classes=num_cls, prob=0.5),
                                 dict(type='BatchCutMix', alpha=1.0, num_classes=num_cls, prob=0.5)])
    ))  # model.update
print(f"* clip_our.py head={head}\n model={model}")
print(f"clip_our.py warmup={warmup_iters}, _base_={_base_}\n load_from={load_from}")
