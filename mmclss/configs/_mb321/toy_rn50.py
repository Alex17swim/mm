# work_dir = '{{fileBasenameNoExtension}}'
print(f" *here is mm_cfg: toy_rn50.py from renet/resnet50_8xb32_in1k.py")
_base_ = [
    # '../_base_/models/resnet50.py',
    'mb321_base_cls.py'
    # '../_base_/datasets/_mb321_bs32.py', # _mb321_bs64, imagenet_bs32
    # '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

from MB321.mm_flag import agDATA # , agNET, agMM, agDIR, agTAG
num_cls = agDATA.num_cls
# type_bb, num_cls, name_cls, topk = agNET.bb_cls, agDATA.num_cls, agDATA.name_cls, agDATA.topk
# val_interval, seed, metrics = agMM.gap_tra, agMM.seed, agMM.metrics
# max_epochs = agDATA.epoch_cls
# fp_pth = agDIR.pth_all_cls # f"/public/home/alex/Docu/Dataset/output/pth_all/llm_{name_cls}/"

model = dict(
    type='ImageClassifier',     # The type of the main model (here is for image classification task).
    backbone=dict(
        type='ResNet',          # The type of the backbone module.
        # All fields except `type` come from the __init__ method of class `ResNet`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.backbones.ResNet.html
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=-1,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),    # The type of the neck module.
    head=dict(type='LinearClsHead',     # The type of the classification head module.
        # All fields except `type` come from the __init__ method of class `LinearClsHead`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.heads.LinearClsHead.html
        num_classes=num_cls, in_channels=2048, loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        # topk=(1, topk),
        cal_acc=False
    ),
) # end model

# val_evaluator = dict(type=metrics, topk=(1,topk)) # # can also set 'topk' in mm_cls.py -> cfg_options={'val_evaluator.topk':(1,3)}
# train_cfg = dict(max_epochs=max_epochs, val_interval=max(1, int(val_interval/2))) # using 'CutMix' batch augment: (type='CutMix', alpha=1.0) # by_epoch=True, # , augments=[dict(type='BatchMixup', alpha=0.8, num_classes=num_cls, prob=0.5)]
# default_hooks = dict(
#     checkpoint=dict(type='CheckpointHook', interval=val_interval, max_keep_ckpts=30, out_dir=fp_pth)
# ) # end default_hooks
# randomness = dict(seed=seed, diff_rank_seed=True)
# load_from = agMM.pre_pth_seg # f"/public/home/alex/Docu/Dataset/trained/resnet50_8xb32_in1k_20210831-ea4938fc.pth" # None

print(f" *_mb321/toy_rn50.py")
