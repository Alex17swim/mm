print(f" *here is mm_cfg: haixia.py")
_base_ = [
    # '../_base_/models/resnet50.py',
    '../_base_/datasets/_mb321_bs32.py', # _mb321_bs64, imagenet_bs32
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

# mm/mmclss/mmpretrain/evaluation/metrics/__init__.py
# mm/mmclss/mmpretrain/evaluation/metrics/single_label.py
type_bb, num_cls, topk, metrics = 'rn50', 24, 3, "Accuracy" # RetrievalRecall, "Accuracy" # , "precision", "recall", "f1-score", "support"]
val_interval, seed, name_cls = 10, 321, f"ichr_{'bil'}"
fp_pth = f"/public/home/alex/Docu/Dataset/output/pth_all/llm_{name_cls}/"

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
        num_classes=num_cls, in_channels=2048, loss=dict(type='PGMC_loss', loss_weight=1.0),
        # topk=(1, topk),
        cal_acc=False
    ),
) # end model
val_evaluator = dict(type=metrics, topk=(1,topk)) # # can also set 'topk' in mm_cls.py -> cfg_options={'val_evaluator.topk':(1,3)}
train_cfg = dict(max_epochs=100, val_interval=max(1, int(val_interval/2))) # using 'CutMix' batch augment: (type='CutMix', alpha=1.0) # by_epoch=True, # , augments=[dict(type='BatchMixup', alpha=0.8, num_classes=num_cls, prob=0.5)]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=val_interval, max_keep_ckpts=30, out_dir=fp_pth)
) # end default_hooks
randomness = dict(seed=seed, diff_rank_seed=True)
load_from = f"/public/home/alex/Docu/Dataset/trained/resnet50_8xb32_in1k_20210831-ea4938fc.pth" # None

print(f" *_mb321/haixia.py -> load_from={load_from}")
# default_hooks = dict(
#     logger=dict(interval=100), checkpoint=dict(interval=1),
#     param_scheduler=dict(step=[150, 200, 250]),)  # The learning rate adjustment has also changed
# init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.), dict(type='Constant', layer='LayerNorm', val=1., bias=0.) ],
# val_cfg = dict(),
# test_cfg = dict(),
# auto_scale_lr = dict(base_batch_size=256),

# runner = Runner(
#     model=MMResNet50(),
#     work_dir='./work_dir',
#     train_dataloader=train_dataloader,
#     optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
#     train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
#     val_dataloader=val_dataloader,
#     val_cfg=dict(),
#     val_evaluator=dict(type=Accuracy),
#     visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')]),
# )