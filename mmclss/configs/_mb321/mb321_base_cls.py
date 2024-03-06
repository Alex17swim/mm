print(f" *here is mm_cfg: mb321_base_cls.py")
_base_ = [
    # '../_base_/models/resnet50.py',
    '../_base_/datasets/_mb321_bs32.py', # _mb321_bs64, imagenet_bs32
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
import os.path as osp
from MB321.mm_flag import agNET, agMM, agDATA, agDIR # , agTAG

type_bb = agNET.bb_cls
num_cls, name_cls, img_h, topk, max_epochs = agDATA.num_cls, agDATA.name_cls, agDATA.img_h, agDATA.topk, agDATA.epoch_cls
val_interval, seed, metrics = agMM.gap_tra, agMM.seed, agMM.metrics
batch_size, nw = agDATA.batch_cls, agMM.worker_cls
data_root, fp_pth = agDIR.root_cls, agDIR.pth_all_cls # f"/public/home/alex/Docu/Dataset/output/pth_all/llm_{name_cls}/"
dataset_type = 'Ichr24mnist'
load_from = pretrained = agMM.pre_pth_cls # None
randomness = dict(seed=seed) # , diff_rank_seed=True
# input(f"data_root={data_root} \n train_root={osp.dirname(agDIR.tra_cls)} \n val_root={osp.dirname(agDIR.val_cls)}")

# train_pipeline = [dict(type='RandomResizedCrop', scale=img_h), dict(type='RandomFlip', prob=0.5, direction='horizontal'),]
# test_pipeline = [dict(type='ResizeEdge', scale=img_h+32, edge='short'), dict(type='CenterCrop', crop_size=img_h)]

train_dataloader = dict(batch_size=batch_size, num_workers=nw,
                        dataset=dict(type=dataset_type, data_root=osp.dirname(agDIR.tra_cls), split='train'),
                        sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(batch_size=batch_size, num_workers=nw,
                      dataset=dict(type=dataset_type, data_root=osp.dirname(agDIR.val_cls), split='val'),
                      sampler=dict(type='DefaultSampler', shuffle=False))
test_dataloader = val_dataloader
    # dict(batch_size=batch_size, num_workers=nw,
    #                   dataset=dict(type=dataset_type, data_root=osp.dirname(agDIR.test_cls), split='test', pipeline=test_pipeline),
    #                   sampler=dict(type='DefaultSampler', shuffle=False))

val_evaluator = dict(type=metrics, topk=(1,topk)) # can also set 'topk' in mm_cls.py -> cfg_options={'val_evaluator.topk':(1,3)}
test_evaluator = dict(type=metrics, topk=(1, topk))

# log_level=logging.ERROR
log_level = 'INFO' # choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
# log_file = agLOG.val_seg # set this in "mm/mmengi/mmengine/runner --> line 401:self.build_logger"
default_hooks = dict(
    timer=dict(type='CheckpointHook'), # CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=30, out_dir=fp_pth),
    visualization=dict(type='VisualizationHook') # , enable=False
) # default_hooks
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
train_cfg = dict(max_epochs=max_epochs, val_interval=max(1, int(val_interval/2))) # using 'CutMix' batch augment: (type='CutMix', alpha=1.0) # by_epoch=True, # , augments=[dict(type='BatchMixup', alpha=0.8, num_classes=num_cls, prob=0.5)]
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

print(f" mb321_base_cls.py -> load_from={load_from}")