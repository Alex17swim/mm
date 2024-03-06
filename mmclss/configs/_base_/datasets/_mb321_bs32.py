# from lazy_import import test_lazy
# test_lazy
# with read_base():
#     import lazy_import
#     np = lazy_import.lazy_module('numpy')
#     print(np.__version__)
print(f" *here is mm_cfg: _mb321_bs32.py") # from imagenet_bs32.py
k = 1
# _dr = f'A://Docu//Dataset//ichr_bil_data//0920kf//kf{k:02d}'
_dr = f'/public/home/alex/Docu/Dataset/ichr_bil_data/0920kf/kf{k:02d}/'

# dataset settings
dataset_type = 'Ichr24mnist'
num_cls, nw, img_h = 24, 2, 224
data_preprocessor = dict(
    num_classes=num_cls,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=img_h),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=img_h+32, edge='short'),
    dict(type='CenterCrop', crop_size=img_h),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=nw,
    dataset=dict(
        type=dataset_type,
        # data_prefix=_dr.rstrip('/')+'/train',
        data_root=_dr,
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=nw,
    dataset=dict(
        type=dataset_type,
        # data_prefix=_dr.rstrip('/')+'/val',
        data_root=_dr,
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# # If you want standard test, please manually configure the test dataset
# test_dataloader = val_dataloader
# test_evaluator = val_evaluator
test_dataloader = dict(
    batch_size=32,
    num_workers=nw,
    dataset=dict(
        type=dataset_type,
        # data_prefix=_dr.rstrip('/')+'/test',
        data_root=_dr,
        split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = dict(type='Accuracy', topk=(1, 5))


# from mmengine.config import read_base, lazy_module
# with read_base():
#     # from lazy_import import lazy_module
#     np=lazy_module("numpy")
#     input(f"v={np.__version__}")

# # from mmengine.config import read_base, lazy
# # with read_base():
# #     # osp = lazy.LazyObject('os', 'path') # lazy_obj
# #     from os import path as osp
# #     if not osp.exists(_dr): print(f"!! NOT exists:{_dr}\n\n")
# #     else: print(f" 1 found:{_dr}")
