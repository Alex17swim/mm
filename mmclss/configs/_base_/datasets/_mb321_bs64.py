print(f" *here is mm_cfg: _mb321_bs64.py") # from imagenet_bs64.py
# dataset settings
k, img_h = 1, 224
# _dr = f'A://Docu//Dataset//ichr_bil_data//0920kf//kf{k:02d}'
_dr = f'/public/home/alex/Docu/Dataset/ichr_bil_data/0920kf/kf{k:02d}/'
dataset_type, num_cls, bs, nw, topk = 'Ichr24', 24, 64, 2, 3
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
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=bs, # 64
    num_workers=nw, # 5
    dataset=dict(
        type=dataset_type,
        data_root=_dr, # 'data/imagenet',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=bs, # 64
    num_workers=nw, # 5
    dataset=dict(
        type=dataset_type,
        data_root=_dr, # 'data/imagenet'
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, topk))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
