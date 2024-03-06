print(f" *here is mm_cfg: _mb321_bs64_swin.py") # from imagenet_bs64_swin_224.py
# from MB321.mm_flag import agDIR, agDATA, agGPU
# data_root = '' # agDIR.root_cls
# dataset settings
# dataset_type, num_cls, batch_size, nw, topk = 'Ichr24', agDATA.num_cls, agDATA.batch_cls, agGPU.worker_cls, agDATA.topk
k, img_h = 1, 224
# _dr = f'A://Docu//Dataset//ichr_bil_data//0920kf//kf{k:02d}'
_dr = f'/public/home/alex/Docu/Dataset/ichr_bil_data/0920kf/kf{k:02d}/'
dataset_type, num_cls, bs, nw, topk = 'Ichr24', 24, 64, 2, 3
# if agDATA.img_h != 224: input(f"!! expect image size (224,224), found:{agDATA.img_h, agDATA.img_w}\n")

if img_h != 224: input(f"!! expect image size=(224,224) for _mb321_resize.py, found:{img_h}")
data_preprocessor = dict(
    num_classes=num_cls,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=img_h,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=img_h+32, # 256
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=img_h),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=bs, # 64
    num_workers=nw, # 5
    dataset=dict(
        type=dataset_type,
        data_root=_dr,
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=bs, # 64
    num_workers=nw, # 5
    dataset=dict(
        type=dataset_type,
        data_root=_dr,
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, topk)) # 5

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
