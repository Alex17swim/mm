print(f" *here is mm_cfg: _mb321_coco_instance.py") # from coco_instance.py
# dataset settings
dataset_type = 'MB321coco' # data_root = 'data/coco/'
# data_root = f'A://Docu//Dataset//Sk_Tag_data//sp_all//'
# outfile_prefix = f"A://Docu//Dataset//output/Sk_Tag/"
data_root = f'/public/home/alex/Docu/Dataset/Sk_Tag_data/sp_all/'
outfile_prefix = f"/public/home/alex/Docu/Dataset/output/Sk_Tag/"

batch_size, num_workers, scale_wh = 2, 2, (1333, 800)
pw, drop_last, format_only = True, False, False
metric = ['bbox', 'segm']

if format_only: input(f"!! format_only={format_only} will NOT show metrics of test() !!")

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

## move piple into mb321_base.py
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', scale=scale_wh, keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=scale_wh, keep_ratio=True),
#     # If you don't have a gt annotation, delete the pipeline
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
# ]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=pw,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_coco/annotations.json',
        data_prefix=dict(img='train_coco/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        # pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=pw,
    drop_last=drop_last,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val_coco/annotations.json',
        data_prefix=dict(img='val_coco/'),
        test_mode=True,
        # pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_coco/annotations.json',
    metric=metric,
    classwise=True,
    format_only=False,
    backend_args=backend_args)

# test_dataloader = val_dataloader
# test_evaluator = val_evaluator
# inference on test dataset and format the output results for submission.
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=pw,
    drop_last=drop_last,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test_coco/annotations.json',
        data_prefix=dict(img='test_coco/'),
        test_mode=True,
        # pipeline=test_pipeline
    ))
test_evaluator = dict(
    type='CocoMetric',
    metric=metric,
    classwise=True,
    format_only=format_only,
    ann_file=data_root + 'test_coco/annotations.json',
    outfile_prefix=outfile_prefix) # './work_dirs/coco_instance/test'
