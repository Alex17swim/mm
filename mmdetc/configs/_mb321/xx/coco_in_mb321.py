# coco_in_mb321.py
print("where is coco_in_mb321.py? __file__={}".format(__file__))
# *from pt_flag import ** # will induce 'TypeError: cannot pickle 'module' object'
from pt_flag import BB_SEG, MM_PY_COCO, MM_DATA_MB321, SEG_NOT_TRAIN, MM_PTH_TRAINED, SEG_MAX_EPOCH

_base_ = ['../{}'.format(MM_PY_COCO[BB_SEG])]
# already added '../_base_/datasets/coco_instance.py' in _base_
# dataset_type = 'COCODataset' # already set in 'coco_instance.py'
# # data_root = DATA_ROOT_SEG # useless...
fp_img_train, fp_ann_train = MM_DATA_MB321['coco_train_img'], MM_DATA_MB321['coco_train_ann']
fp_img_val, fp_ann_val = MM_DATA_MB321['coco_val_img'], MM_DATA_MB321['coco_val_ann']
fp_img_test, fp_ann_test = MM_DATA_MB321['coco_test_img'], MM_DATA_MB321['coco_test_ann']

# Modify dataset related settings
if BB_SEG in ['mm_mrcnn_3x',]: # mstrain-poly_3x_coco_instance
    data = dict(
        train=dict(dataset=dict(img_prefix=fp_img_train,ann_file=fp_ann_train)),
        val=dict(img_prefix=fp_img_val,ann_file=fp_ann_val),
        test=dict(img_prefix=fp_img_test,ann_file=fp_ann_test))
else: # mm_mrcnn_1x:coco_instance; others: coco_detection
    data = dict(
        train=dict(img_prefix=fp_img_train,ann_file=fp_ann_train),
        val=dict(img_prefix=fp_img_val,ann_file=fp_ann_val),
        test=dict(img_prefix=fp_img_test,ann_file=fp_ann_test))
# else: raise ValueError("check BB_SEG:{}".format(BB_SEG))

# # We can use the pre-trained Mask RCNN model to obtain higher performance
if not SEG_NOT_TRAIN:
    runner = dict(type='EpochBasedRunner', max_epochs=SEG_MAX_EPOCH)
    load_from = MM_PTH_TRAINED
    print("coco_in_mb321.py, _base_={}\n load_from={}\n data['train']={}".format(_base_, load_from, data['train']))