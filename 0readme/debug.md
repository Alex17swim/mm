## mm_dir.py, Chengyu Wang, XJTLU, 2023-1107, last update on 2023-1107
import os.path as osp

BB_DICT = {
    # classification    
    **dict.fromkeys(['mmc_swin_s'], 'mmc_swin_s'),
    **dict.fromkeys(['mmc_swin_t', 'da_swin_t', 'da_2262OurSwin', 'mmc_2262OurSwin', ], 'mmc_swin_t'), # 'mmc_2262OurSwinShift'
    **dict.fromkeys(['mmc_RN50'],   'mmc_ResNet50'),
    **dict.fromkeys(['mmc_NEXT50'], ''),
    **dict.fromkeys(['mmc_SE50'],   'mmc_SeNet50'),
    **dict.fromkeys(['mmc_ToySwin', 'mmc_ToyVit', 'mmc_ToyCnn'], 'mmc_toy'),    
    **dict.fromkeys(['mmc_OurSwin', 'mmc_OurSwinShift', 'mmc_2262OurSwinShift', 'mmc_2262OurSwin', 'mmc_2282OurSwin'], 'mmc_OurSwin'), # , 'mmc_OurVit'
    **dict.fromkeys(['da_SeNet50', 'da_ResNet50', 'da_swin_t', 'da_2262OurSwin', 'da_OurSwin', 'da_OurSwinShift'], 'mmc_da'),
    **dict.fromkeys(['da_Toy', 'da_Lenet'], 'mmc_toy_da'),
    **dict.fromkeys(['mmc_vit'], 'mmc_vit'), # 'mmc_vit_b_p32', 'mmc_vit_b_p16'

    # segmentation / detection
    **dict.fromkeys(['mm_swin_s'],  'mm_swin_s_fp16'),
    **dict.fromkeys(['mm_mrcnn'],   'mm_mrcnn'),
    **dict.fromkeys(['mm_mrcnn'],   'mm_mrcnn'), 
} # end BB_DICT
# print(f"BB_DICT.keys=({BB_DICT.keys()})\n")

dict_skip = {**dict.fromkeys(['D2__PR_50_3x', 'etc.'], f"you will not see me"),}
MM_CFG_PY = {
    'mmc_OurSwin':  osp.join('_mb321', 'chr_our.py'),
    'mmc_da':   osp.join('_mb321', 'chr_da.py'),
    'mmc_toy':  osp.join('_mb321', 'chr_da.py'),
    'mmc_toy_da': osp.join('_mb321', 'chr_da_toy.py'),
    'mmc_vit': osp.join('_mb321', 'chr_vit.py'),
    
    'mmc_svt_s':    osp.join('_mb321', 'chr_swin_twins.py'),  # chr_twins_base.py
    'mmc_svt_b':    osp.join('_mb321', 'chr_swin_twins.py'),
    'mmc_pcpvt_s':  osp.join('_mb321', 'chr_swin_twins.py'),
    'mmc_pcpvt_b':  osp.join('_mb321', 'chr_swin_twins.py'),

    'mmc_swin_t': osp.join('_mb321', 'chr_swin_twins.py'),  # chr_swin_tiny_224.py
    'mmc_swin_s': osp.join('_mb321', 'chr_swin_twins.py'),
    'mmc_swin_b': osp.join('_mb321', 'chr_swin_twins.py'),
    'mmc_SeNet50':  osp.join('_mb321', 'chr_senet.py'),
    'mmc_ResNet50': osp.join('_mb321', 'clip_resnet50_cifar.py'),
    # 'mmc_ResNet50_cifar100':    osp.join('_mb321', 'clip_resnet50_cifar.py'),
    # 'mmc_ResNet50_cifar10':     osp.join('_mb321', 'clip_resnet50_cifar.py'),
    # 'mmc_ResNet50_in1k':        osp.join('_mb321', 'clip_resnet50_cifar.py'),
    'mmc_abn': "_mb321/abn_cls.py",
    'mm_mrcnn': "_mb321/abn_seg.py",  # !!!
    'mm_retina_50_2x': "_mb321/chr_retinanet_r50_fpn_1x_coco.py",
    'mm_retina_101_2x': "_mb321/chr_retinanet_r50_fpn_1x_coco.py",
    'mm_swin_t_base': "_mb321/chr_swin-t-p4-w7_fpn_1x_coco.py",
    'mm_swin_t_crop': "_mb321/chr_swin-t-p4-w7_fpn_1x_coco.py",
    'mm_swin_t_fp16': "_mb321/chr_swin-t-p4-w7_fpn_1x_coco.py",
    'mm_swin_s_fp16': "_mb321/chr_swin-t-p4-w7_fpn_1x_coco.py",
    # 'mm_faster_1x':     "_mb321/xx/chr_faster_rcnn_r50_fpn_1x_coco.py",
    'mm_faster_50_2x': "_mb321/chr_faster_rcnn_r50_fpn_1x_coco.py",
    'mm_faster_101_2x': "_mb321/chr_faster_rcnn_r50_fpn_1x_coco.py",
    'mm_mrcnn_50_2x': "_mb321/chr_mask_rcnn_r50_fpn_1x_coco.py",
    'mm_mrcnn_101_2x': "_mb321/chr_mask_rcnn_r50_fpn_1x_coco.py",
    # 'mm_mrcnn_1x':      "_mb321/xx/chr_mask_rcnn_r50_fpn_poly_1x_coco.py",
    # 'mm_mrcnn_3x':      "_mb321/xx/chr_mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py",
    'mm_detr': '_mb321/chr_detr.py',
    'mm_deform_base': "_mb321/chr_deformable_detr_r50_16x2_50e_coco.py",
    'mm_deform_iter': "_mb321/chr_deformable_detr_r50_16x2_50e_coco.py",
    'mm_deform_two': "_mb321/chr_deformable_detr_r50_16x2_50e_coco.py",
    'mm_spa_50_100': "_mb321/chr_sparse_rcnn_r50_fpn_1x_coco.py",
    'mm_spa_50_300': "_mb321/chr_sparse_rcnn_r50_fpn_1x_coco.py",
    'mm_spa_101_100': "_mb321/chr_sparse_rcnn_r50_fpn_1x_coco.py",
    'mm_spa_101_300': "_mb321/chr_sparse_rcnn_r50_fpn_1x_coco.py",
    'mm_pr_1x': "_mb321/chr_point_rend_r50_caffe_fpn_mstrain_1x_coco.py",
    'mm_pr_3x': "_mb321/chr_point_rend_r50_caffe_fpn_mstrain_1x_coco.py",
}
MM_CFG_PY.update(dict_skip)

MM_PRE_PTH = {
    **dict.fromkeys(['mmc_toy', 'mmc_OurSwin', 'mmc_da', 'mmc_da', 'mmc_toy_da', 'mmc_vit'],
                    f"! do NOT pre-trained from ImageNet for your defined models!"),

    'mmc_2282OurSwin': "swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth",
    'mmc_vit_b_p32': "vit-base-p32_3rdparty_pt-64xb64_in1k-224_20210928-eee25dd4.pth",
    'mmc_vit_b_p16': "vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth",

    'mmc_svt_s':    "twins-svt-small_3rdparty_8xb128_in1k_20220126-8fe5205b.pth",
    'mmc_svt_b':    "twins-svt-base_3rdparty_8xb128_in1k_20220126-e31cc8e9.pth",
    'mmc_pcpvt_s':  "twins-pcpvt-small_3rdparty_8xb128_in1k_20220126-ef23c132.pth",
    'mmc_pcpvt_b':  "twins-pcpvt-base_3rdparty_8xb128_in1k_20220126-f8c4b0d5.pth",

    # 'mmc_swin_t':   "swin-base_3rdparty_in21k.pth",
    'mmc_swin_t':   "swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth", # good!
    'mmc_swin_s':   "swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth",
    'mmc_swin_b':   "swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth",
    'mmc_SeNet50':  "se-resnet50_batch256_imagenet_20200804-ae206104.pth",
    'mmc_ResNet50': "resnet50_8xb32_in1k_20210831-ea4938fc.pth",  # "mmc_ResNet50_cifar10_20210528-f54bfad9.pth",
    'da_SeNet50':  "se-resnet50_batch256_imagenet_20200804-ae206104.pth",
    'da_ResNet50': "resnet50_8xb32_in1k_20210831-ea4938fc.pth",
    # 'mmc_ResNet50_cifar100':    "resnet50_b16x8_cifar100_20210528-67b58a1b.pth",
    # 'mmc_ResNet50_cifar10':     "resnet50_b16x8_cifar10_20210528-f54bfad9.pth",
    # 'mmc_ResNet50_in1k':        "resnet50_8xb32_in1k_20210831-ea4938fc.pth",
    'mm_mrcnn': "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth",
    'mm_retina_50_2x': "retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth",
    'mm_retina_101_2x': "retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth",
    'mm_faster_50_2x': "faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth",
    'mm_faster_101_2x': "faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth",
    'mm_mrcnn_50_2x': "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth",
    'mm_mrcnn_101_2x': "mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth",
    'mm_swin_t_base': "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth",
    'mm_swin_t_crop': "mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth",
    'mm_swin_t_fp16': "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth",
    'mm_swin_s_fp16': "mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth",
    'mm_pr_1x': "point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth",
    'mm_pr_3x': "point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth",
    'mm_spa_50_100': 'sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.pth',
    'mm_spa_50_300': 'sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth',
    'mm_spa_101_100': 'sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth',
    'mm_spa_101_300': 'sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth',
    'mm_detr': 'detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth',
    'mm_deform_base': 'deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth',
    'mm_deform_iter': 'deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth',
    'mm_deform_two': 'deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth',
    # 'mm_faster_1x': 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    # 'mm_mrcnn_3x':  'mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth',
    'alexnet': "alexnet-owt-4df8aa71.pth",
    'vggnet': "vgg11_bn-6002323d.pth",
    'resnet50': "resnet50-19c8e357.pth",
    'detr': "detr_demo-da2a99e9.pth",  # ???
    'detr_segm': "detr-r50-e632da11.pth",

    'deform_single': "r50_deformable_detr_single_scale-checkpoint.pth",
    'deform_dc5': "r50_deformable_detr_single_scale_dc5-checkpoint.pth",
    'deform_base': "r50_deformable_detr-checkpoint.pth",
    'deform_iter': "r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth",
    'deform_two': "r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth",
}
MM_PRE_PTH.update(dict_skip)

MM_COCO_PY = {  # only for COCO, re-comment it if you need
    # 'mm_faster_1x': "faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
    # 'mm_mrcnn_1x':  "mask_rcnn/mask_rcnn_r50_fpn_poly_1x_coco.py",
    # 'mm_mrcnn_3x':  "mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py",
    # 'mm_spa_50_100':   "sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco.py",
    # 'mm_spa_50_300':   "sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py",
    # 'mm_deform_base':   "deformable_detr/deformable_detr_r50_16x2_50e_coco.py",
    # 'mm_deform_iter':   "deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py",
    # 'mm_deform_two':    "deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py",
}

if __name__ == '__main__':
    print(f"-mm_dir.py")