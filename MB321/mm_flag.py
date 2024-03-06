# mm_flag.py, 2023-1017, Chengyu Wang, last update 2024-0104
''' *: os  $$: gpu  ##: dataset  &: image  ---: model  --: loss  -: directory  '''
# print(f"accuracy={acc:.3f}; epoch={e:03d}"); print('epoch [{:0>3}/{:0>3}]'.format(epoch, epoch_max))
import sys, os, argparse, platform, datetime, time # , copy  # , argparse
_import_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_import_root) # if 'win' in platform.uname()[0].lower():
import os.path as osp # os.path.basename(sys.argv[0]) # this .py filename
from mm_dir import MM_CFG_PY, MM_PRE_PTH
from mb321_flag import time_str, time_gap, write_txt, TIME_TH, TRUE, FALSE, RSTRIP, COPY, CP_FILE, MOVE, DEL, DEL_EMPTY, TM, TIME_TH_STR_mdHMS, IS_WIN
# bil, sz, com2, com2_bil,
_cn, _k, _n, USE_GPU_LIST, BB_S, BB_C = 1, 10, 'com2_cir', [1], 'mrcnn50', 'toy_rn50' # mrcnn50, swin_t_crop, toy_rn50, RN50, NEXT50, ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
name_seg, name_cls, KF_STR = f'Sk_Tag', f'ichr_{_n}', f'kf{_k:02d}' # mchr_{_n}, # ichr_bil # 10, 5, [10,1,2,3],[5,7,4,6],[,,8,9] # f'kf{4:02d}'
ts, tc, ds_fold_seg, ds_fold_cls = '', 'llm_', f'sp_all_cls{_cn}', osp.join('0920kf', KF_STR) # sp_all, val5_raw162_split_v5t10_cls1
RESUME_C, PRE_TRA_C, OP_BB_C, OP_CLIP, OP_FC, OP_ATTN,  = False, True, 'val', 'val', 'val', 'train' # train, val
RESUME_S, PRE_TRAIN_S, OP_BB_S = False, True, 'val'
QUICK, OPTIMIZER, ACT, DROP, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM = False, 'AdamW', 'relu', 0.0, 0.0001, 0.0001, 0.9 # SGD, AdamW # weight_decay is L2 normalization
ATTN, CLIP_DIM, SHOW_C, SHOW_S, SHOW_GT = 'cross', 'fc1024', False, False, False # 'bb' and 'b' #add, self, cross # fc1024, fc2048, cat3072, repeat
BASE_TASK, MM_SIZE_SEG_WH = ['cls'] + ['clip'], (2000, 1200) # ['seg', 'cls'] + ['clip'] # (2000, 1200), (2666, 1600), (1333, 800), (2592, 1944)
EPOCH_C, EPOCH_S = 12 if QUICK else 100, 2 if QUICK else 50 if PRE_TRAIN_S else 200
# tag, dir, log,
BB_TAG_C = f'{KF_STR}bb={BB_C}'
BB_TAG_S = f'bb={BB_S}' # {KF_STR}
SHOW_MANY, SEG_IMG_SHOW = False, False
# check and auto define
_str_drop = str(DROP).replace('.', 'd')
if ATTN.lower() in ['cross']: CLIP_DIM += f'both' # _{_str_drop} # both
if (ATTN.lower() not in ['add']) and (CLIP_DIM == 'cat3072'): input(f"! only attn='add' support CLIP_DIM={CLIP_DIM}, set to default: {'fc2048'}"); CLIP_DIM = 'fc2048'
B_CHR_C = TRUE if 'chr' in name_cls.lower() else FALSE
B_CHR_S, B_SK_TAG = TRUE if 'chr' in name_seg.lower() else FALSE, TRUE if 'sk' in name_seg.lower() else FALSE
NUM_CLASS_C = 24 if B_CHR_C else 10
NUM_CLASS_S = 1 if B_CHR_S else 2 if B_SK_TAG else 80  # if  else 91 # 91 for stuff segment mask, 81 for thing detection bbox
IMG_H, IMG_W, IMG_C = 224, 224, 3
BUG_TAG_C = f"_{BB_TAG_C}_{ATTN}_{CLIP_DIM}_{OPTIMIZER}_{ACT}_e{EPOCH_C}_dp{DROP}_wd{WEIGHT_DECAY}" # all tags for classification
_tmp_tag_cls = f'_toy' # _{OPTIMIZER}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY} # _e{EPOCH_C}_{ATTN}_{CLIP_DIM}_{ACT}_dp{DROP}
BUG_TAG_S = f"_{BB_TAG_S}" # all tags for segmentation
_w = f'_w{MM_SIZE_SEG_WH[0]}' if MM_SIZE_SEG_WH[0] != 1333 else ''
_tmp_tag_seg = f'_p{int(PRE_TRAIN_S)}_c{_cn}{_w}' #

_output = 'output2' if QUICK else 'output'
# def write_txt(fnp_list, _str, b_prt=True, _mode='a', _write=True):
#     _list_none = [None, '', [None], [''], []]
#     if b_prt: print(_str)
#     if not _write: return
#     if fnp_list in _list_none or len(fnp_list) < 1: return
#     if isinstance(fnp_list, str): fnp_list = [fnp_list]
#     if not type(fnp_list) is list: raise ValueError("??fnp_list={}".format(fnp_list))
#     fnp_list = list(set(fnp_list))  # remove repeated item
#     fnp_list = [e for e in fnp_list if e not in _list_none]
#     assert fnp_list is not None and len(fnp_list) >= 1, "!check fnp_list:{}".format(fnp_list)
#     for fnp in fnp_list:
#         os.makedirs(os.path.dirname(fnp), exist_ok=True)
#         with open(fnp, _mode) as fw:
#             if isinstance(_str, str): fw.writelines(_str + '\n')
#             else: fw.write(str(_str))
parser_base = argparse.ArgumentParser() # formatter_class=argparse.ArgumentDefaultsHelpFormatter
parser_base.add_argument('--uname', default=platform.uname(), help="UNAME")
parser_base.add_argument('--root_data', default="/public/home/alex/Docu/Dataset", help="directory root for input and output") # "/media/alex/AI" if '911'
parser_base.add_argument('--root_code', default=osp.dirname(os.path.dirname(__file__)), help='')
parser_base.add_argument('--tm', default=TM, help=".strftime()")
parser_base.add_argument('--time_date', default=TIME_TH, help="datetime.datetime.now()")
parser_base.add_argument('--task', default=BASE_TASK, help="cls, seg")
parser_base.add_argument('--quick', default=QUICK, help="quick debug")
parser_base.add_argument('--seed', default=321, help="quick debug")
agBASE = parser_base.parse_args()
LOG_BUG_ALL = osp.join(agBASE.root_data, _output, 'debug_all', f"debug_all_{TM}.md")
write_txt(LOG_BUG_ALL, f" task({'_'.join(agBASE.task)}) type={type(agBASE.task)}, quick={QUICK} \n root_data={agBASE.root_data} \n root_code={agBASE.root_code}", b_prt=True)

_gpu_str = ','.join(map(str, USE_GPU_LIST))
parser_gpu = argparse.ArgumentParser()
parser_gpu.add_argument('--gpu_str', default=_gpu_str, help="gpu string")
parser_gpu.add_argument('--dev', default=f"cuda:{_gpu_str}", help="gpu int list")
parser_gpu.add_argument('--gpu_list', default=USE_GPU_LIST, help="gpu int list")
parser_gpu.add_argument('--pin', default=False, help="PIN_MEMORY") # False-faster;
parser_gpu.add_argument('--worker_cls', default=2, help="number workers of classification")
parser_gpu.add_argument('--worker_seg', default=2, help="number workers of segmentation")
# parser_gpu.add_argument('--gpu_all', default=torch.cuda.device_count(), help="total gpus")
agGPU = parser_gpu.parse_args()
if len(agGPU.gpu_list) == 1:
    parser_gpu.add_argument('--gpu_id', default=int(agGPU.gpu_str), help="gpu id")
    agGPU = parser_gpu.parse_args() # update 2

parser_net = argparse.ArgumentParser()
parser_net.add_argument('--bb_seg', default=BB_S, help="segmentation backbone in ['RN50', ]")
parser_net.add_argument('--bb_pretrain_seg', default=PRE_TRAIN_S, help="segmentation bb: pre train, '' for NOT pre train")
parser_net.add_argument('--bb_resume_seg', default=RESUME_S, help="segmentation bb: resume training, '' for NOT resume .pth")
parser_net.add_argument('--bb_train_seg',   default=True if 'train' in OP_BB_S.lower() else False, help="bb need training")
parser_net.add_argument('--bb_cls', default=BB_C, help="classification backbone in ['RN50', 'NEXT50']")
parser_net.add_argument('--bb_pretrain_cls', default=PRE_TRA_C, help="classification bb: pre train, '' for NOT pre train")
parser_net.add_argument('--bb_resume_cls', default=RESUME_C, help="classification bb: resume training, '' for NOT resume .pth")
parser_net.add_argument('--bb_train_cls',   default=True if 'train' in OP_BB_C.lower() else False, help="bb need training")
parser_net.add_argument('--bb_clip', default='RN50', help="backbone of clip in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']")
parser_net.add_argument('--clip_train', default=True if 'train' in OP_CLIP.lower() else False, help="clip of train, val, both")
parser_net.add_argument('--fc_train',   default=True if 'train' in OP_FC.lower() else False, help="linear regression need training")
parser_net.add_argument('--attn', default=ATTN.lower(), help="attention type")
parser_net.add_argument('--attn_train', default=True if 'train' in OP_ATTN.lower() else False, help="attention need training")
parser_net.add_argument('--clip_dim', default=CLIP_DIM.lower(), help="clip dimension for attention")
parser_net.add_argument('--opti', default=OPTIMIZER, help="optimizer") # SGD, AdamW
parser_net.add_argument('--lr', default=LEARNING_RATE, help="start learning rate")
parser_net.add_argument('--wd', default=WEIGHT_DECAY, help="weight_decay")
parser_net.add_argument('--mome', default=MOMENTUM, help="SGD momentum")
agNET = parser_net.parse_args()

ds_fold_seg = 'quick' if QUICK else ds_fold_seg
ds_fold_cls = '0920quick' if QUICK else ds_fold_cls
parser_data = argparse.ArgumentParser()
parser_data.add_argument('--kf', default=KF_STR, help="k-fold of 'kf01' to 'kf10'")
parser_data.add_argument('--name_cls', default=name_cls, help="classification dataset root dir")
parser_data.add_argument('--name_seg', default=name_seg, help="segmentation dataset root dir")
parser_data.add_argument('--ds_fold_cls', default=ds_fold_cls, help="tag of the classification dataset")
parser_data.add_argument('--ds_fold_seg', default=ds_fold_seg, help="tag of the segmentation dataset")
parser_data.add_argument('--num_cls', default=NUM_CLASS_C, help="class number of classification")
parser_data.add_argument('--num_seg', default=NUM_CLASS_S, help="class number of segmentation")
parser_data.add_argument('--augment_cls', default="flip,rotate", help="data augmentation for classification")
parser_data.add_argument('--augment_seg', default="flip,rotate", help="data augmentation for segmentation")
parser_data.add_argument('--img_h', default=IMG_H, help="image height")
parser_data.add_argument('--img_w', default=IMG_W, help="image width")
parser_data.add_argument('--img_c', default=IMG_C, help="image channel")
parser_data.add_argument('--size_wh', default=(IMG_W, IMG_H), help="image channel")
parser_data.add_argument('--mean', default=[0.485, 0.456, 0.406], help="mean of normalization")
parser_data.add_argument('--std', default=[0.229, 0.224, 0.225], help="std of normalization")
parser_data.add_argument('--ext', default='.jpg', help="extension name of image")
parser_data.add_argument('--LB_SUB1', default=int(-1), help="label starts from 0; name from 1")
parser_data.add_argument('--R1_LB', default=r'(__[0-9]+)', help="__number")
parser_data.add_argument('--R2_LB', default=r'([0-9]+)', help="number")
parser_data.add_argument('--R3_LB_D2', default=r'(lb__[0-9]+)', help="lb__number")
# parser_net.add_argument('--iter', default=MAX_ITER, help="training iteration,")
parser_data.add_argument('--epoch_cls', default=EPOCH_C, help="training epoch")
parser_data.add_argument('--epoch_seg', default=EPOCH_S, help="training epoch")
parser_data.add_argument('--batch_cls', default=64, help="batch size for classification") # "Our suggested max number of worker in current system is 16"
parser_data.add_argument('--batch_seg', default=2, help="batch size for segmentation") # 256
parser_data.add_argument('--img_per_batch', default=2, help="image per batch") # 64
parser_data.add_argument('--samples_per_gpu_cls', default=16, help="for classification")
parser_data.add_argument('--samples_per_gpu_seg', default=2, help="for segmentation")
parser_data.add_argument('--samples_per_gpu_da', default=8, help="for domain adaptation")
parser_data.add_argument('--topk', default=2, help="top-k of metric")
agDATA = parser_data.parse_args()
parser_data.add_argument('--img_mode', default='L' if agDATA.img_c == 1 else 'RGB', help="image mode")
agDATA = parser_data.parse_args() # agDATA update 1

_kc, _ks = f'mmc_{BB_C}', f'mm_{BB_S}'
write_txt(LOG_BUG_ALL, f" _kc={_kc}, _ks={_ks}", b_prt=SHOW_MANY)
if B_SK_TAG: MM_CLASSES_SEG = ('T', ) if _cn == 1 else ('T', 'F', )
elif B_CHR_S: MM_CLASSES_SEG = ('chr',)
else: MM_CLASSES_SEG = None
MM_CLASSES_CLS = tuple(["cls{:02d}".format(i) for i in range(1, 25)] if True else tuple(['cls21', 'cls_other']))
MM_GAP_TRA, MM_GAP_VAL, MM_DS, MM_NAME, MM_EXT = 10, 5, 'tap', 'SkTapDataset', '.jpg' #
MM_CFG_CLS = osp.join(agBASE.root_code, 'mmclss', 'configs', MM_CFG_PY[_kc])
MM_CFG_SEG = osp.join(agBASE.root_code, 'mmdetc', 'configs', MM_CFG_PY[_ks])
MM_PRE_PTH_CLS = osp.join(agBASE.root_data, 'trained', MM_PRE_PTH[_kc])
MM_PRE_PTH_SEG = osp.join(agBASE.root_data, 'trained', MM_PRE_PTH[_ks])
parser_mm = argparse.ArgumentParser()
parser_mm.add_argument('--gap_tra', default=MM_GAP_TRA, help="gap of training info")
parser_mm.add_argument('--gap_val', default=MM_GAP_VAL, help="gap of validating info")
parser_mm.add_argument('--ds', default=MM_DS, help="chr, tap")
parser_mm.add_argument('--name', default=MM_NAME, help="SkTapDataset, MchrDataset, CocoDataset")
parser_mm.add_argument('--size_seg_wh', default=MM_SIZE_SEG_WH, help="(1333, 800), (2666, 1600), (2592, 1944)")
parser_mm.add_argument('--ext', default=MM_EXT, help="extension name of image")
parser_mm.add_argument('--cfg_cls', default=MM_CFG_CLS, help="classification config file for mm")
parser_mm.add_argument('--cfg_seg', default=MM_CFG_SEG, help="segmentation/detection config file for mm")
parser_mm.add_argument('--pre_pth_cls', default=MM_PRE_PTH_CLS if agNET.bb_pretrain_cls else None, help="classification config file for mm")
parser_mm.add_argument('--pre_pth_seg', default=MM_PRE_PTH_SEG if agNET.bb_pretrain_seg else None, help="segmentation/detection config file for mm")
parser_mm.add_argument('--classes_seg', default=MM_CLASSES_SEG, help="('chr',), ('T', 'F', )")
parser_mm.add_argument('--classes_cls', default=MM_CLASSES_CLS, help="ICHR_CLASSES, ichr classes")
parser_mm.add_argument('--show_seg', default=SHOW_S, help="whether to display the prediction results in a window.")
parser_mm.add_argument('--show_cls', default=SHOW_C, help="whether to display the prediction results in a window.")
parser_mm.add_argument('--auto_lr', default=True, help="auto-scale-lr")
parser_mm.add_argument('--amp', default=True, help="automatic-mixed-precision training")
parser_mm.add_argument('--no_pw', default=True, help="no_persistent_workers")
parser_mm.add_argument('--tta', default=False, help="to enable the Test-Time-Aug")
parser_mm.add_argument('--seed', default=agBASE.seed, help="random seed")
parser_mm.add_argument('--metrics', default="Accuracy", help="Accuracy, f1-score, RetrievalRecall, Accuracy, precision, recall, support")
parser_mm.add_argument('--worker_seg', default=agGPU.worker_seg, help="")
parser_mm.add_argument('--worker_cls', default=agGPU.worker_cls, help="")
agMM = parser_mm.parse_args()
write_txt(LOG_BUG_ALL, f" cfg_cls={agMM.cfg_cls}\n cfg_seg={agMM.cfg_seg}", b_prt=True)

parser_tag = argparse.ArgumentParser()
parser_tag.add_argument('--bb_cls',  default=BB_TAG_C, help="tag for backbone")
parser_tag.add_argument('--bb_seg',  default=BB_TAG_S, help="tag for backbone")
parser_tag.add_argument('--bug_cls', default=_tmp_tag_cls, help="tag for classification debug")
parser_tag.add_argument('--bug_seg', default=_tmp_tag_seg, help="tag for segmentation debug")
parser_tag.add_argument('--find_cls', default=f"{BB_TAG_C}{_tmp_tag_cls}", help="tag for find .pth of classification")
parser_tag.add_argument('--log_cls', default=f"{BB_TAG_C}{_tmp_tag_cls}_{agBASE.tm}", help="tag for classification log") # {agDATA.name_cls}_
parser_tag.add_argument('--find_seg', default=['.pth', f'{BB_TAG_S}'], help="tag for find .pth of segmentation")
parser_tag.add_argument('--log_seg', default=f"{BB_TAG_S}{_tmp_tag_seg}_{agBASE.tm}", help="tag for segmentation log") # {agDATA.name_seg}_
agTAG = parser_tag.parse_args()

_this_cls = f"{agTAG.log_cls}_gpu{agGPU.gpu_str}"
_this_seg = f"{agTAG.log_seg}_gpu{agGPU.gpu_str}"
if 'com2' in name_cls:
    tra_cls = name_cls[:name_cls.index('com2')+len('com2')]
    val_cls = name_cls.replace('_com2', '')
    _this_cls = f"{val_cls[val_cls.rindex('_') + 1:]}_{_this_cls}"
    write_txt([LOG_BUG_ALL], f" !com2 data tra_cls={tra_cls}, val_cls={val_cls}\n")
    _tra = tra_cls if agNET.bb_train_cls else val_cls
else: tra_cls, val_cls, _tra = name_cls, name_cls, name_cls


parser_dir = argparse.ArgumentParser()
parser_dir.add_argument('--temp', default=osp.join(agBASE.root_data, 'Temp_data'), help="temp root")
parser_dir.add_argument('--root_in', default=agBASE.root_data, help="input root, need to specify cls, seg, etc.")
# parser_dir.add_argument('--cls_in', default=osp.join(agBASE.root_data, f"{name_cls}_data"), help="classification data input root")
# parser_dir.add_argument('--seg_in', default=osp.join(agBASE.root_data, f"{name_seg}_data"), help="segmentation data input root")
parser_dir.add_argument('--root_cls', default=osp.join(agBASE.root_data, f"{name_cls}_data", agDATA.ds_fold_cls), help="classification train set root dir")
parser_dir.add_argument('--tra_cls', default=osp.join(agBASE.root_data, f"{_tra}_data", agDATA.ds_fold_cls, 'train'), help="classification train set root dir")
parser_dir.add_argument('--val_cls', default=osp.join(agBASE.root_data, f"{val_cls}_data", agDATA.ds_fold_cls, 'val'), help="classification validation/test set root dir")
parser_dir.add_argument('--test_cls', default=osp.join(agBASE.root_data, f"{val_cls}_data", agDATA.ds_fold_cls, 'test'), help="classification test set root dir")
parser_dir.add_argument('--root_seg', default=osp.join(agBASE.root_data, f"{name_seg}_data", agDATA.ds_fold_seg), help="segmentation train set root dir")
parser_dir.add_argument('--tra_seg', default=osp.join(agBASE.root_data, f"{name_seg}_data", agDATA.ds_fold_seg, 'train_coco'), help="segmentation train set root dir")
parser_dir.add_argument('--val_seg', default=osp.join(agBASE.root_data, f"{name_seg}_data", agDATA.ds_fold_seg, 'val_coco'), help="segmentation validation/test set root dir")
parser_dir.add_argument('--seg_test', default=osp.join(agBASE.root_data, f"{name_seg}_data", agDATA.ds_fold_seg, 'test_coco'), help="segmentation test set root dir")
parser_dir.add_argument('--root_out', default=osp.join(agBASE.root_data, _output), help="output root")
parser_dir.add_argument('--out_all_cls', default=osp.join(agBASE.root_data, _output, f"{tc}{tra_cls}"), help="classification data output root")
parser_dir.add_argument('--out_all_seg', default=osp.join(agBASE.root_data, _output, f"{ts}{name_seg}"), help="segmentation data output root")
parser_dir.add_argument('--pth_all_cls', default=osp.join(agBASE.root_data, _output, 'pth_all', f"{tc}{tra_cls}"), help="classification pth save root")
parser_dir.add_argument('--pth_all_seg', default=osp.join(agBASE.root_data, _output, 'pth_all', f"{ts}{name_seg}"), help="segmentation pth save root")
# parser_dir.add_argument('--fp_tb_all', default=osp.join(agBASE.root_data, _output, 'tb_all'), help="tb output root") # better to move into each output dir
agDIR= parser_dir.parse_args()
parser_dir.add_argument('--last_cls', default=osp.join(agDIR.out_all_cls, "last"), help="save last/final .pth files for classification, bettern to besides each experiments")
parser_dir.add_argument('--out_cls',    default=osp.join(agDIR.out_all_cls, f"{_this_cls}"), help="output this for classification")
parser_dir.add_argument('--out_tb_cls', default=osp.join(agDIR.out_all_cls, f"{_this_cls}", 'tb'), help="tb output for tensorboard classification")
parser_dir.add_argument('--out_pth_cls', default=osp.join(agDIR.pth_all_cls, f"{_this_cls}"), help="pth save for classification, will be deleted for saving spaces")
parser_dir.add_argument('--last_seg', default=osp.join(agDIR.out_all_seg, "last"), help="save last/final .pth files for segmentation, bettern to besides each experiments")
parser_dir.add_argument('--out_seg', default=osp.join(agDIR.out_all_seg, f"{_this_seg}"), help="output this for segmentation")
parser_dir.add_argument('--out_tb_seg', default=osp.join(agDIR.out_all_seg, f"{_this_seg}", 'tb'), help="tb output for tensorboard segmentation")
parser_dir.add_argument('--out_pth_seg', default=osp.join(agDIR.pth_all_seg, f"{_this_seg}"), help="pth save for segmentation, will be deleted for saving spaces")
agDIR = parser_dir.parse_args() # update 1
# input(f"train_cls({osp.exists(agDIR.tra_cls)})={agDIR.tra_cls}\n val_cls({osp.exists(agDIR.val_cls)})={agDIR.val_cls}\n test_cls({osp.exists(agDIR.test_cls)})={agDIR.test_cls}\n\n")

_ac = '_'.join(agBASE.task) if 'cls' in agBASE.task else 'cls'
_as = '_'.join(agBASE.task) if 'seg' in agBASE.task else 'seg_'
write_txt(LOG_BUG_ALL, f" _ac, _as before: {_ac}, {_as}", b_prt=SHOW_MANY)
_ac, _as = _ac.replace('seg_', ''), _as.replace('cls_', '')
write_txt(LOG_BUG_ALL, f" _ac, _as after: {_ac}, {_as}", b_prt=SHOW_MANY)
parser_log = argparse.ArgumentParser()
parser_log.add_argument('--log_all_cls', default=osp.join(agDIR.root_out, f"{_ac}_{name_cls}_all.md"), help="classification log all at output root")
parser_log.add_argument('--log_all_seg', default=osp.join(agDIR.root_out, f"{_as}_{name_seg}_all.md"), help="segmentation log all at output root")
parser_log.add_argument('--bug_all', default=LOG_BUG_ALL, help="")
parser_log.add_argument('--tra_cls', default=osp.join(agDIR.out_cls, f"{agTAG.log_cls}_train.md"), help="log this for classification training")
parser_log.add_argument('--val_cls', default=osp.join(agDIR.out_cls, f"{agTAG.log_cls}_val.md"), help="log this for classification validation/testing")
parser_log.add_argument('--bug_cls', default=osp.join(agDIR.out_cls, f"{agTAG.log_cls}_bug.md"), help="log this for classification debugging")
parser_log.add_argument('--tra_seg', default=osp.join(agDIR.out_seg, f"{agTAG.log_seg}_train.md"), help="log this for segmentation training")
parser_log.add_argument('--val_seg', default=osp.join(agDIR.out_seg, f"{agTAG.log_seg}_val.md"), help="log this for segmentation validation/testing")
parser_log.add_argument('--bug_seg', default=osp.join(agDIR.out_seg, f"{agTAG.log_seg}_bug.md"), help="log this for segmentation debugging")
parser_log.add_argument('--task', default=agBASE.task, help="seg, cls, clip, etc.")
agLOG = parser_log.parse_args()
parser_log.add_argument('--cls', default=[agLOG.tra_cls, agLOG.val_cls] if agNET.bb_train_cls else [agLOG.val_cls], help="")
parser_log.add_argument('--seg', default=[agLOG.tra_seg, agLOG.val_seg] if agNET.bb_train_seg else [agLOG.val_seg], help="")
agLOG = parser_log.parse_args() # update 1

parser_t21 = argparse.ArgumentParser()
parser_t21.add_argument('--fac', default=0.3, help="FAC_DS, factor of T21 detectoin")
parser_t21.add_argument('--plt_karyo', default=True, help="PLT_KARYO, plot the predicted karyotype image")
parser_t21.add_argument('--abn_ratio', default=0.0, help="ABN_RATIO, ratio of abnormal T21 images")
parser_t21.add_argument('--make1', default=False, help="MAKE1, make 1 image and class==1")
parser_t21.add_argument('--lb_t21', default=False, help="LB_T21, is T21 only label")
parser_t21.add_argument('--abn_data_type', default='', help="ABN_DATA_TYPE, 'hist' for 'pick9', 'make1'")
# parser_t21.add_argument('--', default=False, help="")
# parser_t21.add_argument('--', default=False, help="")
agT21 = parser_t21.parse_args()

parser_sk = argparse.ArgumentParser()
parser_sk.add_argument('--clsT', default=0, help="1st class is True")
parser_sk.add_argument('--clsF', default=1, help="2nd class is False")
parser_sk.add_argument('--sk', default=B_SK_TAG, help="")
parser_sk.add_argument('--skip_f', default=False if B_SK_TAG else False, help="skip plot but still count instances number")
parser_sk.add_argument('--fac', default=1 if B_SK_TAG else 1, help="")
parser_sk.add_argument('--fs', default=20, help="font size, default 13")
parser_sk.add_argument('--show_gt', default=SHOW_GT, help="show ground turth")
agSK = parser_sk.parse_args()

parser_llm = argparse.ArgumentParser()
# for key, value in vars(agBASE).items():
#     # parser_llm.add_argument(key, default=value)
#     setattr(parser_llm, key, value)
parser_llm.add_argument('--base', default=agBASE, help="base")
parser_llm.add_argument('--gpu', default=agGPU, help="gpu")
parser_llm.add_argument('--net', default=agNET, help="network, forward, backward")
parser_llm.add_argument('--data', default=agDATA, help="data")
parser_llm.add_argument('--tag', default=agTAG, help="tag")
parser_llm.add_argument('--dir', default=agDIR, help="directory of input and output")
parser_llm.add_argument('--log', default=agLOG, help="logs")
parser_llm.add_argument('--coco', default=agMM, help="COCO")
parser_llm.add_argument('--t21', default=agT21, help="chromosome T21")
parser_llm.add_argument('--sk', default=agSK, help="Sikai, TAP, AGP")
argsLLM = parser_llm.parse_args()

parser0 = argparse.ArgumentParser() # formatter_class=argparse.ArgumentDefaultsHelpFormatter
parser0.add_argument('--xx1', '-x1', default="", help="")
parser0.add_argument('--x2', default="", help="")
args0 = parser0.parse_args()
# # double check
TXT_FLAG_ALL = f"GLOBAL_TAG={agBASE.task}\n"
if 'cls' in agBASE.task: TXT_FLAG_ALL += f" name_cls={name_cls}, BUG_TAG_C={BUG_TAG_C}\n"
if 'seg' in agBASE.task: TXT_FLAG_ALL += f" name_seg={name_seg}, BUG_TAG_S={BUG_TAG_S}"
if 'cls' in agBASE.task and 'seg' in agBASE.task:
    _seg, _cls, _ig, _ic = name_seg.lower(), name_cls.lower(), name_seg.index('_') + 1, name_cls.index('_') + 1
    write_txt([LOG_BUG_ALL], f"& data name: {_seg[_ig:]}, {_cls[_ic:]}", b_prt=SHOW_MANY)
    if (_seg[_ig:] != _cls[_ic:]) and (('chr' in _seg and 'chr' in _cls) or ('sk' in _seg and 'sk' in _cls)):
        input(f" name clas ({name_cls}) != seg ({name_seg})\n")
_ds_all = ['chr', 'sk']
if not any (item in name_cls.lower() for item in _ds_all):
    # if n not in name_cls.lower():
    input(f"! {name_cls} NOT a chromosome or Sk dataset, will set class number to 10 \n")
    agDATA.num_cls = 10
if not any (item in name_seg.lower() for item in _ds_all):
    input(f"! {name_seg} NOT a chromosome or Sk dataset, will set segmentation class number to 80\n")
    agDATA.num_seg = 80

# def time_str(t=''): return t.strftime("%m%d_%H%M%S") if isinstance(t, datetime.datetime) else datetime.datetime.now().strftime("%m%d_%H%M%S")  # time.strftime("%H%M")
# def time_gap(_t=TIME_TH):
#     if type(_t) is datetime.datetime:
#         _gap = datetime.datetime.now() - _t
#         _gap = _gap.seconds  # /10000
#     elif type(_t) is datetime.timedelta:
#         _gap = _t.seconds
#     elif type(_t) is float:
#         if _t > 60 * 60 * 24 * 30:  # 60s*60m*24h*30day, so _t is: time
#             _gap = time.time() - _t
#         else: _gap = _t  # _t is: time-time
#     else:
#         input(f"!! _t={_t} ,type={type(_t)}\n")
#         return
#     if _gap > 3600.0:
#         _gap /= 3600.0
#         str_d = ": %.2f hours" % _gap
#     elif _gap > 60.0:
#         _gap /= 60.0
#         str_d = ": %.2f minutes" % _gap
#     else: str_d = ": %.2f seconds" % _gap
#     # if _str:
#     #     str_d = str_d[str_d.index(' ')+1:]
#     #     str_d = str_d.replace(' ', '_')
#     return str_d
# # end time_gap
def if_print(str, b_prt=True):
    if b_prt: print(str); return True
    else: return False
def input3(_notice, _err='', _c=None, _s=None):
    if _c is None: _c = ['c', 'C', ''] # continue
    if _s is None: _s = ['s', 'S'] # 'skip'
    r = input(f"{_notice}, press ENTER or {set(_c) - {''}} to continue; {_s} to skip, others quit\n")
    if r not in _c + _s: raise ValueError(f"{r}:{'886'}, {_err}")
    r = 'c' if r == '' else r.lower()
    return r
def get_fns(fp, _check0=True, _only_dir=False, _only_file=False, _only_img=False, _ext='.jpg', _prt=True):
    assert osp.exists(fp), "! not found: {}".format(fp)
    assert not (_only_dir and _only_file), "! set only one of them to be true"
    _fns = os.listdir(fp)
    _dirs = list(filter(lambda x: osp.isdir(osp.join(fp, x)), _fns))
    _files = list(filter(lambda x: osp.isfile(osp.join(fp, x)), _fns))
    _ld, _lf = len(_dirs), len(_files)
    _ldf = len(_fns)
    if _ldf != (_ld + _lf):
        _unknown = set(_fns) - set(_dirs) - set(_files)
        input3("! skip unknown '{}', is a shortcut?\n in: {}".format(_unknown, fp))
        _fns = _dirs + _files
    if _ld + _lf < 1:
        if _prt: print("an empty folder: {}".format(fp))
        if _check0 and _ld + _lf < 1:
            _s = "! NO file/folder found in: {}, input 'del' to delete this empty folder and continue,\n" \
                 " input 'c' or 'C' to continue, others to quit\n".format(fp)
            r = input(_s)
            if 'del' == r: os.system("{} {}".format(DEL_EMPTY, fp))
            elif r not in ['c', 'C']: assert False
    else:
        _dirs.sort()
        _files.sort()
        if _only_dir:
            assert _ld > 0 and _lf < 1, "! expect 'only_dir', but found {} dirs and {} files in {}".format(_ld, _lf, fp)
        elif _only_file or _only_img:
            assert _ld < 1 and _lf > 0, "! expect 'only_files', but found {} dirs and {} files in {}".format(_ld, _lf, fp)
            if _only_img:
                imgs = list(filter(lambda x: x.endswith(_ext), _files))
                assert len(imgs) == _lf, "! expect '_only_img', found {} img and {} file in {}".format(len(imgs), _lf, fp)
    return _fns, _dirs, _files, _ld, _lf, _ldf
# end get_fns
def list321(_list, remove_none=False, _low=False):
    _list2 = list(set(_list))  # remove redundant elements (None)
    if remove_none or (len(_list) > 1 and None in _list): _list2 = [l for l in _list if l not in [None, 'none', 'None', '']]
    _list2.sort()  # !! return is None, so do NOT '='
    if _low: _list2 = [l.lower() for l in _list2]
    if len(_list) != len(_list2): print(f"- update list from {_list} to {_list2}")
    return _list2
_prt_once = []
def print1(s, idx):
    global _prt_once
    if not isinstance(s, str): s = str(s)
    if idx not in _prt_once:
        print(s)
        _prt_once.append(idx)
def os_system3(src, dst, _op=CP_FILE):  # os_system(src, dst, CP_FILE)
    r = os.system(f"{_op} {src} {dst}")
    if r != 0:
        _name = os.path.basename(src)
        _str = "!! make sure no ' ', '(', ')' in you file or folder name !!\n"
        if ' ' in _name or '(' in _name or ')' in _name:
            print(f"!! check '{_name}' NOT contains:{_str}\n use rename_file(--str0='(' --str2='_') to replace '('")
        input(f"! (r={r}) failed:{_op} {src}\n to {dst}\n make sure '=' is NOT in the path, input 'c' to continue\n")
        return 0
    return 1  # 1 for += 1
# print(f" best for classification: cross_both1024_, activate='relu', drop= 0.0, epoch=100"
#       f" best for bil:  AdamW, lr=0.001,  wd=0.001\n"
#       f" best for sz:   AdamW, lr=0.001,  wd=0.0005\n"
#       f" best for com2: AdamW, lr=0.0001, wd=0.0001\n"
#       f" this={name_cls}: {TXT_FLAG_ALL}")

if __name__ == '__main__':
    print(f"- mm_flag.py done at: {time_str()}")