# Copyright (c) OpenMMLab. All rights reserved.
''' mm_det.py, chengyu wang, XJTLU, created=2021.11.23, last updated: 2022-03-30
*: os  $$: gpu  ##: dataset  &: image  ---: model  --: loss  -: directory
single_gpu_test(): xx/mmdetection/mmdet/apis/test.py
    --> line 28: model
    --> line 52: show_result()
        xx/mmdetection/mmdet/models/detectors/base.py # NOT single_stage_instance_seg.py ->255
        --> line 261: def show_result()
        xx/mmdetection/mmdet/core/visualization/image.py
        --> line 29: def imshow_det_bboxes()
            --> line 148: '' if 'chr' in label_text else f'{label_text}',
            !!! pt_flag.py -> PLT_LB_NAME
'''

import sys, os
_import_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_import_root)
print("mm_seg.py _import_root={}".format(_import_root))

from pt_flag import *
assert BB_SEG[:len('mm_')] == 'mm_', "this is mm_seg.py: check BB_SEG={}".format(BB_SEG)
# check_flag('seg')
from all_include.pt_head import find_sth_easy, set_gpu, join_path, end_this_py
from all_include.mm_utils import mm_eval_json, find_and_pick, manual_load_from, prepare_coco
from argparse import ArgumentParser

# import argparse
import copy, os, warnings
import os.path as osp

# from tools/train.py
import mmcv, torch
from mmcv import DictAction # Config,
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmcv.cnn import fuse_conv_bn

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector, multi_gpu_test # , single_gpu_test, inference_detector
from mmdet.datasets import build_dataset, build_dataloader, replace_ImageToTensor
from mmdet.models import build_detector, build_backbone # build_backbone
from mmdet.utils import collect_env, get_root_logger
# from mmdetection.tools.analysis_tools import analyze_logs

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (load_checkpoint, wrap_fp16_model) # get_dist_info, init_dist,
# from mmcls.apis.inference import show_result_pyplot
from all_include.mm_utils import mm_seg_test, get_cfg, pick_pth, del_pth
def parse_args():
    config_file = MM_PY_CFG_FILE
    if not osp.exists(config_file): input3("!NOT found:{}".format(config_file))
    checkpoint = None # set in train_mm, or eval_mm
    if SEG_RESUME_TRAIN:
        r = 'latest' # input(f"input '.pth' key to continue training:\n")
        resume = find_sth_easy(OUTPUT_DIR_SEG, [BB_SEG], dKey=r, ext2='.pth')
    else: resume = None
    bb_name, seg_flag = BB_SEG, f"{sub_str(DATA_NAME_SEG, '_')}" # f"{SUF_ABN[1:SUF_ABN.rindex(('_' + TM))]}_{DATA_NAME_ABN[DATA_NAME_ABN.rindex('_') + 1:]}"

    gpu_all, device, gpu_ids = set_gpu(USE_GPU_LIST)  # set this only ONCE !!
    gpu_all, gpu_ids = None, USE_GPU_LIST # !! we do NOT use 'distributed'
    # backbone = 'resnet50' # default
    fn_final = FN_FINAL_SEG
    img_demo = osp.join(ROOT_DATASET, 'mm_data', 'demo.jpg') 
    # data_root = DATA_ROOT_SEG
    # coco_path = osp.join(ROOT_DATASET, '{}_data'.format(DATA_NAME_SEG))
    output_dir = OUTPUT_THIS_SEG
    dn = DATA_NAME_SEG
    
    fnp_log = LOG_THIS_SEG
    fnp_log_eval = fnp_log.replace('.txt', '_eval.txt')
    fp_last = FP_LAST_SEG
    no_validate = NOT_VALID_DURING_TRAINING
    eval_only = SEG_NOT_TRAIN
    eval_str = ['bbox', 'segm'] if ('mrcnn' in BB_SEG or 'swin' in BB_SEG) else ['bbox']
    seg_score = SEG_SCORE
    temp_dir = os.path.join(output_dir, 'tmp_dir')
    out_pkl = os.path.join(output_dir, 'out_{}_{}.pkl'.format(BB_SEG, SUF_SEG))
    last_pth = join_path([output_dir, 'latest.pth'])
    fp_pth_all = FP_PTH_ALL # osp.join(OUTPUT_DIR_SEG, f"pth_all_{DATA_NAME_SEG}_per{SAMPLES_PER_GPU}", osp.basename(output_dir))
    # fp_pth_all = osp.join(OUTPUT_DIR_SEG, f"pth_all{dn[dn.rindex('_'):]}_per{SAMPLES_PER_GPU}", osp.basename(OUTPUT_THIS_SEG))
    # fp_plt_eval = os.path.join(output_dir, 'plt_eval')
    fp_plt_eval = osp.join(OUTPUT_DIR_SEG, f"plt_eval_{DATA_NAME_SEG}", osp.basename(output_dir)) # (output_dir, 'plt_eval')
    train_json = osp.join(output_dir, f"{bb_name}_{seg_flag}_{TM}.log.json")
    if OUTPUT_DIR_SEG not in fp_pth_all: input3(f"!! set check_flag('seg'):{fp_pth_all}")

    parser = ArgumentParser('get_args_parser()', add_help=False)
    parser.add_argument('--img', default=img_demo, help='Image file')
    # parser.add_argument('--data_root', default=data_root, help='Image file')
    parser.add_argument('--config', default=config_file, help='Config file')
    parser.add_argument('--checkpoint', default=checkpoint, help='Checkpoint file for eval, set in train_mm or eval_mm')
    parser.add_argument('--device', default=device, help='Device used for inference') # 'cuda:0'
    parser.add_argument('--score_thr', type=float, default=seg_score, help='bbox score threshold') # 0.3
    parser.add_argument('--async_test', action='store_true', help='whether to set async options for async inference.')

    # from tools/train.py
    # parser.add_argument('--config', default=config_file, help='train config file path')
    parser.add_argument('--work_dir', default=output_dir, help='the dir to save logs and models') # work-dir
    parser.add_argument('--resume_from', default=resume, help='the checkpoint file to resume from') # resume-from
    parser.add_argument('--no_validate', default=no_validate, type=bool,
        help='whether not to evaluate the checkpoint during training') # no-validate, action='store_true',
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', default=gpu_all, type=int, help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu_ids', default=gpu_ids, type=int, nargs='+', help='ids of gpus to use, !! let MM do itself '
        '(only applicable to non-distributed training)') # gpu-ids
    parser.add_argument('--seed', type=int, default=1, help='random seed') # None
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    # from tools/test.py
    parser.add_argument('--out', default=out_pkl, help='output result file in pickle format')    
    parser.add_argument('--fuse_conv_bn', action='store_true', help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed') # fuse-conv-bn
    parser.add_argument('--format_only', action='store_true', help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and submit it to the test server') # format-only
    parser.add_argument('--eval', default=eval_str, nargs='+', help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', default=True, action='store_true', help='show results')
    parser.add_argument('--show_dir', default='', help='directory where painted images will be saved') # show-dir
    parser.add_argument('--show_score_thr', type=float, default=seg_score, help='score threshold (default: 0.3)') # show-score-thr 0.3
    parser.add_argument('--gpu_collect', action='store_true', help='whether to use gpu to collect results.') # gpu-collect
    parser.add_argument('--tmpdir', default=temp_dir, help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--eval_options', nargs='+', action=DictAction, help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function') # eval-options

    # MB321
    parser.add_argument('--bb_name', default=bb_name, type=str, help='MB321')
    parser.add_argument('--seg_flag', default=seg_flag, type=str)
    parser.add_argument('--train_json', default=train_json, type=str)
    parser.add_argument('--fp_pth_all', type=str, default=fp_pth_all)
    parser.add_argument('--fp_plt_eval', type=str, default=fp_plt_eval)
    parser.add_argument('--last_pth', type=str, default=last_pth)
    parser.add_argument('--fnp_log_eval', type=str, default=fnp_log_eval)
    parser.add_argument('--fnp_log', type=str, default=fnp_log)
    parser.add_argument('--output_dir', type=str, default=output_dir) # work_dir
    parser.add_argument('--fp_last', type=str, default=fp_last)
    # parser.add_argument('--load_from', type=str, default=load_from) # by MM_PTH_TRAINED
    parser.add_argument('--fn_final', type=str, default=fn_final)
    parser.add_argument('--eval_only', type=bool, default=eval_only)
    parser.add_argument('--show_many', type=bool, default=SHOW_MANY)

    parser.add_argument('--cfg_options', nargs='+', action=DictAction, help='hehe') # cfg-options
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options for evaluation, \
        the key-value pair in xxx=yyy format will be kwargs for dataset.evaluate() function (deprecate), change to --eval-options instead.')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ: os.environ['LOCAL_RANK'] = str(args.local_rank)
    if eval_only: # eval
        if args.options and args.eval_options:
            raise ValueError( '--options and --eval-options cannot be both specified, --options is deprecated in favor of --eval-options')
        if args.options:
            warnings.warn('--options is deprecated in favor of --eval-options')
            args.eval_options = args.options
    else: # train
        # parser.add_argument('--options', nargs='+', action=DictAction, help='override some settings in the used config, \
        #  the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.')
        # parser.add_argument('--cfg_options', nargs='+', action=DictAction, help='override some settings in the used config, the key-value pair '
        #     'in xxx=yyy format will be merged into config file. If the value to  be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        #     'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]"  Note that the quotation marks are necessary and that no white space '
        #     'is allowed.') # cfg-options
        # args = parser.parse_args()
    
        if args.options and args.cfg_options: raise ValueError('--options and --cfg-options cannot be both '
                'specified, --options is deprecated in favor of --cfg-options')
        if args.options:
            warnings.warn('--options is deprecated in favor of --cfg-options')
            args.cfg_options = args.options
    
    # args = parser.parse_args() 
    return args
# end parse_args

def train_mm(args):
    t_tr = datetime.datetime.now()
    write_txt([args.fnp_log_eval], f"*start training at:{t_tr.strftime('%m%d_%H%M%S')}")
    # write_txt([args.fnp_log], "*start training at:{}, output_dir={}".format(t_tr.strftime("%m%d_%H%M%S"), args.work_dir))

    cfg = get_cfg(args)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none': distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # init the logger before other steps
    timestamp = TM # time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{BB_SEG}_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    if args.show_many: logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    if args.show_many: logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    if args.show_many: logger.info(f'Set random seed to {seed}, 'f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    if_you_need_backbone = build_backbone(cfg.model.backbone)
    # model = if_you_need_backbone # !! object has no attribute 'train_step'
    model.init_weights()
    manual_load_from(model, cfg.load_from, args.fnp_log)

    # if args.show_many:
        # write_txt([args.fnp_log], f"model.parameters=\n {model.parameters}")
        # from all_include.pt_cnn import MLP, models
        # n1 = MLP(256, 100)
        # torchsummary.summary(n1, input_size=(3, 224, 224), device='cpu')
        # n2 = models.resnet50(pretrained=False, num_classes=NUM_CLASS_SEG)
        # torchsummary.summary(n2, input_size=(3, 224, 224), device='cpu')
        # torchsummary.summary(model, input_size=(3, 224, 224), device='cpu')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2: # we use ==1 as: [('train', 1)]
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets, cfg, distributed=distributed, validate=(not args.no_validate), timestamp=timestamp,meta=meta)
    src = osp.join(args.work_dir, f"{TM}.log.json")
    r = os_system3(src, args.train_json, MOVE)  # src name can't changed -_- ?
    _fnp_eval = str(args.fnp_log).replace('.txt', '_eval.txt') # just for easy read from _eval.txt
    write_txt([args.fnp_log, _fnp_eval, LOG_ALL_SEG], f"* (r={r})end train at:{time_str()}, cost{time_gap(t_tr)}, {osp.basename(args.work_dir)}, output in:{osp.dirname(args.work_dir)}\n")

    # _fnp_model_final = os.path.join(args.output_dir, 'latest.pth')
    # _fnp_pkl_copy = os.path.join(args.fp_last, args.fn_final)  # copy into 'xx/last'
    # write_txt([args.fnp_log], "coping final pkl from: {} \n to: {} ...".format(_fnp_model_final, _fnp_pkl_copy))
    # _r = os.system("{} {} {}".format(CP_FILE, _fnp_model_final, _fnp_pkl_copy))
    # if _r != 0: input3("! failed with {}".format(_r))
    # write_txt([args.fnp_log], "*end train at:{}, cost{}\n going to eval, with last '.pth':\n {}"
    #           .format(time_str(), time_gap(t_tr), _fnp_pkl_copy))
    # # args.config = args.out
    # args.checkpoint = _fnp_pkl_copy # for eval
    return args
# end train_mm

def eval_mm(args, _find_in_last='latest'): # best, latest
    t_v = datetime.datetime.now()
    write_txt([args.fnp_log_eval], f"*start eval at:{t_v.strftime('%m%d_%H%M%S')}, args.checkpoint=None({args.checkpoint is None})")

    checkpoint = args.checkpoint if args.checkpoint is not None else\
        (find_sth_easy(args.fp_last, ['.pth', BB_SEG, _find_in_last]) if MB321_SEG
         else find_sth_easy(FP_TRAINED, ['.pth', TRAINED_PTH[BB_SEG]])) # coco from output/trained    
    
    nn = sub_str(osp.basename(checkpoint).split('.pth')[0], sub_str(DATA_NAME_ABN, '_')).lstrip('_')
    args.show_dir = osp.join(args.fp_plt_eval, f"{nn}") # to distinguish different '.pth'
    if args.eval and args.format_only: raise ValueError('--eval and --format_only cannot be both specified')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')): raise ValueError('The output file must be a pkl file.')
    cfg = get_cfg(args)
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test: ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test: ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none': distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    timestamp = TM # time.strftime('%Y%m%d_%H%M%S', time.localtime())
    json_file = osp.join(args.work_dir, "eval_{}.json".format(osp.basename(checkpoint)[:-len('.pth')]))

    # build the dataloader
    if not (osp.exists(cfg.data.test['ann_file']) and osp.exists(osp.join(cfg.data.test.img_prefix, 'JPEGImages')) ):
        print(f"!! not found, trying make FAKE coco to {cfg.data.test['ann_file']}\n") # input 's' to skip, or 'c' to 
        prepare_coco(True) # if r.lower() == 'c': prepare_coco(True)
        # elif r.lower() == 's': break

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    do_not_need_rtn = load_checkpoint(model, checkpoint, map_location='cpu')
    if args.show_many:
        write_txt([args.fnp_log_eval], f"model.parameters=\n {model.parameters}")
        # torchsummary.summary(model, input_size=(3, 224, 224), device='cpu') # NOT work for eval!

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility

    # if 'CLASSES' in checkpoint.get('meta', {}): model.CLASSES = checkpoint['meta']['CLASSES']
    # else: model.CLASSES = dataset.CLASSES
    model.CLASSES = dataset.CLASSES; outputs_err_on911 = None

    write_txt([args.fnp_log_eval], f"just going to evaluate with distributed={distributed} (suppose False)...")
    if not distributed:
        model = MMDataParallel(model, device_ids=args.gpu_ids)
        outputs_err_on911 = mm_seg_test(model, data_loader, args.show, args.show_dir, args.show_score_thr)
    else:
        input3(f"?? distributed={distributed} for MMDistributedDataParallel")
        model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        outputs_not_used = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs_err_on911, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only: dataset.format_results(outputs_err_on911, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs_err_on911, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0: mmcv.dump(metric_dict, json_file)
    ## do following two lines in: mmdetection\mmdet\datasets\mb321_mchr.py -->evaluate()  
    # result = inference_detector(model, args.img) 
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr, wait_time=2)
    tt = datetime.datetime.now()
    write_txt([args.fnp_log_eval, args.fnp_log], "*end eval at:{}, eval cost{}, total cost{}".format(
        tt, time_gap(tt-t_v), time_gap(tt-TIME_TH)))
    mm_eval_json([json_file], args.fnp_log_eval, len(data_loader))
# end eval_mm

def launch_mm_seg(): # (cfg_file, eval_only, resume_train, _fnp_log, data_root, show_many, _f=''):
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=False) # TM is unique
    write_txt([args.fnp_log], f"-input_dir\t={DATA_SPLIT_SEG}\n output_dir\t={args.work_dir}")

    if USE_TOOLS not in ['0', False, 0, 'F', 'False', 'false']:
        use_tools(args)
        if USE_TOOLS in ['1', 1, 'T', 'True', 'true']: write_txt([args.fnp_log], f"--- return for '0 - only use tools at {time_str()}"); return

    if not args.eval_only: # train + valid
        args = train_mm(args)
        args = pick_pth(args, 'mm_seg')
        pth_list = [args.checkpoint, args.last_pth]
        del_pth(args.work_dir, args.fp_pth_all, osp.basename(args.checkpoint))
    else: # only valid
        write_txt([args.fnp_log], "*eval skip train, current checkpoint={}".format(args.checkpoint))
        if B_FIND_AND_PICK:
            args = find_and_pick(args)
            pth_list = [args.checkpoint, args.last_pth]
        else: pth_list = [None] # will pick from fp_last
    pth_list = list321(pth_list)
    write_txt([args.fnp_log], f"--- pth_list={pth_list}")
    for args.checkpoint in pth_list: # args.checkpoint must be the first
        if args.checkpoint is not None: eval_mm(args, 'load_without_find')
        else:
            # eval_mm(args, 'best')
            eval_mm(args, 'latest')
    # return osp.join(args.output_dir, CHR5_SEG_PRED), osp.join(args.output_dir, 'plt_seg')
# end launch_mm_seg

def use_tools(args, op='get_flops'):
    write_txt([args.fnp_log], f"## use_tools with op={op}")
    if op in ['get_flops']:
        import numpy as np
        # import torch
        # from mmcv import Config, DictAction

        try:
            from mmcv.cnn import get_model_complexity_info
        except ImportError:
            raise ImportError('Please upgrade mmcv to >0.6.2')

        # args = parse_args()
        #
        # if len(args.shape) == 1:
        #     h = w = args.shape[0]
        # elif len(args.shape) == 2:
        #     h, w = args.shape
        # else:
        #     raise ValueError('invalid input shape')
        ds = DATA_NAME_SEG[DATA_NAME_SEG.rindex('_')+1:]
        mci_shape = {'bil':[768, 576], 'adir':[645, 517], 'sz':[1600,1200]} # [w,h] of [BIL, ADIR, SZ]
        w, h = mci_shape[ds][0], mci_shape[ds][1]
        write_txt([args.fnp_log], f" dataset is {ds}, shape[w,h]={mci_shape}")
        ori_shape = (3, h, w)
        divisor = 32 # args.size_divisor
        if divisor > 0:
            h = int(np.ceil(h / divisor)) * divisor
            w = int(np.ceil(w / divisor)) * divisor
        input_shape = (3, h, w)

        cfg = get_cfg(args) # Config.fromfile(args.config)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        else:
            raise NotImplementedError(
                'FLOPs counter is currently not currently supported with {}'.
                    format(model.__class__.__name__))

        flops, params = get_model_complexity_info(model, input_shape)
        split_line = '=' * 30

        if divisor > 0 and \
                input_shape != ori_shape:
            print(f'{split_line}\nUse size divisor set input shape '
                  f'from {ori_shape} to {input_shape}\n')
        print(f'{split_line}\nInput shape: {input_shape}\n'
              f'Flops: {flops}\nParams: {params}\n{split_line}')
        print('!!!Please be cautious if you use the results in papers. '
              'You may need to check if all ops are supported and verify that the '
              'flops computation is correct.')

if __name__ == '__main__':    
    print(f"start mm_seg.py @ {time_str()}")
    launch_mm_seg()
    end_this_py([LOG_THIS_SEG])  # mm_seg.py
