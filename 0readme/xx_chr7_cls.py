'''  mm_cls.py, Chengyu Wang, XJTLU, 2023-1107, last update on 2023-1107
*: os  $$: gpu  ##: dataset  &: image  ---: model  --: loss, etc.  -: directory '''

import sys, os
_import_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_import_root)
print("mm_cls.py _import_root={}".format(_import_root))

from MB321.mm_flag import *
assert 'cls' in agBASE.task, f"this is mm_cls.py: check agBASE.task={agBASE.task}"
from MB321.base.util321 import find_sth_easy, set_gpu, end_this_py # join_path, 
from MB321.base.mm_utils import manual_load_from, get_cfg, pick_pth, del_pth, find_and_pick, mm_eval_json, print_bb_struc
from MB321.base.mm_vis import init_args_dict, vis_cam
from MB321.base.mm_vis_feat import vis_feat
# from all_include.chr_karyo import img_repair, image_repairs
from argparse import ArgumentParser
# from mmcv.cnn.utils import get_model_complexity_info

# import argparse
import copy, os, datetime #, time, warnings
import os.path as osp

# from tools/train.py
import mmcv, torch
# from mmcv import Config #, DictAction
# from mmcv.runner import get_dist_info, init_dist
# from mmcv.utils import get_git_hash
from mmcls import __version__
from mmcls.apis import init_random_seed, set_random_seed, train_model
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import collect_env, get_root_logger # , setup_multi_processes


from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmcls.apis import multi_gpu_test, single_gpu_test
last_str_all = [f'{SUF_ABN}, DROP_PATH_RATE={DROP_PATH_RATE}\n']
def parse_args():
    metrics_in_eval = MM_CLS_METRICS # 
    metric_options_in_eval = {'topk': (1, TOP_K), 'thrs': 0.5} # 'thrs': ??
    bb_name, abn_flag = BB_ABN, f"{SUF_ABN[1:SUF_ABN.rindex(('_'+TM))]}_{sub_str(DATA_NAME_ABN, '_')}"

    config_file = MM_PY_CFG_FILE_ABN
    if not osp.exists(config_file): input3("!NOT found:{}".format(config_file))
    checkpoint = None # set in train_mm, or eval_mm
    if ABN_RESUME_TRAIN:
        r = 'latest' # input(f"input '.pth' key to continue training:\n")
        resume = find_sth_easy(OUTPUT_DIR_ABN, [bb_name], dKey=r, ext2='.pth')
    else: resume = None

    gpu_all, device, gpu_ids = set_gpu(USE_GPU_LIST)  # set this only ONCE !!
    gpu_ids = USE_GPU_LIST # gpu_id = USE_GPU_LIST[0]
    gpu_all = gpu_ids # None
    fn_final = FN_FINAL_ABN
    output_dir = OUTPUT_THIS_ABN
    fp_pth_all = FP_PTH_ALL
    fp_plt_eval = osp.join(OUTPUT_DIR_ABN, f"plt_eval_{DATA_NAME_ABN}", osp.basename(output_dir)) # (output_dir, 'plt_eval')
    fp_pred = osp.join(OUTPUT_DIR_ABN, f"plt_pred_{DATA_NAME_ABN}", osp.basename(output_dir))
    train_json = osp.join(output_dir, f"{bb_name}_{abn_flag}_{TM}.log.json")
    if OUTPUT_DIR_ABN not in fp_pth_all: input3(f"!! set check_flag('abn'):{fp_pth_all}")

    fnp_log = LOG_THIS_ABN
    fnp_log_eval = fnp_log.replace('.txt', '_eval.txt')
    fp_last = FP_LAST_ABN
    no_validate = NOT_VALID_DURING_TRAINING
    eval_only = NOT_TRAIN
    # eval_str = ['bbox', 'segm'] if 'mm_mrcnn' in bb_name else ['bbox']
    temp_dir = os.path.join(output_dir.replace(DATA_NAME_ABN, DATA_NAME_ABN+'_tmp'), 'tmp_dir')
    # out_pkl = os.path.join(output_dir, f"out_{bb_name}{SUF_ABN}.pkl")
    # eval_json = osp.join(output_dir, f"eval_{bb_name}{SUF_ABN}.json")    
    last_pth = os.path.join(output_dir, 'latest.pth') #join_path([output_dir, 'latest.pth'])
    show_dir = os.path.join(output_dir, 'img_show_nothing')

    parser = ArgumentParser('chr7_cls.py get_args_parser()', add_help=False)
    # from tools/train.py
    parser.add_argument('--config', default=config_file, help='train config file path')
    parser.add_argument('--work_dir', default=output_dir, help='the dir to save logs and models') # '--work-dir'
    parser.add_argument('--checkpoint', default=checkpoint, help='the checkpoint file to resume from, set in train_mm or eval_mm') # '--resume-from'    
    parser.add_argument('--no_validate', default=no_validate, type=bool, help='whether not to evaluate the checkpoint during training') # '--no-validate', action='store_true'
    parser.add_argument('--train_json', default=train_json, type=str, help='MB321')
    
        
    group_gpus = parser.add_mutually_exclusive_group() # !! group_gpus
    group_gpus.add_argument('--device', default=device, help='device used for training/testing. (Deprecated)') # 'cuda:0'
    group_gpus.add_argument('--gpus', default=gpu_all, type=int)
    group_gpus.add_argument('--gpu_ids', default=gpu_ids, type=int,nargs='+') # '--gpu-ids',
    # group_gpus.add_argument('--gpu_id', type=int, default=gpu_id) # --gpu-id
    
    parser.add_argument('--resume_from', default=resume, help='the checkpoint file to resume from') # resume-from
    parser.add_argument('--seed', type=int, default=1, help='random seed') # None
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--cfg_options', nargs='+', action=DictAction, help='override some settings') # --cfg-options
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
   
    # from tools/test.py
    parser.add_argument('--out', default='set in eval_mm()', help='output result file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument('--out_items', nargs='+', default=['all'], choices=out_options + ['none', 'all'], help='Besides metrics, ', metavar='') # --out-items
    parser.add_argument('--metrics', type=str, default=metrics_in_eval, nargs='+')
    # parser.add_argument('--show', default=False, action='store_true', help='show results') # too many img
    parser.add_argument('--show_dir', default=show_dir, help='directory where painted images will be saved') # show-dir
    parser.add_argument('--gpu_collect', action='store_true', help='whether to use gpu to collect results') # --gpu-collect
    parser.add_argument('--tmpdir', default=temp_dir, help='tmp dir for writing some results')

    # MB321
    # parser.add_argument('--load_from', type=str, default=load_from) # !!! by ABN_PTH_TRAINED
    parser.add_argument('--bb_name', type=str, default=bb_name)
    parser.add_argument('--abn_flag', type=str, default=abn_flag)
    parser.add_argument('--output_dir', type=str, default=output_dir) # work_dir
    parser.add_argument('--fp_last', type=str, default=fp_last)
    parser.add_argument('--fp_pred', type=str, default=fp_pred)
    parser.add_argument('--fp_pth_all', type=str, default=fp_pth_all)
    parser.add_argument('--fp_plt_eval', type=str, default=fp_plt_eval)
    parser.add_argument('--fnp_log_eval', type=str, default=fnp_log_eval)
    parser.add_argument('--fnp_log', type=str, default=fnp_log)
    parser.add_argument('--fn_final', type=str, default=fn_final)
    parser.add_argument('--last_pth', type=str, default=last_pth)
    parser.add_argument('--eval_only', type=bool, default=eval_only)
    parser.add_argument('--show_many', type=bool, default=SHOW_MANY)
    parser.add_argument('--epochs', type=int, default=CLS_MAX_EPOCH)
    parser.add_argument('--vit_like', type=bool, default=VIT_LIKE)

    parser.add_argument('--metric_options', nargs='+', action=DictAction, default=metric_options_in_eval, help='custom options for evaluation, the key-value pair in xxx=yyy ' 'format will be parsed as a dict metric_options for dataset.evaluate()' ' function.') # --metric-options
    parser.add_argument('--show_options', nargs='+', action=DictAction, help='custom options for show_result. key-value pair in xxx=yyy.' 'Check available options in `model.show_result`.')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ: os.environ['LOCAL_RANK'] = str(args.local_rank)
    assert args.metrics or args.out, 'Please specify at least one of output path and evaluation metrics.'
    return args
# end parse_args

def train_mm(args):
    t_tr = datetime.datetime.now()
    write_txt([args.fnp_log], f"*start training at:{t_tr.strftime('%H%M%S')}")
    cfg = get_cfg(args)
        
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none': distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    timestamp = TM # time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{args.bb_name}_{timestamp}.log')
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

    model = build_classifier(cfg.model)
    model.init_weights()
    manual_load_from(model, cfg.load_from, args.fnp_log)
    if int(cfg.gpu_ids[0]) >= 0: write_txt(LOG_THIS_ABN, "** model to cuda for train"); model.cuda()

    datasets = [build_dataset(cfg.data.train)] # ChrDataset

    if len(cfg.workflow) == 2: # we use ==1 as: [('train', 1)]
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    # save mmcls version, config file content and class names in runner as meta data
    meta.update(dict(mmcls_version=__version__, config=cfg.pretty_text, CLASSES=datasets[0].CLASSES))

    # add an attribute for visualization convenience
    write_txt([args.fnp_log], f"just going to train at {time_str()}")
    train_model(model, datasets, cfg, distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, device=args.device, meta=meta) # 'cpu' if args.device == 'cpu' else 'cuda'

    src = osp.join(args.work_dir, f"{TM}.log.json")
    r = os_system3(src, args.train_json, MOVE) # src name can't changed -_- ?
    _fnp_eval = str(args.fnp_log).replace('.txt', '_eval.txt') # just for easy read from _eval.txt
    write_txt([args.fnp_log, _fnp_eval, LOG_ALL_ABN],
              f"* (r={r})end train at:{time_str()}, cost{time_gap(t_tr)}, {osp.basename(args.work_dir)}, output in:{osp.dirname(args.work_dir)}\n")
    return args
# end train_mm

def eval_mm(args, _find_in_last, set_data='', do_metric=(True if (QUICK and ABN_NOT_TRAIN) else TRUE)): # best, latest
    t_v = datetime.datetime.now(); json_log, _tmp_flag = '', f"{_find_in_last}, {set_data}"
    write_txt([args.fnp_log_eval], f"*start eval at:{t_v.strftime('%H%M%S')}, {_tmp_flag}")
    if not args.checkpoint:
        _base = ['.pth', args.bb_name, FIND_PTH_KEY, f"cls{NUM_CLASS_ABN}_"]
        if type(_find_in_last) is str: _find_in_last = [_find_in_last]
        args.checkpoint = find_sth_easy(args.fp_last, _base + _find_in_last, _confirm=False)
        if args.checkpoint is None:
            input3("!! skip eval_mm, as not found '.pth'\n") # [args.fnp_log_eval], 
            return _find_in_last, args.checkpoint
    
    args.out = osp.join(args.work_dir, "eval_{}.json".format(osp.basename(args.checkpoint)[:-len('.pth')]))  
    nn = sub_str(osp.basename(args.checkpoint).split('.pth')[0], sub_str(DATA_NAME_ABN, '_')).lstrip('_')
    args.show_dir = osp.join(args.fp_plt_eval, f"{nn}") # to distinguish different '.pth'
    write_txt([args.fnp_log_eval, args.fnp_log], f" loaded checkpoint:{args.checkpoint}\n show_dir:, {args.show_dir}", b_prt=False)

    cfg = get_cfg(args) # Config.fromfile(args.config)
    cfg.model.pretrained = None
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none': distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    if set_data.startswith('abn'): # for EVAL_ALL
        assert 'abn886_' in osp.basename(osp.dirname(cfg.data.test.data_prefix)), "! set ABN_RATIO=0('abn886_') in:{cfg.data.test}\n"
        cfg.data.test.data_prefix = cfg.data.test.data_prefix.replace('abn886_', set_data.rstrip('_') + '_') # make sure '_'
        write_txt([args.fnp_log_eval], f"{'~'*30}set cfg.data.test.data_prefix=\n{cfg.data.test.data_prefix}")
    elif set_data.startswith('seg_pred_fit'):
        if not osp.exists(MM_DATA_MB321['seg_pred_fit']):
            if 'c' == input3(f"!! NOT found:{MM_DATA_MB321['seg_pred_fit']}\n change to:{MM_DATA_MB321['seg_pred']}"):
                set_data = 'seg_pred'
        else:
            args.metrics = None  # for NOT evaluate
            cfg.data.test.data_prefix = MM_DATA_MB321['seg_pred_fit']
    elif (set_data not in image_repairs) and len(set_data) > 0: f"?? unexpected set_data={set_data}\n"
    if set_data in image_repairs: # ['seg_pred', 'ver_fit']
        args.metrics = None  # for NOT evaluate
        _fp_cls_in = MM_DATA_MB321['seg_pred']  # abn_ichr_bil_data/seg_pred
        cfg.data.test.data_prefix = img_repair(set_data, _fp_cls_in, 'ichr_bil', args.fnp_log, args.fp_pred)

    write_txt(args.fnp_log_eval, f"PLT_KARYO={PLT_KARYO}, do_metric={do_metric}, args.metrics={args.metrics}")
    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))
    data_loader = build_dataloader(dataset, samples_per_gpu=cfg.data.samples_per_gpu, workers_per_gpu=cfg.data.workers_per_gpu, dist=distributed, shuffle=False)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None: wrap_fp16_model(model)
    do_not_need_rtn = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.show_many: write_txt([args.fnp_log_eval], f"model.parameters=\n {model.parameters}")
        
    CLASSES, outputs_pred = dataset.CLASSES, []
    write_txt([args.fnp_log_eval], f"just going to evaluate({_tmp_flag}) with distributed={distributed} (suppose False)...")
    if not distributed:
        if args.device == 'cpu': model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            if int(cfg.gpu_ids[0]) >= 0: write_txt(LOG_THIS_ABN, "** model to cuda for eval") # ; model.cuda()
            if not model.device_ids: assert mmcv.digit_version(mmcv.__version__) >= (1, 4, 4), \
                    'To test with CPU, please confirm your mmcv version ' \
                    'is not lower than v1.4.4'
        model.CLASSES = CLASSES
        show_kwargs = {} if args.show_options is None else args.show_options
        if do_metric: outputs_pred = single_gpu_test(model, data_loader, show=False, out_dir=args.show_dir, **show_kwargs) #
    else:
        input3(f"?? distributed={distributed} for MMDistributedDataParallel")
        model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        if do_metric: outputs_pred = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    model0 = model.module.backbone # for further use 
    
    print(f" len(outputs_pred) = {len(outputs_pred)}")
    rank, _ = get_dist_info()
    if len(outputs_pred) > 0 and rank == 0:
        results, to_json = {}, {}
        logger = get_root_logger()
        if args.metrics:
            eval_results = dataset.evaluate(results=outputs_pred, metric=args.metrics, metric_options=args.metric_options, logger=logger)
            _cm_str = dataset.confusion_matrix(outputs_pred, args.out) # must args.metrics is True
            last_str_all.append(_cm_str)  # +'\n'
            results.update(eval_results)
            for k, v in eval_results.items():
                if isinstance(v, np.ndarray): v = [round(out, PRINT_DECIMAL) for out in v.tolist()]
                elif isinstance(v, Number): v = round(v, PRINT_DECIMAL)
                else: raise ValueError(f'Unsupport metric type: {type(v)}')
                print(f'\n{k} : {v}')
                eval_results[k] = v
            to_json = {'metric':eval_results}
        if args.out:
            if 'none' not in args.out_items:
                scores = np.vstack(outputs_pred)
                pred_score = np.max(scores, axis=1)
                pred_label = np.argmax(scores, axis=1)
                pred_class = [CLASSES[lb] for lb in pred_label]
                res_items = {'class_scores': scores, 'pred_score': pred_score, 'pred_label': pred_label, 'pred_class': pred_class}
                if 'all' in args.out_items:
                    results.update(res_items)
                    to_json.update({'res':res_items})
                else:
                    for key in args.out_items:
                        results[key] = res_items[key]
                        to_json.update({'res': {key:res_items[key]}})
            print(f'\ndumping results to {args.out}')
            mmcv.dump(to_json, args.out) # results
        tt = datetime.datetime.now()
        write_txt([args.fnp_log_eval, args.fnp_log], f"*end eval at:{tt}, eval cost{time_gap(tt-t_v)}, total cost{time_gap(tt)}")
        if PLT_KARYO > 0: last_str_all.append(dataset.plt_karyo(outputs_pred))
        # # output_size, params_size, total_size = summary(model.backbone, 'cpu', args.fnp_log, args.show_many) # args.device.type
        # if hasattr(model, 'extract_feat'): model.forward = model.extract_feat
        # flops, params = get_model_complexity_info(model, (IMG_C, IMG_H, IMG_W))
        if EVAL_ALL: json_log = mm_eval_json([args.out], args.fnp_log_eval, len(data_loader)) # , f"model[flops, params]=[{flops}, {params}]") # , f"model[output_size, params_size, total_size]=[{output_size}, {params_size}, {total_size}]"
    
    if 'wh' in osp.basename(args.checkpoint):
        _str = osp.basename(args.checkpoint)[osp.basename(args.checkpoint).index('wh'):]
        img_size = int(_str[len('wh'):_str.index('_')])
    else: img_size = 224 # default
    if PLT_FEAT: vis_feat(model.module.backbone, 'cpu', args.vit_like, cfg.data.test.data_prefix, osp.join(args.fp_plt_eval, f'vis_feat_{osp.basename(cfg.data.test.data_prefix)}_{TM}'), img_size, img_size) # args.device, MM_DATA_MB321['ichr_test']
    if PLT_CAM:
        data_prefix = cfg.data.test.data_prefix
        _best_or_latest = 'best' if 'best' in osp.basename(args.checkpoint) else 'latest'
        args_key = {'config': args.config, 'checkpoint': args.checkpoint,
                    'img': osp.join(data_prefix, ABN_DATA_TYPE).rstrip('_'),
                    'save_path': osp.join(args.fp_plt_eval, f'cam_out_{_best_or_latest}_{osp.basename(data_prefix)}'),
                    'device': args.device, 'vit_like': args.vit_like}  # MM_DATA_MB321['ichr_test']
        args_key.update({'fnp_log': args.fnp_log})
        dict_vcam = init_args_dict('vis_cam', args_key, osp.basename(data_prefix))
        vis_cam(dict_vcam)

    return json_log, args.checkpoint # , cfg.data.test.data_prefix
# end eval_mm

def launch_mm_cls(): # ('_test', 'ver_fit', _launch_list, _f=data_name)
    args = parse_args()
    print(f"** vit_like={args.vit_like} for:{BB_ABN}")
    os.makedirs(args.work_dir, exist_ok=False) # TM is unique
    write_txt([args.fnp_log], f"-input_dir\t={DATA_SPLIT_ICHR}\n output_dir\t={args.work_dir}\n bb_name={args.bb_name}, abn_flag={args.abn_flag}")
    
    if EVAL_ALL:
        fp_pth, not_founds = args.fp_last, {}
        write_txt([args.fnp_log], f"** start multi eval from in:{fp_pth}\n")
        for i, r in enumerate([0.25]): # [0., 0.125, 0.25, 0.5]
            abn_str = f"abn{int(r*1000)}_"
            cls_str = f"cls{2 if LB_T21 else 24}_"
            write_txt([args.fnp_log], f"** NO.{i+1}: abn_str(set_data)={abn_str}, cls_str={cls_str}\n")
            r, _ = eval_mm(args, ['latest', abn_str, cls_str], set_data=abn_str)
            if type(r) is list and 'latest' in r:
                r.remove('latest')
                not_founds[args.bb_name] = r
        write_txt([args.fnp_log], f"done, {len(not_founds)} '.pth' not found:\n{not_founds}")
        return args.fnp_log, ''
    if not args.eval_only: # train + valid
        args = train_mm(args)
        args = pick_pth(args, 'mm_cls')
        pth_list = [args.checkpoint, args.last_pth]
        del_pth(args.work_dir, args.fp_pth_all, osp.basename(args.checkpoint), _epoch=args.epochs)
    else: # only valid
        write_txt([args.fnp_log], "*eval skip train, current checkpoint={}".format(args.checkpoint))
        if B_FIND_AND_PICK:
            args = find_and_pick(args)
            pth_list = [args.checkpoint, args.last_pth]
        else: pth_list = [None] # will pick from fp_last
    pth_list, _ = list321(pth_list), '!!NOT eval_mm if NO args.checkpoint or NOT found!'
    write_txt([args.fnp_log], f"--- pth_list={pth_list}")
    for args.checkpoint in pth_list: # args.checkpoint must be the first
        if args.checkpoint is not None: _, ck = eval_mm(args, 'load_without_find')
        elif not ONLY_SEG:
            # eval_mm(args, _find_in_last=None) # for original coco
            # if not QUICK: _, args.checkpoint = eval_mm(args, 'best')
            _, args.checkpoint = eval_mm(args, best_latest) # 'latest'
    if REPAIR: _, args.checkpoint = eval_mm(args, best_latest, REPAIR) #
    return args.fnp_log
    # return Cnn, _fnp_pkl_last, log_path_this, _fp_cls_in, _pred_dir_dict
# end launch_mm_cls

if __name__ == '__main__':
    print(f"start chr7_cls.py @ {time_str()}")
    _log = launch_mm_cls()
    if PLT_STRUCT: print_bb_struc(_log, None, _many=False)
    end_this_py([LOG_THIS_ABN], _tag=f"e{CLS_MAX_EPOCH}", _str=last_str_all) # chr7_cls.py
