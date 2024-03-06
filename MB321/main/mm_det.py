''' mm_det.py, Chengyu Wang, XJTLU, 2023-1209, last update on 2023-1209
*: os  $$: gpu  ##: dataset  &: image  ---: model  --: loss, etc.  -: directory '''
import sys, os
_import_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_import_root)
# print("mm_det.py _import_root={}".format(_import_root))
# from MB321.mm_flag import *
from MB321.base.util321 import *

# Copyright (c) OpenMMLab. All rights reserved.
# import argparse
# import os
# import os.path as osp
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo

from mmengine import ConfigDict
from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from copy import deepcopy
import warnings

if agNET.bb_resume_seg:
        r = 'latest' # input(f"input '.pth' key to continue training:\n")
        pth_resume = find_sth_easy(agDIR.last_seg, [agTAG.bb_seg], dKey=r, ext2='.pth')
else: pth_resume = None
write_txt(agLOG.seg, f'pth_resume={pth_resume}')

cfg_options = {'log_file': agLOG.val_seg}
checkpoint = None # set in train_mm_det, or eval_mm_det
out_results = None # osp.join(agDIR.out_seg, f"{agTAG.log_seg}_out.pkl")
out_item = None if out_results is None else 'metrics' # ['metrics', 'pred']
show_dir = osp.join(agDIR.out_seg, 'vis')
b_show_pred, vis_interval, wait_time, tta = agMM.show_seg, 1, 1, agMM.tta

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--log', default=agLOG, help="logs")
    parser.add_argument('--log_file', default=agLOG.val_seg, help="logs")
    # parser.add_argument('--device', default='cuda:1', help='GPUs') # agGPU.dev
    parser.add_argument('--config', default=agMM.cfg_seg, help='train config file path')
    parser.add_argument('--work_dir', default=agDIR.out_seg, help='the dir to save logs and models') # work-dir
    parser.add_argument('--amp', action='store_true', default=False, help='enable automatic-mixed-precision training')
    parser.add_argument('--auto_scale_lr', default=agMM.auto_lr, action='store_true', help='enable automatically scaling LR.')
    parser.add_argument('--resume', default=pth_resume, help='If specify checkpoint path, resume from it, while if not specify, try to auto resume from the latest checkpoint in the work directory.')
    parser.add_argument('--cfg_options', nargs='+', default=cfg_options, help='override some settings in the used config,') # cfg-options, default=DictAction
    ### for validation / test
    parser.add_argument('--checkpoint', default=checkpoint, help='checkpoint file')
    parser.add_argument('--out', default=out_results, help='the file to output results.')
    parser.add_argument('--out_item', default=out_item, choices=['metrics', 'pred'], help='To output whether metrics or predictions. Defaults to output predictions.') # out-item    
    parser.add_argument('--show_dir', default=show_dir, help='directory where the visualization images will be saved.') # show-dir
    parser.add_argument('--show', default=b_show_pred, help='whether to display the prediction results in a window.') # action='store_true',
    parser.add_argument('--interval', default=vis_interval, type=int, help='SEG_IMG_SHOW; visualize per interval samples.')
    parser.add_argument('--wait_time', default=wait_time, type=float, help='SEG_IMG_SHOW; display time of every window. (second)') # wait-time
    parser.add_argument('--tta', default=tta, action='store_true', help='Whether to enable the Test-Time-Aug (TTA). If the config file ')
    
    parser.add_argument('--launcher', default='none', choices=['none', 'pytorch', 'slurm', 'mpi'], help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch will pass the `--local-rank` parameter to `tools/train.py` instead of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ: os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
# end parse_args

assert osp.exists(agMM.cfg_seg), f"!! not found config file:{agMM.cfg_seg}\n\n"

def merge_args(args, op='train'): # Merge CLI arguments to config.
    print(f" *merge_args() with op={op}\n")
    cfg = Config.fromfile(args.config) # load config
    cfg.launcher = args.launcher
    cfg.op = op
    if args.cfg_options is not None: cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None: # use config filename as default work_dir if cfg.work_dir is None
        input(f" ! work_dir is None, set to ./work_dirs\n\n")        
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.amp is True: # enable automatic-mixed-precision training
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic' # cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
        # cfg.test_cfg.fp16 = True # val

    if op == 'train': # resume training    
        if args.resume == 'auto':
            cfg.resume = True
            cfg.load_from = None
        elif args.resume is not None:
            cfg.resume = True
            cfg.load_from = args.resume
        if args.auto_scale_lr: # enable auto scale learning rate
            if 'auto_scale_lr' in cfg and 'enable' in cfg.auto_scale_lr and 'base_batch_size' in cfg.auto_scale_lr: cfg.auto_scale_lr.enable = True
            else: raise RuntimeError('Can not find "auto_scale_lr" or  "auto_scale_lr.enable" or  "auto_scale_lr.base_batch_size" in your configuration file.')
    else:
        cfg.load_from = args.checkpoint
        if args.show or args.show_dir: cfg = trigger_visualization_hook(cfg, args)
        if args.tta:
            if 'tta_model' not in cfg:
                warnings.warn('Cannot find ``tta_model`` in config, we will set it as default.')
                cfg.tta_model = dict(type='DetTTAModel', tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
            if 'tta_pipeline' not in cfg:
                warnings.warn('Cannot find ``tta_pipeline`` in config, we will set it as default.')
                test_data_cfg = cfg.test_dataloader.dataset
                while 'dataset' in test_data_cfg: test_data_cfg = test_data_cfg['dataset']
                cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
                flip_tta = dict(type='TestTimeAug', transforms=[
                        [dict(type='RandomFlip', prob=1.),dict(type='RandomFlip', prob=0.)],
                        [dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction'))],])
                cfg.tta_pipeline[-1] = flip_tta
            cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
            cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    # end if op != 'train': # val
    print(f"cfg.load_from={cfg.load_from}")
    return cfg
# end merge_args

def train_mm_det(args):
    t1 = time.time()
    cfg = merge_args(args, 'train')
    
    # build the runner from config    
    if 'runner_type' not in cfg: runner = Runner.from_cfg(cfg) # build the default runner
    else: runner = RUNNERS.build(cfg) # build customized runner from the registry if 'runner_type' is set in the cfg
    
    runner.train() # start training
    src = args.checkpoint = osp.join(args.work_dir, f'epoch_{agDATA.epoch_seg}.pth') # latest
    dst = osp.join(agDIR.last_seg, f"{agTAG.bb_seg}{agTAG.bug_seg}_acc={'x'}e={agDATA.epoch_seg}.pth")
    os.makedirs(osp.dirname(dst), exist_ok=True)
    os_system3(src, dst, CP_FILE)
    write_txt([agLOG.tra_seg], f" train_mm_det[quick={QUICK}] end at{time_str()}, cost{time_gap(t1)}")
    return args
# end train_mm_det

def eval_mm_det(args):
    t1 = time.time()
    cfg = merge_args(args, 'val') # build the runner from config

    # build the runner from config
    if 'runner_type' not in cfg: runner = Runner.from_cfg(cfg)  # build the default runner
    else: runner = RUNNERS.build(cfg)  # build customized runner from the registry if 'runner_type' is set in the cfg
    
    if args.out is not None: # add `DumpResults` dummy metric
        assert args.out.endswith(('.pkl', '.pickle')), 'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(DumpDetResults(out_results_path=args.out))

    runner.test() # runner.val()
    write_txt([agLOG.val_seg], f" eval_mm_det[quick={QUICK}] end at{time_str()}, cost{time_gap(t1)}")
# end eval_mm_det

def launch_mm_det():
    args = parse_args()
    if args.out is None and args.out_item is not None: raise ValueError('Please use `--out` argument to specify the path of the output file before using `--out-item`.')
    setup_cache_size_limit_of_dynamo() # Reduce the number of repeated compilations and improve training speed.

    if agNET.bb_train_seg:
        args = train_mm_det(args) # !! update args.checkpoint
    else:
        write_txt(agLOG.val_seg, f" skip training and find {agTAG.find_seg}.pth in {agDIR.last_seg}")
        args.checkpoint = find_sth_easy(agDIR.last_seg, ['.pth', agTAG.bb_seg , agTAG.bug_seg]) #
    eval_mm_det(args)
    write_txt(agLOG.seg, f" launch_mm_det[quick={QUICK}] end at{time_str()}, cost{time_gap()}")
# end launch_mm_det

if __name__ == '__main__':
    print(f" start mm_det.py (quick={QUICK}) @ {time_str()}")
    launch_mm_det()
    # print(f" mmdet.py done at {time_str()}, cost{time_gap()}")
'''
* label category
    ignore_index = sem_seg.metainfo.get('ignore_index', 255)
* 
'''