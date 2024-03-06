''' mm_cls.py, Chengyu Wang, XJTLU, 2023-1107, last update on 2023-1119
*: os  $$: gpu  ##: dataset  &: image  ---: model  --: loss, etc.  -: directory '''
import sys, os
_import_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_import_root)
# print("mm_cls.py _import_root={}".format(_import_root))
# from MB321.mm_flag import *
from MB321.base.util321 import *

# Copyright (c) OpenMMLab. All rights reserved.
# import argparse
# import os
# import os.path as osp

from mmengine.config import Config, ConfigDict # , DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmengine.evaluator import DumpResults
from copy import deepcopy
import mmengine # .dump as dump


if agNET.bb_resume_cls:
        r = 'latest' # input(f"input '.pth' key to continue training:\n")
        pth_resume = find_sth_easy(agDIR.last_cls, [agTAG.bb_cls], dKey=r, ext2='.pth')
else: pth_resume = None
write_txt(agLOG.cls, f'pth_resume={pth_resume}')

# data.train.pipeline.1.flip_prob=0.0
cfg_options = None # {'val_evaluator.topk':(1,agDATA.topk)} # f'val_evaluator.topk="(1,3)"' # cfg-options
checkpoint = None # set in train_mm_cls, or eval_mm_cls
out_item = 'metrics' # ['metrics', 'pred']
out_file =  osp.join(agDIR.out_cls, f"{agTAG.log_cls}_out.json")
show_dir = osp.join(agDIR.out_cls, 'vis')
b_show_pred, vis_interval, wait_time, tta = agMM.show_cls, 1, 1, agMM.tta

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    # parser.add_argument('--device', default=agGPU.dev, help='GPUs')
    parser.add_argument('--config', default=agMM.cfg_cls, help='train/test config file path')
    parser.add_argument('--work_dir', default=agDIR.out_cls, help='the dir to save logs and models') # work-dir
    parser.add_argument('--resume', default=pth_resume, help='resume from the latest checkpoint.')
    parser.add_argument('--amp', default=agMM.amp, action='store_true', help='enable automatic-mixed-precision training / test')
    parser.add_argument('--no_validate', default=False, help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--auto_scale_lr', default=agMM.auto_lr, action='store_true', help='whether to auto scale the learning rate according to the actual batch size and the original batch size.')
    parser.add_argument('--no_pin_memory', default= not agGPU.pin, action='store_true', help='whether to disable the pin_memory option in dataloaders.') # no-pin-memory
    parser.add_argument('--no_persistent_workers', default=agMM.no_pw, action='store_true', help='whether to disable the persistent_workers option in dataloaders.') # no-persistent-workers
    parser.add_argument('--cfg_options', nargs='+', default=cfg_options,help='') # cfg-options ,action=DictAction,
    ### for validation / test
    parser.add_argument('--checkpoint', default=checkpoint, help='checkpoint file')
    parser.add_argument('--out', default=out_file, help='the file to output results.')
    parser.add_argument('--out_item', default=out_item, choices=['metrics', 'pred'], help='To output whether metrics or predictions. Defaults to output predictions.') # out-item    
    parser.add_argument('--show_dir', default=show_dir, help='directory where the visualization images will be saved.') # show-dir
    parser.add_argument('--show', default=b_show_pred, help='whether to display the prediction results in a window.') # action='store_true',
    parser.add_argument('--interval', default=vis_interval, type=int, help='visualize per interval samples.')
    parser.add_argument('--wait_time', default=wait_time, type=float, help='display time of every window. (second)') # wait-time    
    parser.add_argument('--tta', default=tta, action='store_true', help='Whether to enable the Test-Time-Aug (TTA). If the config file ')
    
    parser.add_argument('--launcher', default='none', choices=['none', 'pytorch', 'slurm', 'mpi'], help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch will pass the `--local-rank` parameter to `tools/train.py` instead of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
# end parse_args

assert osp.exists(agMM.cfg_cls), f"!! not found config file:{agMM.cfg_cls}\n\n"

def merge_args(args, op='train'): # Merge CLI arguments to config.
    print(f" *merge_args() with op={op}\n")
    cfg = Config.fromfile(args.config) # load config
    cfg.launcher = args.launcher
    cfg.op = op

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None: # use config filename as default work_dir if cfg.work_dir is None
        input(f" ! work_dir is None, set to ./work_dirs\n\n")        
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.amp is True: # enable automatic-mixed-precision training
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
        cfg.test_cfg.fp16 = True # val

    if op == 'train': # resume training
        if args.no_validate: cfg.val_cfg, cfg.val_dataloader, cfg.val_evaluator = None, None, None
        if args.resume == 'auto':
            cfg.resume = True
            cfg.load_from = None
        elif args.resume is not None:
            cfg.resume = True
            cfg.load_from = args.resume
        if args.auto_scale_lr: cfg.auto_scale_lr.enable = True # enable auto scale learning rate
    else: # op != 'train': # == val
        cfg.load_from = args.checkpoint
        # -------------------- visualization --------------------
        if args.show or (args.show_dir is not None):
            assert 'visualization' in cfg.default_hooks, 'VisualizationHook is not set in the `default_hooks` field of config. Please set `visualization=dict(type="VisualizationHook")`'

            cfg.default_hooks.visualization.enable = True
            cfg.default_hooks.visualization.show = args.show
            cfg.default_hooks.visualization.wait_time = args.wait_time
            cfg.default_hooks.visualization.out_dir = args.show_dir
            cfg.default_hooks.visualization.interval = args.interval

        # -------------------- TTA related args --------------------
        if args.tta:
            if 'tta_model' not in cfg: cfg.tta_model = dict(type='mmpretrain.AverageClsScoreTTA')
            if 'tta_pipeline' not in cfg:
                test_pipeline = cfg.test_dataloader.dataset.pipeline
                cfg.tta_pipeline = deepcopy(test_pipeline)
                flip_tta = dict(
                    type='TestTimeAug',
                    transforms=[
                        [
                            dict(type='RandomFlip', prob=1.),
                            dict(type='RandomFlip', prob=0.)
                        ],
                        [test_pipeline[-1]],
                    ])
                cfg.tta_pipeline[-1] = flip_tta
            cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
            cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    # end else op != 'train': # val

    # ----------------- Default dataloader args ----------------- # set dataloader args
    default_dataloader_cfg = ConfigDict(pin_memory=True, persistent_workers=True, collate_fn=dict(type='default_collate'),)
    if digit_version(TORCH_VERSION) < digit_version('1.8.0'): default_dataloader_cfg.persistent_workers = False

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None: return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory: cfg[field]['pin_memory'] = False
        if args.no_persistent_workers: cfg[field]['persistent_workers'] = False

    set_default_dataloader_cfg(cfg, 'train_dataloader')
    set_default_dataloader_cfg(cfg, 'val_dataloader')
    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None: cfg.merge_from_dict(args.cfg_options)
    return cfg
# end merge_args

def train_mm_cls(args):
    t1 = time.time()
    cfg = merge_args(args, 'train')
    # build the runner from config
    if 'runner_type' not in cfg: runner = Runner.from_cfg(cfg) # build the default runner
    else: runner = RUNNERS.build(cfg) # build customized runner from the registry if 'runner_type' is set in the cfg
    
    runner.train() # start training
    _fp_pth = agDIR.out_pth_cls # NOT args.work_dir
    src = args.checkpoint = osp.join(_fp_pth, f'epoch_{agDATA.epoch_cls}.pth')  # latest
    dst = osp.join(agDIR.last_cls, f"{agTAG.bb_cls}{agTAG.bug_cls}_acc={'x'}e={agDATA.epoch_cls}.pth")
    os.makedirs(osp.dirname(dst), exist_ok=True)
    os_system3(src, dst, CP_FILE)
    write_txt([agLOG.tra_cls], f" train_mm_cls[quick={QUICK}] end at{time_str()}, cost{time_gap(t1)}")
    return args
# end train_mm_cls

def eval_mm_cls(args):
    t1 = time.time()
    cfg = merge_args(args, 'val')

    # build the runner from config
    if 'runner_type' not in cfg: runner = Runner.from_cfg(cfg) # build the default runner
    else: runner = RUNNERS.build(cfg) # build customized runner from the registry if 'runner_type' is set in the cfg

    if args.out and args.out_item in ['pred', None]:
        runner.test_evaluator.metrics.append(DumpResults(out_file_path=args.out))

    # start testing
    metrics = runner.test()

    if args.out and args.out_item == 'metrics': mmengine.dump(metrics, args.out)
    write_txt([agLOG.val_cls], f" eval_mm_cls[] end at{time_str()}, cost{time_gap(t1)}")
# end eval_mm_cls
def launch_mm_cls():
    t1 = time.time()
    # dev = f"cuda:{agGPU.gpu_str}"
    # set_gpu(USE_GPU_LIST)
    args = parse_args()
    if args.out is None and args.out_item is not None: raise ValueError('Please use `--out` argument to specify the path of the output file before using `--out-item`.')
    
    if agNET.bb_train_cls:
        args = train_mm_cls(args) # !! update args.checkpoint
    else:
        print(f" skip training, find the trained .pth with {agNET.bb_cls}({agTAG.bb_cls})")
        _find = agTAG.bb_cls.replace('toy_', '') # if 'toy_' in agTAG.bb_cls else agTAG.bb_cls
        args.checkpoint = find_sth_easy(agDIR.last_cls, ['.pth', _find], _low=True)
    eval_mm_cls(args)
    write_txt(agLOG.cls, f" launch_mm_cls[] end at{time_str()}, cost{time_gap(t1)}")
# end launch_mm_cls

# end launch_mm_cls
if __name__ == '__main__':
    print(f"start mm_cls.py @ {time_str()}")
    launch_mm_cls()
