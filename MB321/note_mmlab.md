*last update on 2023-1206-20:00*
sys.version (Python)    '3.8.10 (default, May 26 2023, 14:05:08) \n[GCC 9.4.0]'
mmcv.__version__        '2.0.1'
mmengine.__version__    '0.10.1'
mmpretrain.__version__  '1.1.1'
os.uname()  posix.uname_result(sysname='Linux', nodename='fm-2080', release='5.4.0-83-generic',
version='#93-Ubuntu SMP Tue Aug 17 10:15:03 UTC 2021', machine='x86_64')

# shortcut
item            Pycharm     VS Code
open project    Alt+1       Ctrl+Shift+E
latest files    Ctrl+E      Ctrl+E

# github
cd ~/1github/mmlab/
git clone git@github.com:open-mmlab/mmpretrain.git mmclss
git status
git branch mb321
git checkout mb321
git status
git pull origin main


# pre-install
## use vertualenv
cd ~/env
virtualenv -p /usr/bin/python3.8 mm231012
source mm231012/bin/activate
(mm231012)cd mm231012
(mm231012)ln -s ~/1github/mm/note_mmlab.md ~/env/mm231012/note_mmlab.md

## check and download the right version of torch, torchvision, mmcv
* check your cuda version by:
nvcc --version
Build cuda_11.3.r11.3/compiler.29920130_0
* check your python version by
python --version
Python 3.8.10
**so you need cu113 and cp3.8!!**
* download torch and torchvision from **https://download.pytorch.org/whl/**
into 'torch', and search with: -1.10.1+cu113-cp38-
into 'torchvision', and serach with: -0.11.2+cu113-cp38-
then find the linux or win you want
* download mmcv from **https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html**
**mmcv-2.0.0-cp38-cp38-manylinux1_x86_64.whl**
~~mmcv_full-1.7.1-cp38-cp38-manylinux1_x86_64.whl~~ for 2.0- version only
* download opencv_python from **https://pypi.org/simple/opencv-python/**
opencv_python-4.8.1.78-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

## pip install -U openmim
## pre-install for MMCV (MMDetection and MMSegmentation) 
don't use this online mode: 
pip install python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
**use the .whl file you downloaded before!!**
pip install ~/env/whl/911/torch-1.10.1+cu113-cp38-cp38-linux_x86_64.whl
pip install ~/env/whl/911/torchvision-0.11.2+cu113-cp38-cp38-linux_x86_64.whl
pip install ~/env/whl/911/opencv_python-4.8.1.78-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
*!!here use mim, NOT pip!!*
mim install ~/env/whl/911/mmcv-2.0.0-cp38-cp38-manylinux1_x86_64.whl 
*don't use online mode: mim install "mmcv>=2.0.0"*
* # '-v' means verbose, or more output; '-e' means installing a project in editable mode, 
thus any local modifications made to the code will take effect without reinstallation.

## MMEngine (mmengine)--mmengi
* https://github.com/open-mmlab/mmengine
~~mim install mmengine~~
(mm231012)cd ~/1github/mm
git clone https://github.com/open-mmlab/mmengine.git mmengi
(mm231012)cd mmengi
mim install -e . **!!!here is ' .' at the end of the command!**

# installation
## MMClassification/MMPreTrain (mmcls/mmpretrain)--mmclss
* https://github.com/open-mmlab/mmpretrain
* install MMPreTrain
(mm231012)(mm231012)cd ~/1github/mm
git clone https://github.com/open-mmlab/mmpretrain.git mmclss
cd mmclss
(mm231012) mim install -e . **!!!here is ' .' at the end of the command!**

## MMDetection (mmdet)--mmdetc
* https://github.com/open-mmlab/mmdetection
**Note: In MMCV-v2.x, mmcv-full is rename to mmcv, if you want to install mmcv without CUDA ops, 
  you can use mim install "mmcv-lite>=2.0.0rc1" to install the lite version**
* see '## pre-install for MMCV'
* install MMDetection
*Case a: If you develop and run mmdet directly, install it from source:*
(mm231012)cd ~/1github/mm
git clone https://github.com/open-mmlab/mmdetection.git mmdetc
(mm231012)cd mmdetc
(mm231012) pip install -v -e . **!!!here is ' .' at the end of the command!**
*Case b: If you use mmdet as a dependency or third-party package, install it with MIM:*
mim install mmdet

## MMSegmentation (mmseg)--mmsegm
* https://github.com/open-mmlab/mmsegmentation
* see '## pre-install for MMCV'
* install MMSegmentation
*Case a: If you develop and run mmseg directly, install it from source:*
(mm231012)cd ~/1github/mm
git clone -b main https://github.com/open-mmlab/mmsegmentation.git mmsegm
(mm231012)cd mmsegm
(mm231012) pip install -v -e . **!!!here is ' .' at the end of the command!**
*Case b: If you use mmsegmentation as a dependency or third-party package, install it with pip:*
pip install "mmsegmentation>=1.0.0"

## MMagic (mmagic)--mmai
* https://github.com/open-mmlab/mmagic
* see '## pre-install for MMCV'
* install MMagic
(mm231012) mim install mmagic
git clone https://github.com/open-mmlab/mmagic.git mmai
(mm231012)cd mmai
(mm231012) pip install -e . **!!!here is ' .' at the end of the command!**

# Q&A
## debug
### mim install mmengine: KeyError: 'pkg_resources'
*ImportError: cannot import name 'six' from 'pkg_resources.extern'...*
 pip install --upgrade pip
 mim install mmengine

### 'UniversalVisualizer is not in the mmengine::visualizer registry.
https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html
 
## ImageClassifier is not in the mmengine
A:\1github\mm\mmclss\docs\en\user_guides\config.md
*Ignore some fields in the base configs*
Sometimes, you need to set `_delete_=True` to ignore some domain content in the basic configuration file.
You can refer to the {external+mmengine:doc}`documentation in MMEngine <advanced_tutorials/config>` 
for more instructions.

The following is an example. If you want to use cosine schedule in the above ResNet50 case,
just using inheritance and directly modifying it will report `get unexpected keyword 'step'` error,
because the `'step'` field of the basic config in `param_scheduler` domain information is reserved,
and you need to add `_delete_ =True` to ignore the content of `param_scheduler` related fields in the basic
configuration file:
**param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, _delete_=True)**


### UniversalVisualizer is not in the mmengine::visualizer registry
* xx is not in the mmengine::visualizer registry
re-install mmpretrain: Successfully installed mmcv-2.0.1 mmpretrain-1.0.2


### ImportError: libcudart.so.10.2: cannot open shared object file: No such file or directory
https://blog.csdn.net/weixin_43881491/article/details/128828413
1. 输入命令： vim ~/.bashrc或者vi ~/.bashrc
2.在~/.bashrc末尾加上：
旧的写法：
export PATH="$PATH:/usr/local/cuda/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64/"
export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/cuda/lib64"
新的写法：
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-11.3/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-11.3
alias sudo='sudo env PATH=$PATH'
3. 使配置文件生效: 执行 source ~/.bashrc

### import torch.utils.tensorboard as tb ERROR: module 'distutils' has no attribute 'version'
A:\env\w_mm231107\Lib\site-packages\torch\utils\tensorboard\__init__.py
comment: LooseVersion = distutils.version.LooseVersion
comment: if not hasattr(tensorboard, '__version__') or
    LooseVersion(tensorboard.__version__) < LooseVersion('1.15'):
comment:     raise ImportError('TensorBoard logging requires TensorBoard version 1.15 or above')
comment: del LooseVersion

### DLL load failed while importing _ext: 找不到指定的模块。
**use ubuntu instead, guy, if you see such as:**
* from mmcv.ops import SyncBatchNorm
    File "A:\1github\mm\mmengi\mmengine\model\utils.py", line 174, in 
* from mmcv.ops.roi_align import roi_align
    File "A:\1github\mm\mmdetc\mmdet\structures\mask\structures.py", line 12


### __init__() got an unexpected keyword argument 'data_root'
* assert len(self.CLASSES) == len(classes)

### topk is str
cfg_options={'val_evaluator.topk':(1,3)}

### runner.logger.info
* Checkpoints will be saved to 'self.out_dir' or 'default_hooks {out_dir=xx}'
* CheckpointHook
* see '### training (display log, save checkpoints)': mmpretrain/configs/_base_/default_runtime.py --> line 3: 5 Hooks can be used

### __init__() got an unexpected keyword argument 'out_results_path'
runner.test_evaluator.metrics.append(DumpDetResults(out_results_path=args.out))



# **Coding, replace mb321 into your own files.**
## GPU (#*mb321*)
* mm/mmengi/mmengine/device/utils.py --> line 89: DEVICE = f'cuda:{gpu_str}'
* mm/mmengi/mmengine/runner/amp.py -->  device_type == 'cuda' --> *2 palces*: 'cuda' in device_type
**env/mm231012/lib/python3.8/site-packages/torch/autocast_mode.py** --> *3 palces*: self.device == 'cuda' --> 'cuda' in self.device
* mm/mmengi/mmengine/_strategy/single_device.py -- class SingleDeviceStrategy(BaseStrategy) --> line 103: Distributed training is not used
* mm/mmengi/mmengine/_strategy/distributed.py -- class DDPStrategy(SingleDeviceStrategy) --> line 89: model = model.to(get_device())


## log
- mm/mmengi/mmengine/runner/log_processor.py --> Epoch({mode}
- mm/mmengi/mmengine/logging/logger.py --> line 327: print_log
* log_file = agLOG.val_seg # set this in "mm/mmengi/mmengine/runner --> line 401:self.build_logger"
* log_level = 'INFO' # choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
  
- mm/mmengi/mmengine/runner/runner.py --> line 2382: self.logger.info()
from MB321.mm_flag import write_txt, agLOG, TM, SHOW_MANY
_txt = f"\n {xx_log}\n" # self.cfg.pretty_text
write_txt([agLOG.bug_all], f"{TM}\n{_txt}", b_prt=SHOW_MANY)

## metric
/mm/mmengi/mmengine/runner/loops.py --> line 438: metrics = self.evaluator.evaluate(len(self.dataloader.dataset))


## detection
**assume the root is: xx/mm/mmdetc/**
### config file
* configs/_mb321/mb321_base.py: a base for other _mb321 config files
* configs/_mb321/toy_mrcnn.py:
  the config file of 'agMM.cfg_seg=join(agBASE.root_code, 'mmdetc/configs', MM_CFG_PY[_ks])',
  you can config by reset the dicts for model, dataset, schedule (optimizer), etc.
* the pretrained .pth is 'agMM.pre_pth_seg=join(agBASE.root_data, 'trained', MM_PRE_PTH[_ks])'

### dataset
* configs/_base_/datasets/_mb321_coco_instance.py: your dataset config file for dataroot, batch size, etc. 
* mmdet/datasets/_mb321_coco.py: your dataset file for classes, data_info, etc.
* mmdet/datasets/__init__.py: from ._mb321_coco import MB321coco
  
* train_dataloader, val_dataloader, test_dataloader
* val_evaluator; test_evaluator: classwise=True
* train_pipeline, test_pipeline

### model
* configs/xx/xx.py: if you like or for multipul versions; NOT recommanded
* mmdet/models/*everything*
* mmdet/models/__init__.py: from xx import *

- type (**!!model type, NOT backbone type!!**)
* mmdet/models/detectors/__init__.py: RPN, FastRCNN, MaskRCNN, MaskScoringRCNN, etc.
- backbone
* type @: mmdet/models/backbones/__init__.py: SwinTransformer, ResNet, ResNeXt, etc.
- neck (can be several dicts)
* type @: mmdet/models/necks/__init__.py: FPN, RFP, etc. 
- roi_head
* type @: mmdet/models/roi_heads/__init__.py: BaseRoIHead, MaskIoUHead, StandardRoIHead, etc.
* bbox_heads @ mmdet/models/roi_heads/bbox_heads/__init__.py: BBoxHead, etc.
* mask_head @ mmdet/models/roi_heads/mask_heads/__init__.py:MaskIoUHead, etc.
* **self-defined head**: mmdet/models/roi_heads/mask_scoring_roi_head.py, also defined in roi_head
- train_cfg, test_cfg

### visulize and **calculate pixels of an instance**
* mmdet/visualization/__init__.py: 'palette_val', 'get_palette', 'DetLocalVisualizer', 
  'jitter_color', 'TrackLocalVisualizer'
*  visualizer.add_datasample
* **mmdet/visualization/local_visualizer.py** --> mmcv.imwrite; label_text; polygons; _draw_instances()
            for _n in range(len(polygons)):
                sk.append(len(polygons[_n]))
            print(f"sk={sk}")

### evaluation
* mmdet/evaluation/functional/ytvis.py --> line 59: loading annotations into memory...
* mmdet/evaluation/metrics/coco_metric.py: CocoMetric

### test without annotation files
* test_pipeline = [
    # If you don't have a gt annotation, delete the _following_ pipeline
    # dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
*   1> make sure you installed labelme (pip install labelme)
    2> copy the test images into 'img' folder;
    3> copy following 4 files beside 'img': 'coco_prepare.py', 'json2coco.py', '''fake_json.json', and 'lb.txt'
    4> **python coco_prepare.py** and you will get 'img_coco', then copy 'img_coco' to your place.
  
## classification
**assume the root is: xx/mm/mmclss/**
### training (display log, save checkpoints)
* mmpretrain/configs/_base_/default_runtime.py --> line 3: 5 Hooks can be used
* mmpretrain/configs/_base_/default_runtime.py --> line 13: IterTimerHook --> CheckpointHook, DistSamplerSeedHook, 

### visulize
* mmpretrain/visualization/visualizer.py --> line :

* mmpretrain/datasets/__init__.py
* mmpretrain/datasets/_mb321_ichr_base.py
### evaluation
* mmpretrain/evaluation/metrics/__init__.py
* mmpretrain/evaluation/metrics/single_label.py


# 
*  module 'PIL.Image' has no attribute 'ANTIALIAS'
* pip show Pillow
1> 10.0 -> 9.5
2> Image.ANTIALIAS -- > Image.Resampling.BILINEAR

* assert self._parent_pid == os.getpid(), 'can only test a child process'
> pnum_workers=0

* UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument
* check the version of a package xxx: pip show xxx