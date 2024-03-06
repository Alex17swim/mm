# mm_vis.py, chengyu wang, 2022.05.10, XJTLU, last update on 2022.05.10
''' *: os  $$: gpu  ##: dataset  &: image  ---: model  --: loss  -: directory  '''
import sys, os
_import_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_import_root)

from pt_flag import PLT_CAM, PLT_21, QUICK, CAM_METHOD, RSTRIP, TEMP_ROOT, TM, write_txt, input3, not_in_demo_case, os_system3
# from pt_flag import DEL, MOVE, COPY, CP_FILE, check_fp_in, time_gap, make_dirs, input1
from pt_flag import FLAG_CHR7_CLS, MM_PY_CFG_FILE_ABN, MM_DATA_MB321, OUTPUT_THIS_ABN, FP_LAST_ABN, ABN_DATA_TYPE, BB_ABN, SUF_ABN_KEY, NUM_CLASS_ABN

from all_include.pt_head import find_sth_easy
from all_include.pt_utils import label_is_21st
import copy # argparse,
import os.path as osp
from PIL import Image

# vis_pipeline
import itertools
from mmcls.core import visualization as vis
from mmcv import Config, ProgressBar # DictAction, 
from pathlib import Path
from mmclassi.tools.visualizations.vis_pipeline import retrieve_data_cfg, build_dataset_pipelines, get_display_img
_skip_, _output_fp_vpipe_, _phase_, _number_, _mode_, _show_, _adaptive_, _min_, _max_, _bgr2rgb_, _window_ = ['ToTensor', 'Normalize', 'ImageToTensor', 'Collect'], 'set output vp later', 'test', 9, 'concat', True, False, 200, 800, True, '12*7'
_cfg_options_ = None # {'model.backbone.depth': 50, 'model.backbone.with_cp':True}
__show_options_ = {'wait_time':1}
# vis_cam
import pkg_resources
from mmcls import digit_version
from mmcls.apis import init_model
from mmclassi.tools.visualizations.vis_cam import apply_transforms, get_layer, get_default_traget_layers, build_reshape_transform, init_cam, show_cam_grad, METHOD_MAP, FORMAT_TRANSFORMS_SET
_img_, _checkpoint_, _layers_, _preview_, _method_, _category_, _eigen_, _aug_, _save_fnp_vcam_, _device, _vit_, _tokens_ = 'set img later', 'set checkpoint later', [], False, 'set method later', [], False, False, 'set path later', 'cpu', True, 2 # GradCAM, GradCAMPlusPlus,  if 'vit' in BB_ABN else False

# parser.add_argument('--mode', default=_mode_, type=str, choices=['original', 'transformed', 'concat', 'pipeline'])
# FORMAT_TRANSFORMS_SET = {'ToTensor', 'Normalize', 'ImageToTensor', 'Collect'}
# METHOD_MAP = ['gradcam', 'gradcam++', 'xgradcam', 'eigencam', 'eigengradcam', 'layercam']
   
assert (osp.join('0sync', 'PT') in os.getcwd()), input3(f"! you can't run this file due to 'all_include' not founded\n")
op_list = ['vis_pipeline', 'vis_cam']
op, key = op_list[1], 'key' # vis_cam, vis_pipeline    
class to_args():
    def __init__(self, d, _type='vis_cam'):
        assert type(d) is dict, f"! d must be dict\n"
        self.__dict__ = copy.copy(d)
        self.type = _type
        self.keys = d.keys()
    def has(self, k): return True if k in self.keys else False
def init_args_dict(_op, d2=None, _suf=TM):
    assert _op in op_list, f"! op({_op}, type={type(_op)}) should be str in:{op_list}"
    if d2 is None: d2 = {'checkpoint': None}
    elif not isinstance(d2, dict) or (len(d2.keys()) == 1 and 'checkpoint' in d2.keys()): print(f" skip init_args_dict due to d2={d2}"); return None
    is_debug, wait_str, fp_input = True, ' ', osp.join(TEMP_ROOT, 'test') # , 'abn'
    _config = MM_PY_CFG_FILE_ABN
    _img_ = osp.join(MM_DATA_MB321['ichr_test'], ABN_DATA_TYPE) if FLAG_CHR7_CLS else osp.join(fp_input, 'vis', 'val') # , '34__1b_roat_fit.jpg' # 
    _find_list = ['.pth', BB_ABN, SUF_ABN_KEY, f'cls{NUM_CLASS_ABN}']
    print(f"c d2={d2}, find_list={_find_list}")
    if TEMP_ROOT in _img_:
        print(f"-- vis_cam in 'temp' mode")
        _suf = _suf if TM in _suf else f"{_suf.rstrip('_')}_TM"
        _output_fp_vpipe_ = osp.join(fp_input, 'mm_output_{_suf}')
        _checkpoint_ = find_sth_easy(FP_LAST_ABN, _find_list) # osp.join(TEMP_ROOT, 'test', 'vis', 'mmc_ToyVit_cls2_debug.pth')
    else:
        print(f"-- vis_cam in 'chr7_cls' mode")
        _output_fp_vpipe_ = d2['save_path'] if 'save_path' in d2.keys() else osp.join(OUTPUT_THIS_ABN, 'mm_output_'+TM)
        _checkpoint_ = d2['checkpoint'] if d2['checkpoint'] else find_sth_easy(FP_LAST_ABN, _find_list + ['latest'], _confirm=False)
    print(f" input3:{_img_}\n output:{_output_fp_vpipe_}\n checkpoint:{_checkpoint_}")
    _save_fnp_vcam_ = _output_fp_vpipe_ # osp.join(_output_fp_vpipe_, 'saved.jpg')
    _method_ = CAM_METHOD # ['gradcam', 'gradcam++', 'xgradcam', 'eigencam', 'eigengradcam', 'layercam']
    _tokens_ = 1 if 'vit' in BB_ABN.lower() else 0
    dict_vpipe = {'skip_type':_skip_, 'output_dir':_output_fp_vpipe_, 'phase':_phase_, 'number':_number_, 'mode':_mode_, 'show':_show_, 'adaptive':_adaptive_, 'min_edge_length':_min_, 'max_edge_length':_max_, 'bgr2rgb':_bgr2rgb_, 'window_size':_window_, 'cfg_options':_cfg_options_}
    dict_vpipe.update({'config':_config, 'cfg_options':_cfg_options_, 'show_options':__show_options_})
    dict_vcam = {'img':_img_, 'checkpoint':_checkpoint_, 'target_layers':_layers_, 'preview_model':_preview_, 'method':_method_, 'target_category':_category_, 'eigen_smooth':_eigen_, 'aug_smooth':_aug_, 'save_path':_save_fnp_vcam_, 'device':_device, 'vit_like':_vit_, 'num_extra_tokens':_tokens_}
    dict_vcam.update({'config':_config, 'cfg_options':_cfg_options_, 'show_options':__show_options_})
    fp_input = osp.abspath(fp_input)
    # assert osp.exists(fp_input), f"!! NOT found:{fp_input}"
    if 'vis_pipeline' in _op: return dict_vpipe
    if 'vis_cam' in _op:
        if len(d2.keys()) > 1: dict_vcam.update(d2)
        return dict_vcam
# end init_args_dict
# args_vp = to_args(dict_vpipe)
# args_vcam = to_args(dict_vcam)

def vis_pipeline(dicts):
    args = to_args(dicts)
    wind_w, wind_h = args.window_size.split('*')
    wind_w, wind_h = int(wind_w), int(wind_h)  # showing windows size
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options, args.phase)

    dataset, pipelines = build_dataset_pipelines(cfg, args.phase)
    CLASSES = dataset.CLASSES
    display_number = min(args.number, len(dataset))
    progressBar = ProgressBar(display_number)

    with vis.ImshowInfosContextManager(fig_size=(wind_w, wind_h)) as manager:
        for i, item in enumerate(itertools.islice(dataset, display_number)):
            image = get_display_img(args, item, pipelines)

            # dist_path is None as default, means not saving pictures
            dist_path = None
            if args.output_dir:
                # some datasets don't have filenames, such as cifar
                src_path = item.get('filename', '{}.jpg'.format(i))
                dist_path = os.path.join(args.output_dir, Path(src_path).name)

            infos = dict(label=CLASSES[item['gt_label']])

            ret, _ = manager.put_img_infos(
                image,
                infos,
                font_size=20,
                out_file=dist_path,
                show=args.show,
                **args.show_options)

            progressBar.update()

            if ret == 1:
                print('\nMannualy interrupted.')
                break
# end vis_pipeline

def _vis_cam(dicts, _log=None, _ext='.jpg'):
    if dicts['method'].lower() not in METHOD_MAP.keys(): raise ValueError(f"invalid CAM type {dicts['method']}, supports {', '.join(list(METHOD_MAP.keys()))}.")
    args = to_args(dicts)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None: cfg.merge_from_dict(args.cfg_options)
    # build the model from a config file and a checkpoint file
    model = init_model(cfg, args.checkpoint, device=args.device)
    # if args.preview_model:
    #     print(model)
    #     print('\n Please remove `--preview-model` to get the CAM.')
    #     return
    # apply transform and perpare data
    fp_in = args.img if osp.isdir(args.img) else osp.dirname(args.img)
    fp_out = args.save_path if not args.save_path.endswith(('.txt', '.log', '.md')) else osp.dirname(args.save_path)
    fp_out = fp_out.rstrip(RSTRIP).rstrip('_')+f"_{args.method}"
    os.makedirs(fp_out, exist_ok=True)
    log_list = [osp.join(fp_out, f"cam_log_{TM}.txt"), _log]
    if 'fnp_log' in args.keys: log_list = log_list + [args.fnp_log]
    n_d, n_f, num = 0, 0, 0
    _str = 'choose the last norm layer before the final attention block as target_layer..' if args.num_extra_tokens >= 1 else 'choose the last norm layer as target_layer.'
    write_txt(log_list, f" vis_cam walk in:{fp_in}\n output:{fp_out}\n{_str}")
    for root, dirs, files in os.walk(fp_in): # for idx, img in enumerate(fns):
        for dir in dirs: n_d += 1
        for file in files:
            if (fp_in != args.img) and (not file == osp.basename(args.img)): write_txt(log_list, f"~ skip as NOT is an img:{osp.basename(args.img)}")
            if not file.endswith(_ext): write_txt(log_list, f"~ skip file:{file}"); continue
            _dir = '' if len(root) == len(fp_in) else root[len(fp_in):]
            if (_dir != '') and ('output' in _dir): write_txt(log_list, f"~ skip dir:{_dir}, file:{file}"); continue
            _dir = _dir.lstrip(RSTRIP)
            if PLT_21 and not_in_demo_case(file, osp.basename(root)):
                _is = label_is_21st(file)
                if 0 == _is or (0 > _is and n_f > PLT_CAM): continue
            fp_out_dict = {'cam': osp.join(fp_out, _dir + '_cam'),
                           'src': osp.join(fp_out, _dir + '_src'),
                           'cam0':osp.join(fp_out, _dir + '_cam0')}
            for k, v in fp_out_dict.items(): os.makedirs(v, exist_ok=True)
            # input1(f"root={root}, file={file}\n fp_out_dict['cam']={fp_out_dict['cam']}\n", '_vis_cam')
            _dst = osp.join(fp_out_dict['cam'], file)
            img = osp.join(root, file)
            n_f += 1
            if n_f <= 3 or n_f == len(files) or (n_f%20) == 0 or (n_f%len(files)) == 0: write_txt(log_list, f"-- {args.method} on No.[{n_f:>03d}/{len(files):>03d}] image:{file}")
            data, src_img = apply_transforms(img, cfg.data.test.pipeline)
            # build target layers
            if args.target_layers: target_layers = [get_layer(layer, model) for layer in args.target_layers]
            else: target_layers = get_default_traget_layers(model, args)
            # init a cam grad calculator
            use_cuda = ('cuda' in (args.device if type(args.device) is str else args.device.type))
            reshape_transform = build_reshape_transform(model, args)
            cam = init_cam(args.method, model, target_layers, use_cuda, reshape_transform)
            # warp the target_category with ClassifierOutputTarget in grad_cam>=1.3.7, to fix the bug in #654.
            targets = None
            if args.target_category:
                grad_cam_v = pkg_resources.get_distribution('grad_cam').version
                if digit_version(grad_cam_v) >= digit_version('1.3.7'):
                    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                    targets = [ClassifierOutputTarget(c) for c in args.target_category]
                else: targets = args.target_category
            # calculate cam grads and show|save the visualization image
            grayscale_cam = cam(data['img'].unsqueeze(0), targets, eigen_smooth=args.eigen_smooth, aug_smooth=args.aug_smooth)
            show_cam_grad(grayscale_cam, src_img, title=args.method, out_path=_dst) # args.save_path
            h, w, c = data['img_metas'].data['ori_shape']
            img_pil_wh = Image.open(_dst)
            pil_resize = img_pil_wh.resize((w, h), Image.BILINEAR)
            pil_resize.save(osp.join(fp_out_dict['cam0'], file))
            os_system3(img, osp.join(fp_out_dict['src'], file))
    if n_f < 1: input(f"?? _vis_cam files [{n_f}/{len(files)}], {len(dirs)}dirs, fp_in={fp_in}\n")
    write_txt(log_list, f"-- _vis_cam files [{n_f}/{len(files)}], {args.method} on last No.[{n_f:>03d}] image:{file}\n vis_cam walk done:{fp_out}")
# end _vis_cam
def vis_cam(dicts, _log=None, _ext='.jpg'):
    if dicts is None:
        write_txt(_log, '** vis_cam skipped due to dicts is None')
        return None
    if not QUICK:
        try: _vis_cam(dicts, _log, _ext)
        except Exception as e: print(f"!! vis_cam err:\n{e}")
        # finally: pass
    else: _vis_cam(dicts, _log, _ext)

if __name__ == '__main__':
    print(f"start mm_vis.py @ {TM}")
    if 'vis_pipeline' in op: vis_pipeline(init_args_dict(op))
    elif 'vis_cam' in op: vis_cam(init_args_dict(op))
    else: raise ValueError("??? mm_vis.py op={op}")
