# util321.py, Chengyu Wang, 2023-0529, last update on 2023-1102
from MB321.mm_flag import *
import torch
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TKagg') # for UserWarning: Matplotlib is currently using agg

PRINT_DECIMAL = 4
def np_mean(v, _decimal=PRINT_DECIMAL):
    v = np.array(v)
    return np.round(np.mean(v), _decimal)
def np_max(v, _decimal=PRINT_DECIMAL):
    v = np.array(v)
    return np.round(np.max(v), _decimal)
def np_min(v, _decimal=PRINT_DECIMAL):
    v = np.array(v)
    return np.round(np.min(v), _decimal)
def np_div(v1, v2, _decimal=PRINT_DECIMAL):
    v1, v2 = np.array(v1), np.array(v2)
    return np.round(v1 / v2, _decimal)

# gpu = int(USE_GPU_LIST[0])
# os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpu}"
# # torch.cuda.set_device(gpu)  # 1
# dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
#
# # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # dev = torch.device('cuda', 1)
# # x = torch.randn(1, device=dev)
# # y = x.to(dev)
# gpu_all, gpu_cur, gpu_num = torch.cuda.device_count(), torch.cuda.current_device(), USE_GPU_LIST
def set_gpu(gpu_str=agGPU.gpu_str, log=None):
    _t0 = datetime.datetime.now() # print("start set_gpu() at {}".format(_t0))
    gpu_all = torch.cuda.device_count()
    gpu_num = len(gpu_str)
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str # gpu_str = f"{int(USE_GPU_LIST[0])}"
    # print("list={}, str={}".format(_gpu_list, gpu_list_str))
    if gpu_str == '-1':
        device = 'cpu'; use_cpu = True
    else:
        device = torch.device(f'cuda:{gpu_str}' if torch.cuda.is_available() else 'cpu'); use_cpu = False

    # print("use={}, count={}".format(use_cpu, count))
    _pre = "$$" if gpu_all > 0 else "!!$$"
    if not use_cpu and gpu_all < 1:
        input3(f"!!$$ failed to use gpu: {gpu_str}\n")
    else:
        if gpu_num > gpu_all: print("!!$$ tring to set {}gpus as maximum {}".format(gpu_num, gpu_all))
        write_txt([log], f"$$ current:{torch.cuda.current_device()} (suppose {gpu_str}), {_pre}using [{gpu_num}/{gpu_all}] gpus, device='{device}', cost {time_gap(_t0)}")
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # if gpu_num == 1:
    return device, gpu_num, gpu_all
# end set_gpu

# to_tensor_chw = T.Compose([T.ToTensor()])
# torch.from_numpy(np.array(img_pil_wh)).float()
def tensor_to_xx(_input_tensor):
    if type(_input_tensor) is dict:
        _keys = _input_tensor.keys()
        _np_dict, _arr_dict = dict.fromkeys(_keys), dict.fromkeys(_keys)
        for k in _keys:
            _np_dict[k] = np.round(_input_tensor[k].item(), PRINT_DECIMAL)
            _arr_dict[k] = np.asarray(_np_dict[k])
        return _np_dict, _arr_dict
    elif type(_input_tensor) is torch.Tensor:
        _np = np.round(_input_tensor.item(), PRINT_DECIMAL)
        _arr = np.asarray(_np)
        return _np, _arr
    else:
        raise ValueError("!found type(_input_tensor)={}".format(type(_input_tensor)))
# end tensor_to_xx

def set_dicts(_keys, _init=None):
    # print("set dict with _keys={}, _init={}, type(_init) is: {}".format(_keys, _init, type(_init)))
    _dicts = {}
    if type(_keys) is str:
        _keys = [_keys]
    elif type(_keys) is list:
        assert [type(v) is str for v in _keys], "! {}".format(type(_keys))
    elif type(_keys) is dict:
        if _init is None:  # just copy all itmes()
            for k, v in _keys.items(): _dicts[k] = v
            return _dicts
        else:
            _keys = list(_keys.keys())

    else:
        raise ValueError("! pectect type(_keys) in ['str', 'list', 'dict'], found:{}".format(type(_keys)))

    if type(_init) is type:
        raise ValueError("!you may use 'list()' or 'dict()' instead of 'list' or 'dict'")
    elif type(_init) is list:
        if len(_init) == len(_keys):
            [_dicts.setdefault(k, _init[i]) for i, k in enumerate(_keys)]
        elif len(_init) == 0:
            [_dicts.setdefault(k, []) for k in _keys]  # "!! list() (or []) MUST be set directly instead of via '_init'"
        else:
            raise ValueError("! '_init' is: {}, '_keys' is: {}".format(_init, _keys))
    else:
        [_dicts.setdefault(k, _init) for k in _keys]

    _d1_err, _d2_err = dict.fromkeys(_keys), dict.fromkeys(_keys)
    for k in _dicts:
        _d1_err[k] = _init  # !!! this will induce error, like each k will append same value at a time
        _d2_err.setdefault(k, _init)  # !!! this will generate all initial value with 'None'
    # print("dict_good={}\n  _d1_err={}\n  _d2_err={}".format(_dicts, _d1_err, _d2_err))
    return _dicts
# end set_dicts

def find_sth_easy(_fp, _key_list, _prt=True, dKey=None, ext2='.pth', _confirm=True, _low=False): # _fnp_log,
    assert osp.exists(_fp), _fp
    if _low: print(f"!! will convert to lowercase in this finding\n")
    if _key_list is None:
        input("using default _key_list?\n")
        _key_list = ['.pth',]
    if type(_key_list) is str: # and '.' in _key_list[0]
        _key_list = [_key_list]
    _key_list = list321(_key_list, remove_none=True, _low=_low)
    print("finding:{} in:{}".format(_key_list, _fp))
    _fns, _dirs, _files, _ld, _lf, _ldf = get_fns(_fp)
    if _ld > 0 and _lf > 0:
        if_print("! only find in {} files, ignore {} dirs".format(_lf, _ld), _prt)
    _found = []; fn2=None
    if dKey is None: # only find in files
        dir_or_file = _files
    else: # first find in dirs, then in files
        dir_or_file = _dirs
        if type(dKey) is not str: dKey = str(dKey) # dKey = 'latest', (epoch_) '30', etc.
    for i, fn in enumerate(dir_or_file):
        _f, fn_low = True, fn.lower() if _low else fn
        for n2 in _key_list: # not including keys in _key_list
            if n2 not in fn_low: _f = False; break
        if dKey is not None and _f: # into the dir to find files with dKey
            _fns2, _dirs2, _files2, _ld2, _lf2, _ldf2 = get_fns(osp.join(_fp, fn))
            if _lf2 > 0:
                for fn2 in _files2:
                    fn_low2 = fn2.lower() if _low else fn2
                    if dKey in fn_low2 and fn2.endswith(ext2): _f = True; break
                    else: _f = False
        if _f:
            if dKey is None:
                if_print("found No.{}, name={}".format(len(_found)+1, fn), _prt)
                _found.append(fn)
            else:
                if_print(f"found dKey='{dKey}' with '{ext2}' file in dir={fn}:{fn2}", _prt)
                return osp.join(_fp, fn, fn2)
    _num = len(_found)
    _n = f"\n !! this is QUICK mode, check output2={agDIR.root_out}" if QUICK else ''
    if _num < 1:
        if _confirm: input(f"... NOT found {_key_list} and dKey={dKey} in {_fp}{_n}\n press any key to continue\n")
        return None
    elif _num > 1:
        _names = [" {}: {}".format(i + 1, osp.basename(n)) for i, n in enumerate(_found)]
        r = input("--found {} '{}' files, select (1 to {}): {}\n".format(_num, _key_list[0], _num, _names))
        r = _found[int(r)-1]
    else: r = _found[0]
    if_print("easy found: {}".format(r), _prt)
    return osp.join(_fp, r)
# end find_sth_easy

def plt_subplot(hwc, row, col, pos, xlb='', ylb=''):
    print(f"r={row}, col={col}, pos={pos}")
    plt.subplot(row, col, pos)
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(hwc)

def plt_dataset(ds, _tag=''):
    l, col_num, i = len(ds), 3, 0
    print(f"ploting {l} images of {_tag}")
    os.makedirs(osp.join(agDIR.out_cls, 'plt'), exist_ok=True)
    fnp = osp.join(agDIR.out_cls, 'plt', f"{_tag}_NO._{i}.jpg")

    fig = plt.figure() # fig, ax = plt.subplots()
    for p in [0, 1, int(l/2), int(l/2)+1, l-2, l-1]:
        img, lb = ds[p]
        hwc = img.permute(1,2,0).numpy()
        row, col, pos, xlb, ylb = int(i/col_num)+1, (i%col_num)+1, (i%col_num)+1, lb, f'No.{i+1}'
        # plt_subplot(hwc, row, col, pos, )
        print(f"r={row}, col={col}, pos={pos}")
        plt.subplot(row, col, pos)
        plt.xlabel(xlb)
        plt.ylabel(ylb)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(hwc)
        i += 1
    # plt.subplots_adjust
    # plt.margins(0,0)
    plt.savefig(fnp)
    # plt.show()
    plt.close()
    plt.clf() # plt.gca()
    print(f" util321 done, saved at {fnp}")