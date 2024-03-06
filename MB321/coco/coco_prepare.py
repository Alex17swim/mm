# coco_prepare.py, Chengyu Wang, last update on 2023.12.23
''' *: os  $$: gpu  ##: dataset  &: image  ---: model  --: loss  -: directory  '''
import sys, os
_import_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_import_root)
import datetime, platform, time, json, argparse
import os.path as osp


# from all_include.pt_utils_file import rename_file # , time_this
# from all_include.pt_utils_img import convert_to_jpg
# from all_include.labelme.npy2json import repair_name
from json2coco import to_coco
IS_WINDOWS, TIME_TH, TRUE, FALSE, RSTRIP, COPY, CP_FILE, MOVE, DEL, DEL_EMPTY = False, datetime.datetime.now(), True, False, '/', 'cp -rf', 'cp', 'mv', 'rm -rf', 'rm -d'
TM, SHOW_MANY, SEG_IMG_SHOW = TIME_TH_STR_mdHMS = TIME_TH.strftime(f"%m%d_%H%M%S"), False, False
if 'Windows' in platform.uname()[0]:
    print(f"os is Windows")
    IS_WINDOWS, RSTRIP, COPY, CP_FILE, MOVE, DEL, DEL_EMPTY = True, '\\', 'xcopy /s /c /y', 'copy /y', 'move', 'del', 'rd'  # 'xcopy /s /c /y'
ROOT = os.path.dirname(os.path.abspath(__file__))
print(f"ROOT={ROOT}")
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='img', help='input raw images')
args = parser.parse_args()

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
def os_system_del(fp, _cmd=DEL_EMPTY, cfm=False):
    if not osp.exists(fp): return 0 # print(f"... skip delete due to not found: {fp}");
    if _cmd not in [DEL_EMPTY, DEL]:
        input(f"!! _cmd({_cmd}) must in [DEL_EMPTY, DEL], we will set it to {DEL_EMPTY}\n")
        _cmd = DEL_EMPTY
    c = input(f"input 'y' or 'c' to delete {fp}\n") if cfm else 'y'
    if c.lower() not in ['c', 'y']: print("skip delete"); return 1
    r = os.system("{} {}".format(_cmd, fp))
    if 0 != r: input(f"! failed {DEL} {fp} \n input 'c' to continue\n"); return 0
    return 1  # 1 for += 1
def if_print(str, b_prt=True):
    if b_prt:
        print(str)
        return True
    else:
        return False
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
            # if 'del' == r: os.system("{} {}".format(DEL_EMPTY, fp))
            # elif r not in ['c', 'C']: assert False
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
def path_leagle(_path, _how='skip'):
    assert type(_how) is str, type(_how)
    if _path == '': return _path
    _err = False
    if IS_WINDOWS:
        assert RSTRIP == '\\', "?? found WINDOWS RSTRIP={}".format(RSTRIP)
        _path = str(_path).replace('/', '\\')
    else:
        assert RSTRIP == '/', "?? found LINUX RSTRIP={}".format(RSTRIP)
        _path = str(_path).replace('\\', '/')
    return _path
def join_path(_path_list, _prt=False):
    assert isinstance(_path_list, list) and len(_path_list) > 1
    _path_list = [e for e in _path_list if e is not None]
    if _prt: input3(_path_list)
    # path_leagle(_path_list[0], _how='input' if _prt else 'print') # !! do NOT do this, as it changes _path_list[0]
    _strip = '\\' if IS_WINDOWS else '/' # if '/' in _path_list[0] else '' # -_- I don't know why # RSTRIP # 
    if _prt: input3("_strip={}, _path_list[0]={}".format(_strip, _path_list[0]))
    new_path, ct = "", 1
    for p in _path_list:
        if p == '': continue
        p = path_leagle(p, _how='input' if _prt else 'skip')
        if _prt: print("p[0]={}, _strip={}, _path_list={}".format(p[0], _strip, _path_list))
        new_path = p if new_path == '' else new_path + p if p[0] in ['/'] else "{}{}{}".format(new_path,_strip,p)
    if_print("new path: {}".format(new_path), _prt)
    if ('\\' in new_path == '/' in new_path): input3("?? _path_list={}\n new_path={}\n".format(_path_list, new_path))
    new_path = path_leagle(new_path, _how='skip')
    return new_path
# end join_path
def check_name(json_this, fnp_txt):
    data = json.load(open(json_this))
    name_json = osp.splitext(osp.basename(json_this))[0]
    fnp_in_json = data['imagePath'] # osp.basename()  # osp.splitext()[0]
    fn_new = name_json + '.jpg'
    # print("old:{}, new:{}".format(fnp_in_json, fn_new))
    if fnp_in_json != fn_new:
        if fnp_txt is not None:
            print("--rename data['imagePath'] from '{}' to '{}'".format(fnp_in_json, fn_new))
        data['imagePath'] = str(data['imagePath']).replace(fnp_in_json, fn_new)
        json.dump(data, open(json_this, 'w'))
        return 1
    else: return 0
# end check_name
def repair_name(_fp_in, _cfm=True, time_str=TM):
    print("repair data['imagePath'] of: {}".format(_fp_in))
    assert osp.exists(_fp_in), "! {}".format(_fp_in)
    if __file__[1:2] != ':' and os.system("which jq") != 0: input3('apt install jq') # RuntimeError('apt install jq')
    fp_temp = osp.join(osp.dirname(_fp_in), "{:s}_tmp_{:s}".format(osp.basename(_fp_in), time_str))
    num_d, num_f, num_json, num_r, _list = 0, 0, 0, 0, []
    t0, t1 = time.time(), time.time()
    for root, dirs, files in os.walk(_fp_in):
        for dir in dirs:
            num_d += 1
        for file in files:
            num_f += 1
            if file.endswith('.json'):
                num_json += 1
                src = osp.join(root, file)
                try:
                    r = check_name(src, None)
                    if r == 1 and not IS_WINDOWS: # bug in 'os.system(f'cat {src} | jq . > {dst}')'
                        num_r += 1
                        os.makedirs(fp_temp, exist_ok=True)
                        dst = osp.join(fp_temp, file)
                        rtn = os.system(f'cat {src} | jq . > {dst}') # ("cat {} | jq . > {}".format(src, dst))
                        if rtn != 0: raise RuntimeError("!src={}, dst={}, rtn={}".format(src, dst, rtn))
                        os.system("{} {}".format(DEL, src))
                        os.system("{} {} {}".format(MOVE, dst, src))
                        _list.append(file)
                except Exception as e:  # NOT work -_-
                    raise Exception("!!file: {} \n {}".format(file, e))
                finally: continue
            if (num_f % 1000) == 0 or num_f <= 10:
                print("checked {} dirs, {} files, {} json, repair {}".format(num_d, num_f, num_json, num_r))
    if _list and _cfm:
        r = input("input 'y' or 'Y' to display {} repaired file name, others to continue\n".format(num_r))
        if r in ['y', 'Y']: print(_list)
    print("checked all {} dirs, {} files, {} json, repair {}. deleting tmp ..".format(num_d, num_f, num_json, num_r))
    os.system(f'{DEL} {fp_temp}')
# end repair_name()
def prepare_coco(_cp=False, _fp_in='', _ext0='.jpg', _data_name='sz', dev_dir=osp.join(ROOT, 'pkl')):
    dev_dir = ROOT # for debug
    print(f"dev_dir={dev_dir}")
    if not isinstance(_cp, bool): input(f" !! check _cp to be 'True' or 'False' !\n"); return _fp_in, ''
    if _fp_in in ['', None]:
        _fp_in = osp.join(ROOT, 'img')
    assert osp.exists(_fp_in), f"?? prepare_coco with:{_fp_in}"
    _root = osp.dirname(_fp_in)  # _fp_in = osp.join(_root, 'input')
    # print(f"root is {_root}")
    _fp_coco = osp.join(_root, "{}_coco".format(osp.basename(_fp_in)))  # _fp_coco = osp.join(_root, 'input_coco')
    # input3('_fp_coco={}'.format(_fp_coco))
    _log_path_ = osp.join(_root, 'output_tmp', 'ex_{}_{}'.format(_data_name, TM))
    _log = osp.join(_log_path_, 'log_{}.txt'.format(TM))
    os.makedirs(_log_path_, exist_ok=True)
    if osp.exists(_fp_coco):
        r = input("! already exist: {}\n input 'del' to delete, 'c' to use this coco and continue, others quit\n".format(_fp_coco))
        if r in ['del', 'DEL', 'Del']: os.system("{} {}".format(DEL, _fp_coco))
        elif r in ['c', 'C', '']: return _fp_coco, _log
        else: raise KeyError("886")

    _fnp_fake_json = join_path([dev_dir, 'fake_json.json']) # path_leagle(DEV_FAKE_JSON)  # 
    _lb_file = 'lb.txt' # 'labels_chr2.txt' if NUM_CLASS_SEG == 2 else 'labels_chr1.txt'
    _fnp_label = join_path([dev_dir, _lb_file])
    assert osp.exists(_fnp_fake_json) and osp.exists(_fnp_label), "--json={}\n lb={}".format(_fnp_fake_json, _fnp_label)

    # convert_to_jpg(_fp_in, _ext0, '.jpg', _debug=False, _cfm=False)
    # rename_file(_fp_in, _str0=' ', _str2='_', _cfm=False)
    _fns, _dirs, _files, _ld, _lf, _ldf = get_fns(_fp_in)  # , _only_img=True)
    _files.sort()
    if _ldf < 1: input(f"??? {_ldf} fiels/dirs, fns={_fns}")
    else: print(f" going to make {_lf} files ({_fns[0]})")
    nf, nr, nc = 0, 0, 0
    for file in _files:
        nf += 1
        file1 = file.replace(' ', '_')
        file2 = file1.replace('(', '_')
        file3 = file2.replace(')', '_')
        if file != file3: os.rename(osp.join(_fp_in, file), osp.join(_fp_in, file3))
        name = osp.splitext(file3)[0]
        # input3("_fp_in={}\n _fnp_fake_json={}\n".format(_fp_in, _fnp_fake_json))
        _fnp_json_dst = osp.join(_fp_in, f'{name}.json')  # , _prt=True)
        # input(f"copy to {_fnp_json_dst}")
        nc += os_system3(_fnp_fake_json, _fnp_json_dst, COPY)
        if nf < 3 or nf % 50 == 0: print(f" NO.[{nf}/{_ldf}], copied: {nc}")
    _fns2, _, _, _, _, _ldf2 = get_fns(_fp_in, _only_file=True)
    # assert _ldf * 2 == _ldf2, "ldf={}, ldf2={}".format(_ldf, _ldf2)
    print(_fns2)

    # r = input("manual make coco, and input 'c' to continue\n")
    # time.sleep(2)
    repair_name(_fp_in, _cfm=False)
    to_coco(_fp_in, _fnp_label, _fp_coco)

    # py_img = osp.join(_root, 'pt_utils_img.py')
    # py_npy2json = osp.join(_root, 'npy2json.py');
    # py_json2coco = osp.join(_root, 'json2coco.py')
    # os.system("python {} name --input_dir  {}".format(py_npy2json, _fp_in))
    # os.system("python {} --input_dir {}".format(py_json2coco, _fp_in))

    _fns, _dirs, _files, _ld, _lf, _ldf = get_fns(_fp_coco)
    assert 'annotations.json' == _files[0], '{}'.format(_files)
    if _cp:
        for fn in _fns:
            src = osp.join(_fp_coco, fn)
            dst = osp.join(_root, fn)
            os_system_del(dst, DEL, True)
            r = os_system3(src, dst, COPY)
    return _fp_coco, _log
# end prepare_coco

if __name__ == '__main__':
    print(f"start coco_prepare.py @ {TM}")

    prepare_coco(True, args.input_dir)
