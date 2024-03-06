# mb321_flag.py, 2024-0225, Chengyu Wang, last update 2024-0225
''' *: os  $$: gpu  ##: dataset  &: image  ---: model  --: loss  -: directory  '''
# print(f"accuracy={acc:.3f}; epoch={e:03d}"); print('epoch [{:0>3}/{:0>3}]'.format(epoch, epoch_max))
import sys, os, platform, datetime, time, argparse # , copy  #
_import_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_import_root) # if 'win' in platform.uname()[0].lower():
import os.path as osp # os.path.basename(sys.argv[0]) # this .py filename

USE_GPU_LIST = [1]
os.environ['CUDA_VISIBLE_DEVICES'] = f"{int(USE_GPU_LIST[0])}"

ITER = 10000 # 140000
TIME_TH, TRUE, FALSE, RSTRIP, COPY, CP_FILE, MOVE, DEL, DEL_EMPTY = datetime.datetime.now(), True, False, '/', 'cp -rf', 'cp', 'mv', 'rm -rf', 'rm -d'
TM = TIME_TH_STR_mdHMS = TIME_TH.strftime("%m%d_%H%M%S")
IS_WIN = True if 'win' in platform.uname()[0].lower() else False
DEV = 'cpu' if IS_WIN else f'cuda:{int(USE_GPU_LIST[0])}' # f"cuda" # :1
print(f"IS_WIN={IS_WIN}; DEV={DEV}")
if IS_WIN:
    # print(f"os is Windows")
    IS_WINDOWS, RSTRIP, COPY, CP_FILE, MOVE, DEL, DEL_EMPTY = True, '\\', 'xcopy /s /c /y', 'copy /y', 'move', 'del', 'rd'  # 'xcopy /s /c /y'
root_github = '/public/home/alex/1github'
prs = argparse.ArgumentParser() # formatter_class=argparse.ArgumentDefaultsHelpFormatter
prs.add_argument('--num_steps', type=int, default=ITER)
prs.add_argument('--exp_name', type=str, default=f"{DEV.replace(':','')}_{TM}")
prs.add_argument('--model_enc', type=str, default=os.path.join(root_github, 'output', 'saved_models', 'encoder.pth'))
prs.add_argument('--model_dec', type=str, default=os.path.join(root_github, 'output', 'saved_models', 'decoder.pth'))
prs.add_argument('--train_path', default=osp.join(root_github,'data',
                                                    # 'VOC/voc2012/JPEGImages'),
                                                    'mirflickr/images7000'),
                 help="/public/home/alex/1github/data/VOC/voc2012/JPEGImages")
prs.add_argument('--checkpoints_path', default=osp.join(root_github,'output', 'checkpoints/'), help="/public/home/alex/1github/output/checkpoints/")
# prs.add_argument('--image', type=str, default=None) # one image
prs.add_argument('--images_dir', type=str, default=osp.join(root_github, 'data', 'water_test')) # many images
prs.add_argument('--save_dir', type=str, default=osp.join(root_github, 'output', 'images'))
prs.add_argument('--hide_dir', type=str, default=osp.join(root_github, 'output', 'hide'))
prs.add_argument('--secret', type=str, default='Stega!!') # XJTLU!!, Stega; len < 7
prs.add_argument('--secret_size', type=int, default=100) # 100 in pt; 20 in tf
# prs.add_argument('--cuda', type=bool, default=True)
agStega = prs.parse_args()
prs.add_argument('--saved_models', default=osp.join(root_github, 'output', 'saved_models', agStega.exp_name), help="/public/home/alex/1github/output/saved_models")
prs.add_argument('--logs_path', default=osp.join(root_github,'output', 'logs/'), help="/public/home/alex/1github/output/logs/")
prs.add_argument('--fnp_log', default=osp.join(root_github,'output', 'logs', f"{agStega.exp_name}.md"), help="/public/home/alex/1github/output/logs/")
agStega = prs.parse_args() # update 1


def time_str(t=''): return t.strftime("%m%d_%H%M%S") if isinstance(t, datetime.datetime) else datetime.datetime.now().strftime("%m%d_%H%M%S")  # time.strftime("%H%M")
def time_gap(_t=TIME_TH):
    if type(_t) is datetime.datetime:
        _gap = datetime.datetime.now() - _t
        _gap = _gap.seconds  # /10000
    elif type(_t) is datetime.timedelta:
        _gap = _t.seconds
    elif type(_t) is float:
        if _t > 60 * 60 * 24 * 30:  # 60s*60m*24h*30day, so _t is: time
            _gap = time.time() - _t
        else: _gap = _t  # _t is: time-time
    else:
        input(f"!! _t={_t} ,type={type(_t)}\n")
        return
    if _gap > 3600.0:
        _gap /= 3600.0
        str_d = ": %.2f hours" % _gap
    elif _gap > 60.0:
        _gap /= 60.0
        str_d = ": %.2f minutes" % _gap
    else: str_d = ": %.2f seconds" % _gap
    # if _str:
    #     str_d = str_d[str_d.index(' ')+1:]
    #     str_d = str_d.replace(' ', '_')
    return str_d
# end time_gap
def write_txt(fnp_list, _str, b_prt=True, _mode='a', _write=True):
    _list_none = [None, '', [None], [''], []]
    if b_prt: print(_str)
    if not _write: return
    if fnp_list in _list_none or len(fnp_list) < 1: return
    if isinstance(fnp_list, str): fnp_list = [fnp_list]
    if not type(fnp_list) is list: raise ValueError("??fnp_list={}".format(fnp_list))
    fnp_list = list(set(fnp_list))  # remove repeated item
    fnp_list = [e for e in fnp_list if e not in _list_none]
    assert fnp_list is not None and len(fnp_list) >= 1, "!check fnp_list:{}".format(fnp_list)
    for fnp in fnp_list:
        os.makedirs(os.path.dirname(fnp), exist_ok=True)
        with open(fnp, _mode) as fw:
            if isinstance(_str, str): fw.writelines(_str + '\n')
            else: fw.write(str(_str))
# end write_txt
if __name__ == '__main__':
    print(f"- mb321_flag.py done at: {time_str()}")