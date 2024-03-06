# plt321.py, Chengyu Wang, 2023-1129, last update on 2023-1129
import random
import matplotlib.pyplot as plt
from MB321.mm_flag import *
# import matplotlib
# matplotlib.use('TKagg') # for UserWarning: Matplotlib is currently using agg

def plt_subplot(img, row, col, pos, xlabel='', ylabel=''):
    # plt.title(title) # !! NOT set here
    plt.subplot(row, col, pos)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([]); plt.yticks([])
    plt.imshow(img)

def plt_subplot3(fp_out, data_infos, _key='img', _title='', fs=18, _sec=1):
    def _rand_i(_len, _num):
        if _len <= _num: return random.randint(0, _len)
        else: return _num
    
    if fp_out: os.makedirs(fp_out, exist_ok=True)
    L = len(data_infos) - 1 # !!! -1
    _str_rand = '_rand' if L <= 24 else ''
    plot_ids = [_rand_i(L, 24), _rand_i(L, 25), _rand_i(L, 26), # [24, 25, 26] for [21a, 21b, 21tri]
                _rand_i(L, 46-2), _rand_i(L, 46-1), _rand_i(L, 46),
                _rand_i(L, 92-2), _rand_i(L, 92-1), _rand_i(L, 92)]
    for i, _id in enumerate(plot_ids): # range(0,4)
        img = data_infos[_id][_key]
        _tag = f"No.{_id+1:<02d}_{data_infos[_id]['name']}{_str_rand}" if 'name' in data_infos[_id].keys() else f"No.{_id+1}{_str_rand}"
        if fp_out: plt.imsave(osp.join(fp_out, f"{_tag.rstrip('.jpg')}.jpg"), img)
        plt_subplot(img, 3, 3, (1+i), xlabel=_tag, ylabel=f"{data_infos[_id]['gt_label']}")
        # plt_subplot(img, 1, len(plot_ids), (i+1), xlabel=_tag, ylabel=f"{data_infos[_id]['gt_label']}")
    if _title: plt.suptitle(_title, fontsize=fs)
    plt.savefig(osp.join(fp_out, f"plot{len(plot_ids)}_{time_str()}.jpg"))
    if _sec > 0: plt.pause(_sec)
    elif _sec == 0: plt.show()
    plt.clf()