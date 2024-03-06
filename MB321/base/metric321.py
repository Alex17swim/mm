# metric321.py, Chengyu Wang, 2023-09-21, XJTLU, last update: 2023-11-02 16:45

import sys, os
_import_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_import_root)
# print("metric321.py _import_root={}".format(_import_root))

# import numpy as np
# import matplotlib.pyplot as plt
# import os.path as osp
from MB321.base.util321 import * # , np_mean

cm_axis_name = []
LB_T21 = False
TEMP_ROOT = agDIR.temp
if LB_T21: cm_axis_name = ['other','21']
else:
    for i in range(24):
        cm_axis_name.append("%02d" % (i+1))  # ("chr'%02d'" % (i))
    assert type(cm_axis_name) is list and type(cm_axis_name[0]) is str


def do_metric(np_y_, np_lb, _set, _epoch, _out_dir=None, _deci=4, _prt=False, _log=None, _axis_name=None): # PRINT_DECIMAL y_: the predicted yhat; lb: true label
    if _axis_name is None: _axis_name = cm_axis_name
    assert type(np_y_) is np.ndarray and type(np_lb) is np.ndarray,"! {}, {}".format(type(np_y_), type(np_lb))
    _num_sample, _num_cls = len(np_y_), len(np_y_[0])
    assert _num_sample == len(np_lb), "!  {}, {}".format(_num_sample, len(np_y_))
    assert _num_cls == 24, "! {}".format(_num_cls)
    pred = np.argmax(np_y_, 1) #; _values, pred = torch.max(y_.data, 1)
    correct = np.array(pred == np_lb).sum() # (pred == lb).squeeze().sum().to('cpu').numpy()
    acc = np_div(correct, _num_sample, _deci)
    if_print("acc={}\n pred={}\n lb  ={}".format(acc, pred, np_lb), False)

    confusion_matrix, _wrong, _total, _bg = get_confusion_matrix(_num_cls, np_lb, pred)
    _recall_list, _preci_list, _correct_list, _acc_mean, _recall_mean, _precision_mean =\
        show_confMat(confusion_matrix, _axis_name, _set, _epoch, _out_dir, _log=_log, verbose=_prt, _check=[acc, correct, _num_sample])
    assert abs(acc - _acc_mean) <= 0.000001, "! {}, {}".format(acc, _acc_mean)

    _metric_dict = {'acc': _acc_mean, 'recall': _recall_mean, 'precision': _precision_mean}
    return _metric_dict, _recall_list, _preci_list # [acc, _recall_mean, _precision_mean]
# end do_metric

def get_confusion_matrix(_num_cls, np_lb, pred, _log_list=None, _use_bg=False, _prt=False):
    assert len(np_lb) >= len(pred), "! lb={}, pred={}".format(len(np_lb), len(pred))
    _range = list(np.arange(_num_cls, dtype=int)) # <=23
    if _use_bg: _range.append(_num_cls) # _use_bg -1 # _use_bg = 24+1
    _num_sample = len(pred)
    _matrix = np.zeros((len(_range), len(_range)))  # confusion_mat with _num_cls*_num_cls matrix
    assert np.max(np_lb) <= max(_range) and np.max(pred) <= max(_range), f"lb={np.max(np_lb)}, pred={np.max(pred)}"
    assert np.min(np_lb) >= 0 and np.min(pred) >= 0, "lb={}, pred={}".format(np.min(np_lb), np.min(pred))
    _wrong, _total, _bg = [], [], []; _num_wrong, _num_total = 0, 0
    for j in range(_num_sample):  # sample range
        cate_i = np_lb[j]  # .cpu().numpy()
        pre_i = pred[j]  # .cpu().numpy()
        _matrix[cate_i, pre_i] += 1.
    # if_print(confusion_matrix, _prt)
    # write_txt(_log_list, "saving confusion matrix all at {} ... ".format(time_str()))

    _matrix_N = _matrix.copy()  # normalization
    for i in range(_num_cls):
        c = _matrix[i, :]
        _wrong.append(int(c.sum() - c[i]))
        _total.append(int(c.sum()))
        _matrix_N[i, :] = c[i] / c.sum() if c.sum() > 0 else 0
        _bg.append(int(c[len(_range) - 1]))
        # write_txt(_log_cm, "No.{} wrong=[{}/{}]".format(i + 1, _wrong[i], _total[i]), _prt)
    _num_wrong, _num_total, _num_bg = sum(_wrong), sum(_total), sum(_bg)
    if _prt:
        write_txt(_log_list, " wrong=({}){}\n total=({}){}\n bg=({}){}\n"
                  .format(_num_wrong, _wrong, _num_total, _total, _num_bg, _bg), True)
        _tag, _dir = f"temp_{time_str()}_total{_num_total}", osp.dirname(_log_list)
        plt_confusion_matrix(_matrix, cm_axis_name, _tag, _dir,  _num_wrong, _num_total, _num_bg)
    return _matrix, _num_wrong, _num_total, _num_bg
# end get_confusion_matrix

def plt_confusion_matrix(_matrix, _axis_name, _tag, _out_dir, _wrong, _total, _bg=0, _ext='.pdf', _color_thr=2): # 36
    assert _ext in ['.eps', '.pdf', '.jpg'], _ext
    _color_fig, _color_font = 'Purples', 'black' # Blues, Oranges # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    if not len(_matrix) == len(_axis_name):
        print("! mat={}, axis={}, append background".format(len(_matrix), len(_axis_name)))
        _axis_name.append('BG')
    _num_cls = len(_matrix); _corr = int(_total - _wrong)
    _matrix_N = _matrix.copy() # ???
    _fnp = osp.join(_out_dir, f"{_tag}_{time_str()}_{_ext}")
    # _fnp2 = osp.join(_out_dir, "{}2.jpg".format(_tag))
    if osp.exists(_fnp): input(f"!! already exists: {_fnp}, contine\n")
    else: # if not osp.exists(_fnp):
        os.makedirs(_out_dir, exist_ok=True)
        if _ext != '.jpg':
            _label_x = 'Predicted label'
            _fs = 18
            fig = plt.figure(_tag, figsize=(8, 6), clear=True)
        else:
            _label_x = 'Predict label corr={}({:.3f}), err={}({:.3f}), bg={}({:.3f})'.\
                format(_corr, np_div(_corr, _total), _wrong, np_div(_wrong, _total), _bg, np_div(_bg, _total))
            _fs = 22
            fig = plt.figure(_tag, figsize=(16, 14), clear=True)
        # set color
        cmap = plt.cm.get_cmap(_color_fig) # 更多颜色: https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
        plt.imshow(_matrix_N, cmap=cmap)
        plt.colorbar()
        plt.savefig(_fnp[:-4] + f"color.jpg")

        # set label
        xlocations = np.array(range(_num_cls))
        plt.xticks(xlocations, list(_axis_name), rotation=60)
        plt.yticks(xlocations, list(_axis_name))
        plt.xlabel(_label_x, fontsize=_fs)
        plt.ylabel('True label', fontsize=_fs)
        plt.tick_params(labelsize=_fs-6)
        if _ext == '.jpg': plt.title("{}_of_{}_class".format(_tag, _num_cls), fontsize=26, y=1.02)  # 'xx-large'

        # print figure
        for r in range(0, _matrix_N.shape[0]):
            for c in range(0, _matrix_N.shape[1]):
                _v = int(_matrix[r, c]); _f = int(_fs/2); _color = _color_font
                if _v > 99: _f = int(_f-2)
                if r == c and _v > _color_thr: _color = 'white' # _v > _color_thr to avoid white font in white back color
                plt.text(x=c, y=r, s=_v, va='center', ha='center', color=_color, fontsize=_f) # 10
                if _matrix_N.shape[0] < 3:
                    # print(f"c{c+1}r{r+1}.jpg: c+1={c+1}, r+1={r+1}, v={_v}, color={_color}")
                    plt.savefig(_fnp[:-4] + f"c{c+1}r{r+1}.jpg")

        # fig_snap = plt.gcf(); fig_snap.savefig(_fnp2) # more clear
        if _ext != '.jpg':
            plt.subplots_adjust(top=0.98, bottom=0.12, right=0.9, left=0.1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(_fnp[:-4]+'.jpg')
            plt.pause(1) # plt.show()
            fig.savefig(_fnp, dpi=200)
            plt.close()
        else: plt.savefig(_fnp)
# end plt_confusion_matrix

def show_confMat(_matrix, _axis_name, set_name, _epoch, _out_dir, _log=None, verbose=False, _deci=PRINT_DECIMAL, _check=None):
    if type(_log) is str: _log=[_log]
    assert type(_axis_name) is list and type(_axis_name[0]) is str,\
        "! '_axis_name' is a list of class name(str), found:{}".format(_axis_name)
    if verbose and 'train' in set_name: verbose=False
    _confusion_mat_save, _axis_save = _matrix, _axis_name
    if (not LB_T21) and (_matrix[23, :].sum() == 0): # and _matrix[:, 23].sum() == 0: # case 34, 41, 44, 57, 65, 66, 68, 78, 89
        if_print("~a female, with only 24-1 classes: ", verbose)
        # _axis_name.remove(_axis_name[23])
        # _matrix = np.delete(_matrix, 23, axis=0) # last line (class 24)
        # _matrix = np.delete(_matrix, 23, axis=1) # last column in each line (class 24)
    _num_cls = len(_axis_name)
    _matrix_N = _matrix.copy() # normalization
    for i in range(_num_cls):
        if _matrix[i, :].sum() > 0: _matrix_N[i, :] = _matrix[i, :] / _matrix[i, :].sum()
        else:
            _matrix_N[i, :] = 0
            # if_print("!! _matrix_N[{},:]={}, sum={}".format(i, _matrix_N[i,:], _matrix_N[i,:].sum()), verbose)

    _correct_list, _recall_list, _preci_list, _acc, _recall, _preci = [], [], [], 0.0, 0.0, 0.0
    _nan = np.round(0.1**(_deci+1), _deci+1) # to avoid 'nan' due to 0 in gt or pred; bad pred [0%, 62%]: case[10, 68]
    try:
        update_r, update_p = [], []
        for i in range(_num_cls):
            _correct_list.append(_matrix[i, i])
            _r, _p = np.sum(_matrix[i, :]), np.sum(_matrix[:, i])
            if _r > 0: update_r.append(_r)
            if _p > 0: update_p.append(_p)
            _recall_list.append(_matrix[i, i] / (_nan + _r)) # of ground truth row
            _preci_list.append(_matrix[i, i] / (_nan + _p)) # of prediction column
            # if False:
            #     print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(_axis_name[i], np.sum(_matrix[i, :]), _correct_list[i], _recall_list[i], _preci_list[i]))
        # if len(update_r) != len(update_p): print("? update_r={}\n  update_p={}\n  _correct={}"
        #                                          .format(update_r, update_p, _correct_list))
        _recall_list, _preci_list, _correct_list = np.round(_recall_list, _deci), np.round(_preci_list, _deci), np.round(_correct_list, _deci)
        
        _acc = np_div(np.array(_correct_list).sum(), _matrix.sum())
        _real_class_num = len(update_r) # female only have 23 instead of 24
        _recall = np_div(np.array(_recall_list).sum(), _real_class_num) # len(update_r) # len(_matrix)
        _preci = np_div(np.array(_preci_list).sum(), _real_class_num) # len(update_p) # len(_matrix[0])
        if abs(_acc - 1.0) <= 0.00001 and (abs(_recall - 1.0) > 0.00001 or abs(_preci - 1.0) > 0.00001):
            write_txt(_log, "\n ??? _acc={}, _recall={}, _preci={}".format(_acc, _recall, _preci))

        write_txt(_log, "_acc={}, _recall={}, _preci={}".format(_acc, _recall, _preci), b_prt=verbose)
        if _check:
            acc, correct, _num_sample = _check
            if abs(acc - np.round(acc, _deci)) > _nan:
                write_txt(_log, "_acc={}, acc={}, correct={}, _samples={}".format(_acc, acc, correct, _num_sample))

    except Exception as e:
        write_txt(_log, f"{e}\n !! denominator can't be 0 !")
    if _out_dir is not None:  # and _epoch > 0:
        _total = int(_matrix.sum())
        _corr = int(np.array(_correct_list).sum())
        _wrong = int(_total - _corr)
        _tag = 'CM_{}_e{}_samples{}'.format(set_name, _epoch, _total)
        plt_confusion_matrix(_matrix, _axis_name, _tag, _out_dir, _wrong, _total)

    return _recall_list, _preci_list, _correct_list, _acc, _recall, _preci
# end show_confMat

def confusion_matrix_all(pred, np_lb, _set, _epoch, _out_dir=None, _log_list=None, _axis_name=None, _deci=4, _prt=False):
    assert type(_set) is str, "type={}, {}".format(type(_set), _set)
    assert type(_epoch) is int, "type={}, {}".format(type(_epoch), _epoch)
    assert type(_out_dir) is str, "type={}, {}".format(type(_out_dir), _out_dir)
    assert type(_log_list) in [None, str, list], "type={}, {}".format(type(_log_list), _log_list)
    if _axis_name is None: _axis_name = cm_axis_name # 'Default argument value is mutable'

    if type(pred) is list: pred = np.array(pred)
    if type(np_lb) is list: np_lb = np.array(np_lb)
    assert type(pred) is np.ndarray and type(np_lb) is np.ndarray, "! {}, {}".format(type(pred), type(np_lb))
    assert len(pred) == len(np_lb), "!pred={}, lb={}".format(len(pred), len(np_lb))
    _num_sample, _num_cls = len(pred), len(_axis_name)
    # pred = np.argmax(np_y_, 1)  # ; _values, pred = torch.max(y_.data, 1)
    _tag = 'all_ConfMatrix_{}_e{}_samples{}'.format(_set, _epoch, _num_sample)
    _log_cm = osp.join(_out_dir, "{}.txt".format(_tag))
    _fnp = osp.join(_out_dir, "{}.jpg".format(_tag))
    # _fnp2 = osp.join(_out_dir, "{}2.jpg".format(_tag))
    if osp.exists(_fnp):
        r = input("!! already exists: {}\n input 'c' to replace, others to quit\n")
        if r not in ['c', 'C']: raise KeyError("886")

    if _log_list:
        if type(_log_list) is str: _log_list = [_log_list]
        else: assert type(_log_list) is list, "! type={}".format(type(_log_list))
        _log_list.append(_log_cm) # .extend(_log_cm)
    else: _log_list = [_log_cm]

    confusion_mat, _wrong, _total, _bg = get_confusion_matrix(_num_cls, np_lb, pred, _log_list=_log_list)
    plt_confusion_matrix(confusion_mat, _axis_name, _tag, _out_dir, _wrong, _total)

    # write_txt(_log_list, "done at {}, cost {}".format(time_str(), time_gap(tt, True)))
    return confusion_mat
# end confusion_matrix_all

if __name__ == '__main__':
    _out_dir = osp.join(TEMP_ROOT, 'log', time_str())
    _log = osp.join(_out_dir, 'log.txt')
    print("_out_dir:", _out_dir)

    cls_num, batch_size = 2, 10
    # y_ = np.random.randint(0, cls_num, (batch_size, cls_num))
    # lb = np.random.randint(0, cls_num, (batch_size))
    # print(" y_ = {}\n lb = {}".format(y_, lb))

    # _metric_dict, _recall_list, _preci_list = do_metric(y_, lb, _set='test', _epoch=888, _out_dir=_out_dir)
    # print("{}".format(_metric_dict))

    yhat = np.random.randint(cls_num, size=cls_num) # np.arange(0,cls_num) # np.array([1,2,3,4,5])
    yhat = np.concatenate((yhat, yhat, yhat), 0)
    # print("pred=", pred, "\n concatenate=", np.stack((pred, pred), 0))
    np_lb = np.arange(0,cls_num) #np.array([1,2,3,4,5])
    np_lb = np.concatenate((np_lb, np_lb, np_lb), 0)

    # yhat = np.array([0,0,0,0], dtype=int)
    # np_lb = np.array([0,0,0,0], dtype=int)
    axis_name = ['aaa', 'bbb', 'ccc']  # cm_axis_name
    
    # np_lb = yhat # to make 100%
    print("yhat=", yhat, "\n lb=", np_lb)
    _set, _epoch = 'temp', 2
    confusion_mat = confusion_matrix_all(yhat, np_lb, _set, _epoch, _out_dir, _log_list=[_log, _log], _axis_name=axis_name, _prt=True)

    # _recall_list, _preci_list, _correct_list, _acc, _recall, _preci =\
    #     show_confMat(confusion_mat, axis_name, _set, _epoch, _out_dir, _log=_log)
    # write_txt(_log, " _recall_list={}\n _preci_list={}\n _correct_list={}\n"
    #                 " _acc={}, _recall={}, _preci={}"
    #           .format(_recall_list, _preci_list, _correct_list, _acc, _recall, _preci))

# --------------------
# confusion_mat, _wrong, _total, _bg = get_confusion_matrix(len(cm_axis_name), np_lb_gt, np_lb_pred, _log_list)
# _recall_list, _preci_list, _correct_list, _acc_mean, _recall_mean, _precision_mean = show_confMat(confusion_mat, cm_axis_name, 'test', -1, _out_dir, _log=log_cm)