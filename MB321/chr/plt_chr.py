# plt_chr.py, Chengyu Wang, 2023-1129, last update on 2023-1129
from mm_flag import *
import torch
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TKagg') # for UserWarning: Matplotlib is currently using agg
from torch import Tensor, isin
from PIL import Image
from MB321.mm_flag import *
num_cls  = agDATA.num_cls
def sub_str(_fn0, _head=None, _tail=None):
    while _head and _head in _fn0:  # use 'while' for multiple '_head'
        assert type(_head) is str and len(_head) > 0, _head
        _fn1 = _fn0[_fn0.index(_head) + len(_head):]
        _fn0 = _fn1
    else: _fn1 = _fn0
    while _tail and _tail in _fn1:
        assert type(_tail) is str and len(_tail) > 0, _tail
        _fn2 = _fn1[: _fn1.rindex(_tail)] # .index
        _fn1 = _fn2
    else: _fn2 = _fn1
    return _fn2
def str_replace(s, k1_list=None, k2=''):
    if k1_list is None: k1_list = ['_pred_roat_fit', '_roat_fit', '_pred_roat_keep_fit', '_roat_keep_fit']
    if isinstance(k1_list, str): k1_list = [k1_list]
    for k1 in k1_list: s = s.replace(k1, k2)
    return s
def plt_one_pred_karyotype(_pred, X1_q, _fns, fp_case_out, fp_src, sp0_hwc=None, _tail_="", plt_t=0, _each=True, _new=True):
    global ignore_pos    
    if isinstance(_pred[0], Tensor) and _pred[0].shape[0] > 1:
        _str = f"... expect (predict)yhat label of numpy, not scores of {type(_pred[0])}, _pred[0].shape={_pred[0].shape}"
        _pred = [p.argmax().cpu().numpy() for p in _pred]
    elif isinstance(_pred[0], np.ndarray) and _pred[0].size > 1:
        _str = f"... expect (predict)yhat label, not scores, _pred[0].size={_pred[0].size}"
        _pred = [p.argmax() for p in _pred]
    elif not isinstance(_pred[0], np.int64):
        _str = f"!!! expect (predict)yhat label is np.int64, not {type(_pred[0])}"
        _pred = [p.astype(np.int64) for p in _pred]
    else: _str = None # isinstance(_pred[0], np.int64)
    if _str: print(_str)            
        
    # _case_name, _root_grand = osp.basename(fp_case_out), osp.dirname(fp_case_out)
    _case_name = osp.basename(fp_src)
    _case_name = sub_str(_case_name, 'original_')
    _keys = ['karyo', 'src', 'chw']
    if _new: # create new folder for karyotype, chw, src
        _root_grand = osp.dirname(fp_case_out)
        _root_dict, _fp_case_dict = {'root': _root_grand}, {}
        for k in _keys:
            if not QUICK and k == 'chw': continue # 'chw' for debug
            _root_dict[k] = osp.join(_root_grand, k)
            _fp_case_dict[k] = osp.join(_root_dict[k], _case_name)
            if k == _keys[0]: _fp_case_dict[k] = _root_dict[k]  # karyo do NOT need a case folder
            os.makedirs(_fp_case_dict[k], exist_ok=True)
    else:
        _each = False
        _root_grand = fp_case_out
        _root_dict, _fp_case_dict = {'root': _root_grand}, {}
        for k in _keys:
            _root_dict[k] = _root_grand
            assert osp.exists(_root_dict[k]), _root_dict[k]

    _pred_num = len(_pred)
    if _pred_num == 92: # raw92_xx
        write_txt(osp.join(_root_grand, 'not_plot_92.txt'), _case_name)
        return
    elif _pred_num > 46 + 10: print("!!NOT plot karyotype of {} ichr in:\n {}, ".format(_pred_num, fp_src)); return
    _need_plt = True # False if (plt_t == 0 or plt_t == SKIP) else True
    name_dict = {}

    ## 111 save individual prediction for cam plot
    fns_src, fns_chw = [''] * _pred_num, [''] * _pred_num
    if _each:
        for i in range(_pred_num):  # save individual prediction
            _lb = get_label(_fns[i]) # xx_idfn_str, _lb, xx_tag_p0 = _get_fn_id(_fns[i])
            _src = osp.join(fp_src, _fns[i])
            assert osp.exists(_src), "!!src NOT found: {}".format(_src)
            _fn = str_replace(_fns[i])
            fns_src[i] = f"ky_={_pred[i]}_lb={_lb}_src_fn={_fn}" # source file
            fns_chw[i] = f"ky_={_pred[i]}_lb={_lb}_chw_fn={_fn}" # x_bchw from transform()

            pil_src = Image.open(_src)
            pil_src.save(osp.join(_fp_case_dict['src'], fns_src[i]))
            # imshow_with_lable([pil_src], [sp0_hwc[i]], [fn_pred], flag_l=[flag1], tag=_case_name, fp=_fp_case_dict['src'])

            imgChw = X1_q[i].cpu().numpy() if isinstance(X1_q[i], Tensor) else X1_q[i] # imgChw: x_bchw from transform()
            _hwC = np.uint8(imgChw).transpose((1,2,0)) # *255
            # b,g,r = cv2.split(_hwC); _bgR = cv2.merge((b,g,r))
            pil_Chw = Image.fromarray(_hwC).resize(pil_src.size, Image.Resampling.BILINEAR) # Image.ANTIALIAS #  .convert(mode)
            if QUICK: pil_Chw.save(osp.join(_fp_case_dict['chw'], fns_chw[i]))
            # imshow_with_lable([imgChw], [sp0_hwc[i]], [fn_pred], flag_l=[flag1], tag=_case_name, fp=_fp_case_dict['chw'])

    # 222 save a whole karyotype image
    tag = "{}_karyotype".format(_case_name)
    if _tail_: tag = f"{tag}_{_tail_.lstrip('_')}"
    row, col = 5, 5 * 2 + 5  # 5 pairs and 4 gaps
    row_err, pos_err = row + 1, row * col + 1
    pos_list, _sit, _str_debug, i = [0] * (row_err*col+1), np.zeros(num_cls, dtype=int), "", 0
    # # if _need_plt:
    # plt.rcParams['figure.figsize'] =  (20, 18)# (16, 12)
    # # fig.canvas.set_window_title(tag)
    plt.figure(figsize=(20,16))
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    # plt.subplots_adjust(top=0.98, bottom=0.12, right=0.9, left=0.1, hspace=0, wspace=0)
    # plt.margins(0, 0)            
    _id1 = -1
    for i in range(_pred_num): # plot on a whole  karyotype image
        _id1 = i + 1; _lb = _pred[i]
        # if _id1 == 13: input("wow...13\n")
        _, _idfn_str = get_label(_fns[i], b_str=True) # _idfn_str, xx_p0, xx_tag_p0 = _get_fn_id(_fns[i])
        _idfn_str = str_replace(_fns[i]); _idfn_str = _idfn_str.replace('.jpg', '')
        # fn_pred = "{}_lb__{}_fn_{}".format(_case_name, _lb, _fns[i]) # _fn)
        name_dict["No.{:2}_idfn={:2}_lb={}".format(_id1, _idfn_str, _lb)] = _fns[i]

        _src = osp.join(fp_src, _fns[i])
        img_pil_wh = Image.open(_src)
        if sp0_hwc is not None:
            _method = Image.BILINEAR # if '_bil_' in img_dir else Image.ADAPTIVE
            _sp_wh = sp0_hwc[i]
            if len(_sp_wh) == 3: _sp_wh = (_sp_wh[1], _sp_wh[0])
            # img_plt, img_pil_wh, Chw_shape, hwC_shape, mode, cmap = get_plt_pil(imgXq, _sp_wh)
            img_pil_wh = img_pil_wh.resize((_sp_wh), _method)
        r, c = np.divmod(_lb, row)  # r == 0 is the first row
        if _sit[_lb] > 2: pos = -1 # move to the last row
        else:
            pos = r * col + c * 3 + 1 + _sit[_lb] # _lb '+1' for [1,24] instead of [0,23]
            _sit[_lb] += 1  # mark the same class
        _str_debug += "class={}, r={}, c={}, pos={}\n".format(_lb, r, c, pos)
        if pos < 1 or pos > row * col: # exceeded to the last row
            # print("!!!row={}, col={}, total={}, pos={}\n{}".format(row, col, row*col, pos, _str_debug))
            pos, pos_err = pos_err, pos_err + 1
            print("!lb={}, row={}, col={}, total={}, pos={}".format(_lb, row, col, row * col, pos))
            # if r in ['quit']: raise ValueError("886")
            # break
        if pos >= len(pos_list):
            if ignore_pos: continue
            r = input("~!! bad model... inut 'c' to ingnore pos {} and continue, 'all' to ignore all,"
                      " others to quit\n".format(pos))
            if r in ['all', 'ALL', 'All']: ignore_pos = True; continue
            elif r in ['c', 'C']: continue # can NOT plot
            else: raise ValueError("886")

        pos_list[pos] += 1
        plt.subplot(row_err, col, pos) # ax=fig.add_subplot(projection='3d')
        plt.xlabel(f"{_lb}+1_{pos}")
        plt.ylabel(f"No.{_id1}_id:{_idfn_str}", fontsize=12)
        plt.xticks([]); plt.yticks([])
        # plt.axis('off')
        # plt.title("{}+1".format(_lb)) # 1 to 24  # y=-0.1
        # plt.ylabel(str(_lb == lb_int)) # if (FN1_q[_find_q] == fn_pred_d[i]) # predict name
        plt.imshow(img_pil_wh)  # , cmap=cmap, vmin=img_plt.min(), vmax=img_plt.max())
    tag = "{}{}plt{}".format(tag, _pred_num, _id1)
    plt.suptitle(tag, fontsize='xx-large', y=0.92) # 0.2)
    _fnp_save = osp.join(_root_dict[_keys[0]], "{}.jpg".format(tag))
    plt.savefig(_fnp_save)
    plt.savefig(_fnp_save.replace('.jpg', '_2.pdf'), dpi=200)
    with open(osp.join(_root_dict[_keys[0]], '{}_name_dict.txt'.format(tag[:tag.index('_karyotype')])), 'w') as fw:
        for k,v in name_dict.items():
            fw.writelines("{}:\t{}\n".format(k, v))
    assert max(pos_list) < 2, "!!duplicated: {}, \n{}\n {}".format(max(pos_list), pos_list, _fnp_save)
    if plt_t == -1:
        plt.show()
    elif plt_t > 0:
        plt.pause(plt_t)
    # plt.clf() # !! will always display 'fig' blank window
    plt.close()
    return _root_dict, _keys, _fnp_save
# end plt_one_pred_karyotype()