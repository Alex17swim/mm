# _mb321_ichr.py
import imp
import os, time
import os.path as osp
import pickle
from typing import Optional, Union # List,

import numpy as np
import torch.distributed as dist

from mmengine.dataset import Compose
from mmengine import fileio
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
# from .categories import IMAGENET_CATEGORIES
from .custom import CustomDataset


from torch import from_numpy
from MB321.mm_flag import *
from MB321.base.util321 import np_div
from MB321.base.data321 import get_label
from MB321.base.plt321 import plt_subplot3
from MB321.base.metric321 import get_confusion_matrix, cm_axis_name, show_confMat # plt_confusion_matrix, 
from MB321.chr.plt_chr import plt_one_pred_karyotype # , img_repair
from PIL import Image
import matplotlib.pyplot as plt

output_this, plt_karyo = agDIR.out_cls, agT21.plt_karyo
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
@DATASETS.register_module()
class Ichr24(CustomDataset):
    IMG_EXTENSIONS = ('.jpg') # , '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = agMM.classes_cls
    def __init__(self,
                 data_root: str = '', split: str = '', data_prefix: Union[str, dict] = '', ann_file: str = '', metainfo: Optional[dict] = None, multi_label: bool = False, _txt=None, **kwargs):
                 # data_root, data_prefix, pipeline, size_wh=(IMG_W, IMG_H), _txt=None, test_mode=False):
        super().__init__(data_root=data_root, data_prefix=data_prefix, ann_file=ann_file, metainfo=metainfo, **kwargs)
        # super(BaseDataset, self).__init__()
        # self.data_prefix = data_prefix # os.path.join(data_prefix, data_type)
        # self.test_mode = test_mode
        # self.pipeline = Compose(pipeline)
        # if type(size_wh) is not tuple or len(size_wh) != 2:
        #     input3(f"!! size_wh({size_wh}) must be tuple (w,h), we set to ({IMG_W}, {IMG_H}) for you\n")
        #     size_wh = (IMG_W, IMG_H)
        ann_file = None # !!! to avoid: self._find_samples()
        self.size_wh = agDATA.size_wh
        self.img_mode = agDATA.img_mode
        self.data_infos = self.load_annotations() # self
        self.log = _txt

    # @abstractmethod # !!
    def load_annotations(self, _ext=agDATA.ext, _type=agT21.abn_data_type):
        _skip_dir = agT21.make1
        raw_dir = osp.join(self.data_prefix, _type).rstrip('_')
        if _type != '': _type = f"_{_type.rstrip('_')}"
        assert osp.exists(raw_dir), f"! not found:{raw_dir}"
        _fns, _dirs, _files, _ld, _lf, _ldf = get_fns(raw_dir)
        if _lf == _ldf: _skip_dir = True
        print(f"_skip_dir={_skip_dir}, loading data_infos into memory by _mb321_ichr.py from:\n {raw_dir}...")
        if osp.basename(self.data_prefix) in ['test', 'val']:  # check number
            _dirs = os.listdir(raw_dir)
            for i in [0, len(_dirs) - 1, int(len(_dirs) / 2)]:  # check 3 dirs for NOT flipped
                _fns = _dirs[i] if _skip_dir else os.listdir(osp.join(raw_dir, _dirs[i]))
                if len(_fns) >= 92: input(f"!! test/val should NOT flipped to 92({len(_fns)}) in:\n{raw_dir}\n")
        elif not osp.basename(self.data_prefix).startswith(('train', 'val', 'test', 'seg_pred')): input(f" ??expect ['train'] in:{self.data_prefix}")
        data_infos, info = list(), dict()
        t1 = time.time() # ; n_d = 0
        for n_d, (root, dirs, files) in enumerate(os.walk(raw_dir)): # images in dirs            
            if _skip_dir: dirs = files
            self.case_num = len(dirs)
            if len(dirs) == 0 and len(files) > 0:
                print1(f"!! dirs=0 and files={len(files)}, check:{root}\nmb321_ichr.py->load_annotations(), line57\n", f"57")
                if 'xx' not in KF_STR: break
                else: continue
            if (self.case_num < 8) and agNET.bb_train_cls and not QUICK: input(f"?? only {self.case_num} cases for test:\n {raw_dir}")

            for dir in dirs: # tqdm.tqdm(dirs,ncols=20): # ,desc='data folder'
                if _skip_dir: imgs_d = [dir]; dir = ''
                else: imgs_d = list(filter(lambda x:x.endswith(_ext), os.listdir(os.path.join(root, dir))))
                len_img = len(imgs_d); n_d += 1
                if n_d < 3 or n_d % 50 == 0 or n_d == len(dirs):
                    _str = imgs_d[0] if _skip_dir else f"in '{dir}'\twith {len_img}\t'{_ext}' images"
                    print(f"# No.[{n_d}/{self.case_num}]\t{_str}  ...")
                    if n_d > self.case_num: input(f"?? n_d={n_d}, case_num={self.case_num}\n")
                for i in range(len_img):
                    name = imgs_d[i]
                    fnp = os.path.join(root, dir, name)
                    img_np_hwc = plt.imread(fnp) # HWC, from cifar.py
                    img2 = Image.fromarray(img_np_hwc).resize(self.size_wh, Image.BILINEAR).convert(agDATA.img_mode) # .ANTIALIAS 
                    img2_np_hwc = np.array(img2)
                    lb = get_label(name) # !! lb.shape== error, type(lb) is: int
                    gt_label = np.array(lb, dtype=np.int64) # !! gt_label.shape== (), type(gt_label) is: numpy.ndarray, 
                    info = {'img': img2_np_hwc,
                            'gt_label':gt_label,
                            'case':dir,
                            'name': name,
                            'img_shape': img2_np_hwc.shape, # for img_metas in mmcls/apis/test.py
                            'ori_shape':img2_np_hwc.shape, # !! use img2, else low acc, ands img_np_hwc will error: cannot reshape array of size 27104 into shape (28,20,4)
                            'ori_hwc':img_np_hwc.shape,
                            'ori_filename': name,
                            'img_mode': self.img_mode
                            # 'img_info': {'filename':osp.join(dir, name)},
                            # 'raw_dir': raw_dir, # self.data_prefix,
                            }
                    data_infos.append(info)
        if ('seg_pred' in raw_dir) == (info['gt_label'] >= 0): input(f"??? label={info['gt_label']} for:{raw_dir}\n") 
        _t = "- expect: init_cfg {'type': 'Normal', 'layer': 'Linear', 'std': 0.01}\n"
        print(f"{_t}ichr load_annotations done '{osp.basename(self.data_prefix)}' with {len(data_infos)}, [0].shape={np.shape(data_infos[0]['img'])} with time {time_gap(t1)}\n")
        fp_out = osp.join(output_this, f"preview_{osp.basename(self.data_prefix)}{_type}")
        plt_subplot3(fp_out, data_infos, _key='img', _title=f"{osp.basename(self.data_prefix)}{_type}, wh={self.size_wh}")
        # os.makedirs(fp_out, exist_ok=True)
        # for i in range(0,3):
        #     img = data_infos[i]['img']
        #     _tag = f"No.{i+1}" # f"No.{i+1}_{data_infos[i]['name']}" # 
        #     plt.imsave(osp.join(fp_out, f"{_tag}.jpg"), img)
        #     plt_subplot(img, 1, 3, (i+1), xlabel=_tag, ylabel=f"{data_infos[i]['gt_label']}")
        #     # plt.subplot(1, 3, (i+1))
        #     # plt.xlabel()
        #     # plt.ylabel()
        #     # plt.xticks([]); plt.yticks([])
        #     # plt.imshow(img)
        # plt.suptitle(f"{osp.basename(self.data_prefix)}{_type}", fontsize=22)
        # plt.pause(1)
        # # plt.savefig(osp.join(fp_out, f"No.{i+1}_{time_str()}.jpg"))
        # plt.clf()
        return data_infos
    # end load_annotations
    def confusion_matrix(self, outputs_pred, json_log=''): # ConfusionMatrixDisplay, confusion_matrix
        have_t21 = True if agT21.abn_ratio > 0. else False
        if self.log and osp.exists(self.log):
            n = osp.splitext(osp.basename(self.log))[0]
            _name = f" log={sub_str(n, 'log_')}\n" # sub_str(a, 'log_mmc_')
        else: _name = f"no name({self.log}) of {TM}"
        if json_log != '':
            js = osp.splitext(osp.basename(json_log))[0]
            _s = '_latest' if '_latest' in js else '_best' if '_best' in js else '_at'
            _str_js = _s if TM in js else js # js[len('eval_mmc_'):js.index(_s) + len(_s)]
            _str_at = sub_str(js, '_at', '_e') # js[js.index('_at')+len('_at'):js.rindex('_e')]
            # input(f" json  ={js}, s={_s}\n str_js={_str_js}\n str_at={_str_at}")
            if TM not in _str_at: _name += f" trained at:{_str_at};"
            _name += f" json={_str_js}\n"
        else: js, _str_js, _str_at = '', '', ''
        write_txt([self.log], f"~~ calculate confusion matrix, log={self.log}\n")
        _out_dir = osp.join(osp.dirname(self.log), 'cm_plt')
        log_cm = osp.join(_out_dir, f"cm_log_{TM}.txt")
        _log_list = [self.log, log_cm, agLOG.log_all_cls]
        assert type(outputs_pred) is list, f"! type={type(outputs_pred)}"
        # to_tensor = (lambda x: from_numpy(x) if isinstance(x, np.ndarray) else x)
        pred = from_numpy(np.array(outputs_pred)) # np.vstack(outputs_pred)
        np_lb_gt = gt_labels = self.get_gt_labels() # numpy.ndarray
        # target = from_numpy(gt_labels)

        maxk = max((1,1))
        num = pred.size(0)
        pred_score, pred_label = pred.topk(maxk, dim=1)
        pred_label = pred_label.t()
        # correct = pred_label.eq(target) # .view(1, -1).expand_as(pred_label))

        # pred_score, pred_label = pred.topk(maxk, dim=1)
        # pred_label = pred_label.t()
        # correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
        # for k in topk:
        #     res_thr = []
        #     for thr in thrs:
        #         # Only prediction values larger than thr are counted as correct
        #         _correct = correct & (pred_score.t() > thr)
        #         correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
        #         if correct_k != correct[:k].reshape(-1).float().sum(0, keepdim=True):
        #             print(f"! pred_score.t() <= thr({thr}) \n{pred_score.t()[:k]}\n")
        #         res_thr.append((correct_k.mul_(100. / num)))

        np_lb_pred = pred_label.detach().cpu().numpy()
        if len(np_lb_pred.shape) > 1: np_lb_pred = np_lb_pred[0]
        confusion_mat, _wrong, _total, _bg = get_confusion_matrix(len(cm_axis_name), np_lb_gt, np_lb_pred, _log_list)
        _recall_list, _preci_list, _correct_list, _acc_mean, _recall_mean, _precision_mean = show_confMat(confusion_mat, cm_axis_name, 'test', -1, _out_dir, _log=log_cm) # _log_list
        # assert abs(acc - _acc_mean) <= 0.000001, "! {}, {}".format(acc, _acc_mean)
        # plt_confusion_matrix(confusion_mat, cm_axis_name, _name, _out_dir, _wrong, _total)

        '''
        # https://wenku.baidu.com/view/db977e5ea16925c52cc58bd63186bceb19e8ed1d.html
        # https://zhuanlan.zhihu.com/p/392767960
         TPR（True Positive Rate，  真阳性率）：就是召回率; 
         TNR（True Negative Rate，  真阴性率）：就是“负类别”的召回率, 预测对的负样本 占 全体负样本 的比例;
         FPR（False Positive Rate， 假阳性率）or FAR（False Acceptance Rate）：预测错的正样本 占 全体正样本 的比例，也叫误识别率、虚警率。假阳性率 = 误识率 = 虚警率。
            假阳性：人本来是阴性，叫你给识别成阳性了；
            误识率：本来是俩人脸，叫你给误匹配（误识）了；
            虚警率：本来是正常情况，叫你给识别成有情况，误报警了
         FNR（False Negative Rate，假阴性率）or FRR（False Reject Rate）：预测错的负样本 占 全体正样本 的比例，也叫拒识率。
        '''
        sum_pred_row, sum_gt_col = np.sum(confusion_mat, axis=0), np.sum(confusion_mat, axis=1)
        _cls = 1 if agT21.lb_t21 else 21-1 # trisomy21; 22(or whatever) for test
        TP21 = int(confusion_mat[_cls, _cls]) # True Positive
        TP_FP = sum_pred_row[_cls] # all prediction
        FP21 = int(TP_FP - TP21) # False Positive
        FN21 = int(sum_gt_col[_cls]) - TP21 # False Negative,
        gt21 = sum_gt_col[_cls] # ground truth = TP21+FN21
        if gt21 < 1 and have_t21: input(f" !! gt21={gt21}\n")
        # negative_all = (confusion_mat.sum() - gt21) # negtive = all - positive
        # TN21 = negative_all - TP21 # True Negative = negtive - TP21
        negative_all = int(confusion_mat.sum() - TP_FP) # negtive = all - (TP21+FP21)
        TN21 = negative_all - FN21 # True Negative = negtive - FN21
        acc21, prc21, rec21 = np_div(TP21 + TN21, TP_FP + FN21 + TN21), np_div(TP21, TP_FP), np_div(TP21, gt21)
        TNR, FPR, FNR = np_div(TN21, TN21+FP21), np_div(FP21, TN21+FP21), np_div(FN21, TP21+FN21)
        _l = f"\n precis={_preci_list}, recalls={_recall_list}, corrects={_correct_list}\n" if agT21.lb_t21 else ''
        str_21 = f" TP21={TP21}, FP21={FP21}, FN21={FN21}, TN21={TN21}\n acc={_acc_mean:.4f}, prc={_precision_mean:.4f}, rec={_recall_mean:.4f};  acc21={acc21}, prc21={prc21}, rec21(TPR)={rec21}, TNR={TNR}, FPR(1-TNR)={FPR}, FNR(1-TPR)={FNR}{_l}\n"

        case_dict = {} # {[gt, pred]}
        for i, info in enumerate(self.data_infos):
            c = info['name'][:info['name'].index('__')] # case__lb.jpg
            # lb = info['name'][:]
            if c not in case_dict.keys(): case_dict[c] = [0, 0]
            if c in case_dict.keys():
                if np_lb_gt[i] == _cls: case_dict[c][0] += 1
                if np_lb_pred[i] == _cls: case_dict[c][1] += 1

        t21_gt, t21_pred, tp, fp, fn, tn = [], [], 0, 0, 0, 0

        if agT21.make1: # _cls == 1: #
            for k, v in case_dict.items():
                if v[0] == 1: t21_gt.append(k)
                if v[1] == 1: t21_pred.append(k)

                if   v[0] == 1 and v[1] == 1: tp += 1
                elif v[0] == 0 and v[1] == 1: fp += 1
                elif v[0] == 1 and v[1] == 0: fn += 1
                elif v[0] == 0 and v[1] == 0: tn += 1
        else: # 21-1
            for k, v in case_dict.items():
                if v[0] < 2 and not QUICK: input(f"!! No.21 class in case {k} is:{v[0]}\n")
                if v[0] > 2: t21_gt.append(k)
                if v[1] > 2: t21_pred.append(k)

                if   v[0] > 2  and v[1] >  2: tp += 1
                elif v[0] <= 2 and v[1] >  2: fp += 1
                elif v[0] > 2  and v[1] <= 2: fn += 1
                elif v[0] <= 2 and v[1] <= 2: tn += 1
                else: input(f" ?? name={k}, gt={v[0]}, pred={v[1]}\n")
        # if tn > 0: input("?? expect tn({tn})=0 here\n")
        # tn += len(self.data_infos) - len(case_dict)
        DSa, DSp, DSr = np_div(tp+tn, tp+fp+tn+fn), np_div(tp, tp+fp), np_div(tp, tp+fn)
        tnr, fpr, fnr = np_div(tn, tn+fp), np_div(fp, tn+fp), np_div(fn, tp+fn)
        str_ds = f" DS: tp={tp}, fp={fp}, fn={fn}, tn={tn};\n DSa={DSa}, DSp={DSp}, DSr(tpr)={DSr}; tnr={tnr}, fpr(1-tnr)={fpr}, fnr(1-tpr)={fnr}; ({len(case_dict)})case_dict={case_dict}\n"

        str_mean = f"---- &{_acc_mean:<.4f} &{_precision_mean:<.4f} &{_recall_mean:<.4f}  &{prc21:<.4f} &{rec21:<.4f} &{TNR:<.4f}  &{DSp:<.4f} &{DSr:<.4f} &{tnr:<.4f}"
        write_txt(_log_list, f"{'===='}\n{_name}{str_21}{str_ds}{str_mean} of: acc, preci, recall;  prc21, TPR21, TNR21;  DSpreci DStpr DStnr\n")
        if (np.isnan(rec21) or np.isnan(DSr)) and (have_t21 and 'xx' in KF_STR): input(f"!!! rec21={rec21}, DSr={DSr}\n")
        print(" confusion_matrix done from ichr.py")
        _s = f" 21: TP{TP21} FP{FP21} TN{TN21} FN{FN21};  DS: tp{tp} fp{fp} tn{tn} fn{fn}"
        _cm_str_for_final_print = f"{str_mean}\n{_s}\n Trained at:{_str_at}, .json={_str_js}\n"
        return _cm_str_for_final_print
    # end confusion_matrix

    def plt_karyo(self, outputs_pred, T21=True, FAC=agT21.fac): # FAC to deside if is a T21
        if plt_karyo < 1: print(f"skip plot karyotype due to plt_karyo={plt_karyo}"); return ''
        print(f" ploting karyotype from:{self.data_prefix}")
        T21_names = None
        cell_tp, cell_fp, cell_fn, cell_tn, cell_dicts = 0, 0, 0, 0, {'T21': [], 'norm': []}
        case_tp, case_fp, case_fn, case_tn, singleton_dicts = 0, 0, 0, 0, {}
        if T21 == True or (isinstance(T21, list) and len(T21) > 0):
            if T21 == True:
                T21_names, fnp_t21 = '', osp.join(osp.dirname(self.data_prefix), 'T21_names.txt')
                if osp.exists(fnp_t21):
                    print(f"fnp_t21={fnp_t21}")
                    with open(fnp_t21) as fr:
                        T21_names = fr.readlines()
                    for s in ['\n', ',']: T21_names = [x.rstrip(s) for x in T21_names]
                    # input(f"T21_names={T21_names}, type={type(T21_names)}")
                    T21_names = list(set(T21_names))
                    T21_names.sort()
                else:
                    print1(f" going to use default T21_names = ['SZ2200222', 'SZ1800639']\n")
                    T21_names = ['SZ2200222', 'SZ1800639']
            else: assert len(T21) > 0, f"?? T21={T21}"
            print(f" going to check T21_names={T21_names}")
        cases_dict, fp_src = {}, self.data_prefix
        _fns, _dirs, _files, _ld, _lf, _ldf = get_fns(fp_src)
        if _ld != _ldf:
            input(f"!! expect all ({_ld})dirs, found {_lf} files:\n{fp_src}\n press anykey to continue\n")
            return ''
        for i, data in enumerate(self.data_infos):
            case, fn, sp0_hwc = data['case'], data['name'], data['ori_hwc'] # img_shape
            if case not in cases_dict.keys(): cases_dict[case] = {'np_pred':[], 'bchw':[], 'fns':[],'sp0_hwc':[]}
            np_pred = outputs_pred[i].argmax() # if isinstance(outputs_pred[i], np.ndarray) else # .cpu().numpy()
            cases_dict[case]['np_pred'].append(np_pred)
            np_hwc = data['img']
            cases_dict[case]['bchw'].append(np.transpose(np_hwc, (2,0,1)))
            cases_dict[case]['fns'].append(fn) # case + fn
            cases_dict[case]['sp0_hwc'].append(sp0_hwc)
        for k, v in cases_dict.items():
            fp_in_case = osp.join(fp_src, k)
            fp_out_case = osp.join(output_this, f"karyo_{osp.basename(self.data_prefix)}", k)
            f0 = v['fns'][0] # v['np_pred'] = [1,2,3]
            if plt_karyo == 2: plt_one_pred_karyotype(v['np_pred'], v['bchw'], v['fns'], fp_out_case, fp_in_case, v['sp0_hwc'], _tail_=f"{f0[f0.index('_'):-len('.jpg')]}")

            # # # calculate T21 with both cell and case
            if T21_names is None or plt_karyo < 1: return ''
            # print(f"f0={f0}, cell_name={k}") # f0=14_pred_roat_fit.jpg, cell_name=case654_of_s2_c10_image0_met
            name = k[:k.index('_')] if '_' in k else k
            if name not in singleton_dicts.keys(): singleton_dicts[name] = {'T21':0, 'norm':0, 'fac':0.0, 'is':'?', 'all':0}
            # input(f"name={name}, pred={v['np_pred']}") # name=case654, pred=[20, 20, ...]
            cell_t21 = True if v['np_pred'].count(21-1) > 2 else False
            if cell_t21: # positive prediction
                singleton_dicts[name]['T21'] += 1
                cell_dicts['T21'].append(k)
                if name in T21_names: cell_tp += 1
                else: cell_fp += 1
            else: # negative prediction
                singleton_dicts[name]['norm'] += 1
                cell_dicts['norm'].append(k)
                if name in T21_names: cell_fn += 1
                else: cell_tn += 1
        pred_tp_fp, gt = cell_tp + cell_fp, cell_tp + cell_fn
        acc, prc, rec = np_div(cell_tp + cell_tn, pred_tp_fp + cell_fn + cell_tn), np_div(cell_tp, pred_tp_fp), np_div(cell_tp, gt)
        tnr, fpr, fnr = np_div(cell_tn, cell_tn+cell_fp), np_div(cell_fp, cell_tn+cell_fp), np_div(cell_fn, gt)
        ctT, ctF = 0, 0
        for k, v in singleton_dicts.items():
            v['fac'] = np_div(v['norm'], v['T21'])
            v['all'] = v['norm'] + v['T21']
            if v['fac'] < FAC: # T21
                v['is'] = True; ctT += 1;
                if k in T21_names: case_tp += 1
                else: case_fp += 1
            else:
                v['is'] = False; ctF += 1
                if k in T21_names: case_fn += 1
                else: case_tn += 1
        pred_TP_FP, GT = case_tp + case_fp, case_tp + case_fn
        ACC, PRC, REC = np_div(case_tp + case_tn, pred_TP_FP + case_fn + case_tn), np_div(case_tp, pred_TP_FP), np_div(case_tp, GT)
        TNR, FPR, FNR = np_div(case_tn, case_tn + case_fp), np_div(case_fp, case_tn + case_fp), np_div(case_fn, GT)

        d1,d2 = f"cell_dicts(len={len(cell_dicts['T21'])+len(cell_dicts['norm'])})", f"singleton_dicts(len={len(singleton_dicts)},T={ctT}, F={ctF})"
        str_c = f"cell(tp,fp,fn,tn;all)={cell_tp,cell_fp,cell_fn,cell_tn,(cell_tp+cell_fp+cell_fn+cell_tn)}; acc,preci,recall(tpr),tnr,fpr,fnr={acc, prc, rec, tnr, fpr, fnr}"
        str_s = f"single(TP,FP,FN,TN;ALL)={case_tp,case_fp,case_fn,case_tn,(case_tp+case_fp+case_fn+case_tn)}; ACC,PRECI,RECALL(TPR),TNR,FPR, FNR={ACC, PRC, REC, TNR, FPR, FNR}"
        str1 = f"{str_c}\n{str_s}\nDS fac={FAC}; {d1}; {d2}\n"
        t21_str = f"{str1}\n\n{d1}=\n{cell_dicts}\n\n{d2}=\n{singleton_dicts}"
        print(f"t21_str={str_c if (cell_tp + cell_fp + cell_fn + cell_tn) > 10 else t21_str}")
        fnp_t21 = osp.join(output_this, f"t21_results_{TM}_{agNET.bb_cls}.txt") # [len('mmc_'):]
        with open(fnp_t21, 'a') as fw:
            fw.writelines(t21_str)
        print(f" t21 out to: {fnp_t21}")
        return str1