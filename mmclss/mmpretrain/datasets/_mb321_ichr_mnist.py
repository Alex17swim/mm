# _mb321_ichr_mnist.py, from 'mnist.py', Chengyu Wang, 2024-01-04,
from typing import Optional # List,
# from mmengine.fileio import join_path
from mmengine.logging import MMLogger
from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset

import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
from MB321.mm_flag import *
from MB321.base.data321 import get_label

# class Resampling():
#     NEAREST = 0
#     BOX = 4
#     BILINEAR = 2
#     HAMMING = 5
#     BICUBIC = 3
#     LANCZOS = 1
output_this, plt_karyo = agDIR.out_cls, agT21.plt_karyo

def sub_str(_fn0, _head=None, _tail=None):
    while _head and _head in _fn0:  # use 'while' for multiple '_head'
        assert type(_head) is str and len(_head) > 0, _head
        _fn1 = _fn0[_fn0.index(_head) + len(_head):]
        _fn0 = _fn1
    else:
        _fn1 = _fn0
    while _tail and _tail in _fn1:
        assert type(_tail) is str and len(_tail) > 0, _tail
        _fn2 = _fn1[: _fn1.rindex(_tail)]  # .index
        _fn1 = _fn2
    else:
        _fn2 = _fn1
    return _fn2

@DATASETS.register_module()
class Ichr24mnist(BaseDataset):
    METAINFO = {'classes': agMM.classes_cls}
    write_txt([agLOG.bug_all], f"## Ichr24mnist METAINFO: agMM.classes_cls={agMM.classes_cls}")
    def __init__(self, data_root: str = '', split: str = 'train', metainfo: Optional[dict] = None, data_prefix: str = '', test_mode: bool = False, **kwargs):
        splits = ['train', 'test', 'val'] # *mb321* add 'val'
        assert split in splits, f"The split must be one of {splits}, but get '{split}'"
        self.split = split if split == 'train' else 'val' # ! 'test'

        # To handle the BC-breaking
        if split == 'train' and test_mode:
            logger = MMLogger.get_current_instance()
            logger.warning('split="train" but test_mode=True. The training set will be used.')

        if not data_root and not data_prefix: raise RuntimeError('Please set ``data_root`` to specify the dataset path')

        super().__init__(
            ann_file='', # doesn't need specify annotation file
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(root=data_prefix),
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self, _ext=agDATA.ext):  # load_annotations()
        split_root = osp.join(self.data_root, self.split)
        if not osp.exists(split_root): input(f" !! split_root NOT exists:{split_root}")
        data_list, info = list(), dict()
        t1 = time.time()  # ; n_d = 0
        for n_d, (root, dirs, files) in enumerate(os.walk(split_root)):  # images in dirs
            _skip_dir = True if (agT21.make1 or (len(dirs) == 0)) else False
            self.case_num = len(dirs)
            for dir in dirs:  # tqdm.tqdm(dirs,ncols=20): # ,desc='data folder'
                if _skip_dir:
                    imgs_d = [dir]; dir = ''
                else:
                    imgs_d = list(filter(lambda x: x.endswith(_ext), os.listdir(os.path.join(root, dir))))
                len_img = len(imgs_d)
                n_d += 1
                if n_d < 3 or n_d % 50 == 0 or n_d == len(dirs):
                    _str = imgs_d[0] if _skip_dir else f"in '{dir}'\twith {len_img}\t'{_ext}' images"
                    print(f"# No.[{n_d}/{self.case_num}]\t{_str}  ...")
                    if n_d > self.case_num: input(f"?? n_d={n_d}, case_num={self.case_num}\n")

                for i in range(len_img):
                    name = imgs_d[i]
                    img_path = fnp = os.path.join(root, dir, name)
                    lb_int = get_label(name)  # !! lb.shape== error, type(lb) is: int
                    # gt_label = np.array(lb_int, dtype=np.int64)
                    # img_path = self.backend.join_path(self.img_prefix, path)
                    info = dict(img_path=img_path, gt_label=lb_int)
                    data_list.append(info)
                    # img_np_hwc = plt.imread(fnp)  # HWC, from cifar.py
                    # img2 = Image.fromarray(img_np_hwc).resize(agDATA.size_wh) # .convert(agDATA.img_mode)  # .ANTIALIAS
                    # img2_np_hwc = np.array(img2)
                    # lb = get_label(name)  # !! lb.shape== error, type(lb) is: int
                    # gt_label = np.array(lb, dtype=np.int64) # !! gt_label.shape==(), type(gt_label) is: numpy.ndarray,
                    # info = {
                    #         'img_path': fnp,
                    #         'img': img2_np_hwc,
                    #         'gt_label': gt_label,
                    #         'case': dir,
                    #         'name': name,
                    #         'img_shape': img2_np_hwc.shape,  # for img_metas in mmcls/apis/test.py
                    #         'ori_shape': img2_np_hwc.shape,
                    #         # !! use img2, else low acc, ands img_np_hwc will error: cannot reshape array of size 27104 into shape (28,20,4)
                    #         'ori_hwc': img_np_hwc.shape,
                    #         'ori_filename': name,
                    #         # 'img_mode': agDATA.img_mode
                    #         # 'img_info': {'filename':osp.join(dir, name)},
                    #         # 'raw_dir': raw_dir, # self.data_prefix,
                    #         }
                    # data_list.append(info)
        return data_list
    # end load_data_list
# end class Ichr24mnist
