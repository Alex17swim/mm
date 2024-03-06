# data321.py, Chengyu Wang, 2023-05026, last update on 2023-1102

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os, time, re # shutil,
from PIL import Image
import torchvision.transforms as T
from MB321.mm_flag import agDATA, agGPU, time_gap

def set_transform(is_train=True):
    size = (agDATA.img_h, agDATA.img_w)
    norm_mean, norm_std = agDATA.mean, agDATA.std
    if is_train:
        tsfm = T.Compose([
            T.Resize(size=size),
            T.RandomCrop(224, padding=8),
            T.ToTensor(),
            T.Normalize(norm_mean, norm_std)
        ])
    else:
        tsfm = T.Compose([
            T.Resize(size=size),
            T.ToTensor(),
            T.Normalize(norm_mean, norm_std)
        ])
    return tsfm

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index): return self.data[index]

    def __len__(self): return self.len

rmb_label = {'1':0, '100': 1}
class RmbDataset(Dataset):
    def __init__(self,data_dir, tsfm ):
        self.label_name = rmb_label
        self.data_info = self.get_data_info(data_dir)
        self.tsfm  = tsfm

    def __getitem__(self, index):
        dir_img, lb = self.data_info[index]
        img = Image.open(dir_img)
        if self.tsfm  is not None:
            img = self.tsfm (img)
        return img, lb

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_data_info(data_dir):
        data_info = list()
        for root, dirs, files in os.walk(data_dir):
            for dir in dirs:
                fn = os.listdir(os.path.join(root, dir))
                img_names = list(filter(lambda x:x.endswith('.jpg'), fn))
                for i in range(len(img_names)):
                    name = img_names[i]
                    fnp = os.path.join(root, dir, name)
                    lb = rmb_label[dir]
                    data_info.append((fnp, int(lb)))
        return data_info

def get_label(name, _sub1=agDATA.LB_SUB1, b_str=False):
    global skip_check_lb
    _sub1, name1 = int(_sub1), name
    if not '__' in name:
        if b_str: return -8, name
        else: return -8
    try:
        name1 = str(os.path.splitext(name)[0])
        _r = re.findall(agDATA.R3_LB_D2, name1) if 'lb__' in name1 else re.findall(agDATA.R1_LB, name1)
        _len = len(_r)
        assert _len >= 1 # , "!! no label found in:{}".format(name1)
        for _r2 in _r[1:]:
            assert _r[0] == _r2, "!!!found {} different like '__[0-9]+' in {}".format(len(_r), name1)
        lb1 = _r.pop()
        # assert name1.count(lb1) == 1, "!!!found {} like '__[0-9]+' in {}".format(name1.count(lb1), name1)
        lb1to24 = int(re.findall(agDATA.R2_LB, str(lb1)).pop())
        int_lb0to23 = lb1to24 + _sub1  # Karyotype from 0 to 23
    except Exception as e:
        # print(e)
        input(f"!! get label error, check {agDATA.R1_LB} for {name1}\n")
        if b_str: return -7, '-7'
        else: return -7

    if int_lb0to23 < 0 and not skip_check_lb:
        r = input("!!!found label value = {} in {}\n input 'skip' to skip, 'c' to continue, others to quit\n".format(int_lb0to23, name1))
        if r in ['skip']: skip_check_lb = True
        elif r not in ['c', 'C']: raise ValueError("886")

    if b_str:
        lb_to_end_without_ext = name1[name1.index(lb1)+2:] # with extend '.jpg' for 'name'
        return int_lb0to23, lb_to_end_without_ext
    return int_lb0to23
# end get_label

class ChrDataset(Dataset): # return img, lb, name, h, w
    def __init__(self, raw_dir, tsfm, _sub1=agDATA.LB_SUB1):
        self.data_info = self.get_data_info(raw_dir, _sub1)
        self.tsfm = tsfm

    def __getitem__(self, index):
        dir_img, lb, name = self.data_info[index]
        img = Image.open(dir_img).convert(agDATA.img_mode)
        h, w = img.size[1], img.size[0]
        if self.tsfm is not None:
            img = self.tsfm (img)
        return img, lb # , name, h, w

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_data_info(raw_dir, _sub1):
        data_info = list()
        print('raw_dir=%s'%raw_dir)
        num_d, num_f, num_f2, num_i, num_i2 = 0, 0, 0, 0, 0

        t0 = time.time()
        fn_all = os.listdir(raw_dir)
        imgs_ = list(filter(lambda x:x.endswith(agDATA.ext),fn_all))
        if imgs_: # images in root
            num_i = len(imgs_)
            print("#dataset processing '{}' images={} in root ...".format(agDATA.ext, num_i), end='')
            for i in range(num_i): # tqdm.trange(num_i):
                name = imgs_[i]
                fnp = os.path.join(raw_dir, name)
                lb = get_label(name,_sub1)
                data_info.append((fnp, lb, name))
        print('done with time {}'.format(time_gap(time.time()-t0)))

        for root, dirs, files in os.walk(raw_dir):
            # print("#dataset processing folder: ", end='')
            for dir in dirs: # tqdm.tqdm(dirs,ncols=20): # ,desc='data folder'
                t1 = time.time()
                fn_d = os.listdir(os.path.join(root, dir))
                num_d += 1
                imgs_d = list(filter(lambda x:x.endswith(agDATA.ext),fn_d))
                l = len(imgs_d)
                num_i2 += l
                num_f2 += (len(fn_d)-l)
                print("#dataset processing '%s' images=%d in folder %s...\t"%(agDATA.ext, l, dir), end='')
                for i in range(l):
                    name = imgs_d[i]
                    fnp = os.path.join(root, dir, name)
                    lb = get_label(name, _sub1)
                    data_info.append((fnp, lb, name))
                print("done with time {}".format(time_gap(time.time()-t1)))
        num_f = len(fn_all) - num_i - num_d
        print("total '{}' images={} with time {}: {} from root, {} from {} folders.".format(
            agDATA.ext, (num_i+num_i2), time_gap(time.time()-t0), num_i, num_i2, num_d))
        if (num_f + num_f2) > 0:
            print("-unknown file={}, {} from root, {} from folder".format((num_f+num_f2), num_f, num_f2))
        if len(data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to '.jpg' images!".format(raw_dir))
        return data_info
# end class ChrDataset()

def get_dataloader(dir_list, tsfm_tra, tsfm_val, batch_size, sub1, num_wk, pin_mem=agGPU.pin):
    _tsfm_dict = {'train': tsfm_tra, 'val': tsfm_val, 'test': tsfm_val}
    assert type(dir_list) in [list, dict], type(dir_list)
    if type(dir_list) is list and len(dir_list) > 1:
        _dl_dict, _len_dict = \
            multi_dataloader(dir_list, tsfm_tra, tsfm_val, batch_size, sub1, num_wk, pin_mem=pin_mem)
    else:
        _dir = dir_list[0] if type(dir_list) is list else dir_list
        _dl_dict, _len_dict = \
            make_dataloader(_dir, _tsfm_dict, batch_size, sub1, num_wk, pin_mem=pin_mem)
    return _dl_dict, _len_dict
# end get_dataloader

def multi_dataloader(dir_list, tsfm_tra, tsfm_val, batch_size, sub1, num_wk, pin_mem=agGPU.pin):
    _dataset_dict = {'train': [], 'val': [], 'test': []}
    _concat_dict = {'train': '', 'val': '', 'test': ''}
    _dl_dict = {'train': '', 'val': '', 'test': ''}
    _len_dict = {'train': 0, 'val': 0, 'test': 0}
    for _dir in dir_list:
        for k, v in _dir.items():
            if k == 'split': continue
            if v is not None:
                _dataset_dict[k].append(ChrDataset(raw_dir=v, tsfm=tsfm_tra, _sub1=sub1))
    for k, v in _concat_dict.items():
        _concat_dict[k] = ConcatDataset(_dataset_dict[k])

    _dl_dict['train'] = DataLoader(dataset=_concat_dict['train'], batch_size=batch_size, shuffle=True,
                                   num_workers=num_wk,
                                   pin_memory=pin_mem, drop_last=True)
    _dl_dict['val'] = DataLoader(dataset=_concat_dict['val'], batch_size=batch_size, shuffle=False, num_workers=num_wk,
                                 pin_memory=pin_mem, drop_last=True)
    _dl_dict['test'] = DataLoader(dataset=_concat_dict['test'], batch_size=batch_size, shuffle=False, num_workers=num_wk,
                                  pin_memory=pin_mem, drop_last=True)
    for k in _len_dict.keys():
        _len_dict[k] = len(_dl_dict[k])

    return _dl_dict, _len_dict
# end multi_dataloader()

def make_dataloader(_dir_dict, _tsfm_dict, batch_size, sub1, num_wk, pin_mem=agGPU.pin):
    _ds_dict = {'train': '', 'val': '', 'test': ''}
    _dl_dict = {'train': '', 'val': '', 'test': ''}
    _len_dict = {'train': '', 'val': '', 'test': ''}
    assert _tsfm_dict.keys() == _ds_dict.keys(), "tsfm={}, ds={}".format(_tsfm_dict.keys(), _ds_dict.keys())
    for k, v in _dir_dict.items():
        if k == 'split': continue
        if v is not None:
            _shuffle = True if k == 'train' else False
            _ds_dict[k] = ChrDataset(raw_dir=v, tsfm=_tsfm_dict[k], _sub1=sub1)
            _dl_dict[k] = DataLoader(dataset=_ds_dict[k], batch_size=batch_size, shuffle=_shuffle, num_workers=num_wk,
                                     pin_memory=pin_mem, drop_last=True)
            _len_dict[k] = len(_ds_dict[k])
            if _len_dict[k] < 1: print("!!!No data in: {}={}".format(k, v))

    return _dl_dict, _len_dict
# end make_dataloader()

def labelme_test():
    # pip install pyqt5
    # pip install labelme
    print("--labelme test done")

if __name__ == '__main__':

    _dir = os.path.join(agDATA.root, 'D2_mchr_bil_data', 'keep', 'raw22')
    # split_raw(_dir)
    # test_dataset()

    # end_this_py()
