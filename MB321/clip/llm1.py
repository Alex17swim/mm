# llm1.py, Chengyu Wang, XJTLU, 2023-1005, last update on 2023-1102

import clip
# from torch import tensor # optim
from tqdm import tqdm
# import torch, os
# from torch.utils.data import DataLoader, Dataset
# from torchvision import models
#
# from PIL import Image
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# import os.path as osp
# from chr.data import ChrDataset, set_transform
from MB321.base.cnn321 import *
from MB321.base.metric321 import get_confusion_matrix, cm_axis_name, show_confMat
from llm_resnet import run_llm_resnet

def toyCLIP():
    # dev = "cuda" if torch.cuda.is_available() else "cpu"
    net_clip, preprocess = clip.load("RN50", device=dev)  # ViT-B/32

    image = preprocess(Image.open("9__1a.jpg")).unsqueeze(0).to(dev)  # CLIP.png
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(dev)

    batch_size = 32  # agDATA.batch_cls
    print(f"batch_size={batch_size}")

    from torch.utils.data import Dataset
    class RandomDataset(Dataset):
        def __init__(self, size, length):
            self.len = length
            self.data = torch.randn(length, size)
        def __getitem__(self, index): return self.data[index]
        def __len__(self): return self.len

    tsfm_tra = set_transform(True)
    tsfm_val = set_transform(False)
    ds_train = ChrDataset(agDIR.tra_cls, tsfm_tra)
    ds_val = ChrDataset(agDIR.val_cls, tsfm_val)  # RandomDataset(224, 100) #
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    # ds_val = dataset=RandomDataset(224, 100)
    # dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    with torch.no_grad():
        # for step, (x, lb) in tqdm(enumerate(dl_train), ncols=100):
        t1 = datetime.datetime.now()
        print(f"start validating at {time_str()}...")

        # for step, (image, lb) in enumerate(dl_val, 0):
        #     image = image.to(dev)
        for images in dl_val:  # tqdm():
            image_features = net_clip.encode_image(image)
            text_features = net_clip.encode_text(text)

            logits_per_image, logits_per_text = net_clip(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            print("Label probs:", probs)
    print(f" toyCLIP() done at {time_str()}, spent {time_gap(t1)}")
# end toyCLIP

def fun1(fnp_pth, dev, ds_tra, ds_val, _find_in_last='', bb_clip=agNET.bb_clip):
    bb = agNET.bb_cls
    num_cls = agDATA.num_cls
    batch_size = agDATA.batch_cls # 64
    epoch_max = agDATA.epoch_cls
    # max_iter = agNET.iter # 100
    write_txt([agLOG.val_cls], f" fun1 start with model:{bb}({bb_clip}), device:{dev}, attn:{agNET.attn}, clip_dim method:{agNET.clip_dim} at {TM}\n {TXT_FLAG_ALL}")
    # Load the model
    net_clip, preprocess = clip.load(bb_clip, dev)
    def get_clip_features(dataset, batch_size):
        clip_features, clip_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=batch_size, num_workers=2)):
                features = net_clip.encode_image(images.to(dev))
                clip_features.append(features)
                clip_labels.append(labels)
        return torch.cat(clip_features).cpu().numpy(), torch.cat(clip_labels).cpu().numpy()

    # Calculate the image features
    fnp_cf_tra, fnp_cf_val = osp.join(agDIR.last_cls, f'clip_feat_{KF_STR}_train.npy'), osp.join(agDIR.last_cls, f'clip_feat_{KF_STR}_val.npy')
    if agNET.clip_train:
        clip_feat_train, clip_lb_train = get_clip_features(ds_train, batch_size)
        clip_feat_val, clip_lb_val = get_clip_features(ds_val, batch_size) # [b, 1024]
        np.save(fnp_cf_tra, clip_feat_train)
        np.save(fnp_cf_val, clip_feat_val)
        write_txt([agLOG.tra_cls, agLOG.val_cls], f"saved clip features at: {agDIR.last_cls}\n")
        print(f" double save in origin dir")
        _t, _v = osp.join(agDIR.out_cls, f'clip_feat_{KF_STR}_train.npy'), osp.join(
        agDIR.out_cls, f'clip_feat_{KF_STR}_val.npy')
        np.save(_t, clip_feat_train)
        np.save(_v, clip_feat_val)

    # else:
    #     clip_feat_train, clip_feat_val = np.load(fnp_cf_tra), np.load(fnp_cf_val)
    #     write_txt([agLOG.tra_cls, agLOG.val_cls], f"loaded clip features at: {agDIR.last_cls}\n")

    dl_tra = DataLoader(ds_tra, batch_size=batch_size, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=2)

    # input(f" outside: clip_tra={len(clip_feat_train)}, clip_val={len(clip_feat_val)}; dl_tra={len(dl_tra)}, dl_val={len(dl_val)}\n")
    
    _, _, acc_bb_classify_val, _, _ = run_llm_resnet('keep', 'x', fnp_pth, dev, num_cls, fnp_cf_tra, fnp_cf_val, dl_tra, dl_val, agNET.fc_train, epoch_max)

    _tag = f"TM={TM}(gpu{agGPU.gpu_str}), {agTAG.bug_cls}"
    write_txt([agLOG.val_cls, agLOG.log_all_cls], f"*Start validating {_tag} on validating set @ {time_str()}")
    np_lb_pred, lb_batch, acc_atten_val, _, _ = run_llm_resnet(agNET.attn, agNET.clip_dim, fnp_pth, dev, num_cls, fnp_cf_tra, fnp_cf_val, dl_tra, dl_val, agNET.attn_train, epoch_max)
    write_txt([agLOG.val_cls, agLOG.log_all_cls], f"*End validating {_tag}: acc_bb_classify={acc_bb_classify_val:.4f}; acc_{agNET.attn}={acc_atten_val:.4f}({(acc_atten_val-acc_bb_classify_val):.4f})")  # ; bb_acc_clip={bb_acc_clip}

    _out_dir = osp.join(agDIR.out_cls, 'cm_plt')
    log_cm = osp.join(_out_dir, f"cm_log_{TM}.md")
    confusion_mat, _wrong, _total, _bg = get_confusion_matrix(len(cm_axis_name), lb_batch, np_lb_pred, [agLOG.val_cls, agLOG.log_all_cls])
    _recall_list, _preci_list, _correct_list, _acc_mean, _recall_mean, _precision_mean = show_confMat(confusion_mat, cm_axis_name, 'test', -1, _out_dir, _log=log_cm)
    write_txt([agLOG.val_cls, agLOG.log_all_cls], f"_acc_mean={_acc_mean:.4f}, _recall_mean={_recall_mean:.4f}, _precision_mean={_precision_mean:.4f}\n") # precis={_preci_list}, recalls={_recall_list}, corrects={_correct_list}\n
# end fun1

if __name__ == '__main__':
    write_txt([osp.join(agDIR.last_cls, 'xx_update_time.txt')], f" {agTAG.bb_cls}{agTAG.bug_cls} llm1.py (gpu{agGPU.gpu_str}) start at:{TIME_TH})")
    dev = f"cuda:{agGPU.gpu_str}" # , gpu_num, gpu_all = set_gpu(agGPU.gpu_str)
    md = clip.available_models()
    tsfm_tra,tsfm_val = set_transform(True), set_transform(False)
    ds_train, ds_val = ChrDataset(agDIR.tra_cls, tsfm_tra), ChrDataset(agDIR.val_cls, tsfm_val)
    plt_dataset(ds_train, f"{agDIR.tra_cls[agDIR.tra_cls.index('ichr'):].replace('/', ' ')}\n")
    plt_dataset(ds_val, f"{agDIR.val_cls[agDIR.val_cls.index('ichr'):].replace('/',' ')}\n")

    # print(f"op={args.file_op}; lr={args_train.lr}; root={args_data.root}")
    # input(f"name={args_data.name}; root={args_data.root}\n train={args_data.train}\n")
    fnp_pth = train_cnn(dev, ds_train, ds_val) if agNET.bb_train_cls else ''
    fun1(fnp_pth, dev, ds_train, ds_val)
    write_txt([agLOG.val_cls], f" llm1.py (gpu{agGPU.gpu_str}) done at:{TIME_TH}, cost {time_gap(datetime.datetime.strptime(TM, '%m%d_%H%M%S'))}")
    write_txt([osp.join(agDIR.last_cls, 'xx_update_time.txt')], f" {agTAG.bb_cls}{agTAG.bug_cls} llm1.py (gpu{agGPU.gpu_str}) done at:{TIME_TH}, cost {time_gap(datetime.datetime.strptime(TM, '%m%d_%H%M%S'))}")

