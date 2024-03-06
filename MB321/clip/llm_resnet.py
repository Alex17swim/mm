# llm_resnet.py, Chengyu Wang, 2023-1005, last update on 2023-1102
import sys, os
_import_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_import_root)
# print("llm_resnet.py _import_root={}".format(_import_root))
                
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import torch.utils.tensorboard as tb
from math import sqrt
from tqdm import tqdm
import matplotlib
matplotlib.use('TKagg') # for UserWarning: Matplotlib is currently using agg
# from torch import optim
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import os, random, tqdm, re
# import os.path as osp
# from PIL import Image
from MB321.mm_flag import *
from MB321.base.util321 import *
from MB321.base.cnn321 import reload_net, get_criterion, get_optimizer
# from MB321.data321 import set_transform, ChrDataset, RmbDataset # , RandomDataset

class RN50Features(models.resnet.ResNet):
    def __init__(self, *args, **kwargs):
        super(RN50Features, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3], *args, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x  # 返回的是features，而不是logits
# end RN50Features

import torch
from torch import Tensor
import torch.nn as nn
# from .._internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
try: from torch.hub import load_state_dict_from_url
except ImportError: from torch.utils.model_zoo import load_url as load_state_dict_from_url
__all__ = ['LLM_ResNet'] #, 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def _get_activation_fn(activation=ACT):
    """Return an activation function given a string"""
    if activation == 'relu': # nn.modules.activation.ReLU
        return nn.functional.relu
    elif activation == 'gelu':
        return nn.functional.gelu
    elif activation == "glu":
        return nn.functional.glu
    elif activation == 'sigmoid':
        return nn.functional.sigmoid
    elif activation == 'sm':
        # print(f"!! you are choosing 'softmax', maybe -20% performance \n\n")
        return nn.functional.softmax
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
def get_clip_net(bb_tag, num_cls, attn, cd, _prt=False):
    _log = [agLOG.tra_cls, agLOG.val_cls] if agNET.clip_train else [agLOG.val_cls]
    write_txt(_log, f"!! this is a self-designed model for combining with CLIP which will reload .pth from trained LLM_ResNet !!\n")
    if 'rn50' in bb_tag.lower():
        # net = models.resnet50(pretrained=False, num_classes=num_cls)
        net = LLM_resnet50(pretrained=False, num_classes=num_cls, attn=attn, cd=cd)
    elif 'next50' in bb_tag.lower():
        net = LLM_next50(pretrained=False, num_classes=num_cls, attn=attn, cd=cd)
    else: input3(f"!!unexpect backbone tag:{bb_tag}(bb_cls={agNET.bb_cls})"); net = None
    if _prt: summary(net, (3,224,224), -1, 'cpu')
    return net

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride,),
                     padding=dilation, groups=groups, bias=False, dilation=(dilation,))


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=(stride,), bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LLM_ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000, zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64, replace_stride_with_dilation: Optional[List[bool]] = None, norm_layer: Optional[Callable[..., nn.Module]] = None, attn=agNET.attn, cd=agNET.clip_dim) -> None:
        super(LLM_ResNet, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=(7,7), stride=(2,), padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.attn, self.cd = attn, cd
        if 'keep' not in self.attn:
            # self-attention
            input_dim, attn_drop, proj_drop = 1024, DROP, DROP
            if '2048' in self.cd:
                input_dim = 2048
                self.fc2048 = nn.Linear(1024, input_dim)
            else: self.fc1024 = nn.Linear(2048, input_dim)

            self.qkv = nn.Linear(input_dim, input_dim * 3)
            self._norm_fact = 1 / sqrt(input_dim)

            # cross-attention
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(input_dim, input_dim)
            self.proj_drop = nn.Dropout(proj_drop)

            self.act = _get_activation_fn()

            # emb_dim, dim_c, dim_bb, dim_out = 256, 1024, 2048, num_classes
            # self.fc2048 = nn.Linear(dim_c, dim_bb) # clip
            # self.activation = _get_activation_fn()
            # self.clip2mask = nn.Sigmoid()
            # self.toQ = nn.Linear(dim_bb, emb_dim)
            # self.toK = nn.Linear(dim_c, emb_dim)
            # self.toV= nn.Linear(dim_c, emb_dim)
            # self.cls = nn.Linear(emb_dim, dim_out)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, feat_clip: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # # feat_clip1 for alex
        # if 'repeat' in self.cd: # [b, 1024] -> [b, 2048]
        #     feat_clip1 = feat_clip.repeat(1,2)
        # elif 'cross' in self.attn:
        #     feat_clip1 = feat_clip
        # else: # elif 'fc2048' in self.cd
        #     feat_clip1 = self.fc2048(feat_clip.float())

        if 'keep' in self.attn:
            x = x
        elif 'add' in self.attn:
            if x.shape[1] == feat_clip.shape[1]: x = x + feat_clip
            else: x = x + self.fc2048(feat_clip.float())
        # elif 'self' in self.attn:
        #     K = self.clip2mask(feat_clip1) # [b, dim_bb] = [b,2048]
        #     KQ = K * x # [b, dim_bb] = [b,2048]
        #     AA = self.activation(KQ)
        #     x = VKQ = AA
        # elif 'cross' in self.attn:
        #     bb_query, clip_kv = x, feat_clip1
        #     clip_kv = clip_kv.float()
        #     b = batch_size = bb_query.shape[0]
        #     # key, value = clip_kv, clip_kv
        #     query = self.toQ(bb_query) # [b, emb_dim]; 2048 ->  256
        #     key = self.toK(clip_kv) # [b, emb_dim]; 1024 -> 256
        #     value = self.toV(clip_kv) # [b, emb_dim]; 1024 -> 256
        #     Q = query = query.view(batch_size, int(np.sqrt(query.shape[1])),-1)
        #     K = key = key.view(batch_size, int(np.sqrt(key.shape[1])),-1)
        #     V = value = key.view(batch_size, int(np.sqrt(value.shape[1])),-1)

        #     # acc=1.0, loss=0.0003
        #     AA = self.activation(query*key)  # [b, emb_dim]
        #     out = torch.matmul(value, AA)  # [b, emb_dim]
        #     out = out.view(batch_size, -1)
        #     out = self.cls(out)
        #     return x
        # else: input3(f"!!unexpected default attension of {'self'}\n")
        else:
            if '2048' in self.cd:
                x1, x2 = x, self.fc2048(feat_clip.float())
            else:
                x1, x2 = self.fc1024(x), feat_clip.float()
            # Q = self.q(x)  # Q: batch_size * seq_len * dim_k
            # K = self.k(x)  # K: batch_size * seq_len * dim_k
            # V = self.v(x)  # V: batch_size * seq_len * dim_v
            B, C = x1.shape

            qkv1 = self.qkv(x1).reshape(B, 3, -1).permute(1, 0, 2)
                # .permute(2, 0, 3, 1)
            q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

            qkv2 = self.qkv(x2).reshape(B, 3, -1).permute(1, 0, 2)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

            attn1 = (q1 @ k1.transpose(-2, -1))
            attn2 = (q2 @ k2.transpose(-2, -1))

            attn1 = self.act(attn1) * self._norm_fact # attn1.softmax(dim=-1) * self._norm_fact
            attn2 = self.act(attn2) * self._norm_fact # attn2.softmax(dim=-1) * self._norm_fact

            attn1 = self.attn_drop(attn1)
            attn2 = self.attn_drop(attn2)

            # self
            attn12 = (q1 @ k2.transpose(-2, -1))
            attn12 = self.act(attn12) * self._norm_fact # attn12.softmax(dim=-1) * self._norm_fact
            attn12 = self.attn_drop(attn12)
            x1_attn = attn1 @ v1
            x12_attn = attn12 @ v2
            x1_proj = self.proj(x1_attn + x1)
            x12_proj = self.proj(x12_attn + x1)
            out1 = x1 + x1_attn + x12_attn + x1_proj + x12_proj

            # cross
            attn21 = (q2 @ k1.transpose(-2, -1))
            attn21 = self.act(attn21) * self._norm_fact # attn21.softmax(dim=-1) * self._norm_fact
            attn21 = self.attn_drop(attn21)
            x2_attn = attn2 @ v2
            x21_attn = attn21 @ v1
            x2_proj = self.proj(x2_attn + x2)
            x21_proj = self.proj(x21_attn + x2)
            out2 = x2 + x2_attn + x21_attn + x2_proj + x21_proj

            if 'self' in self.attn:
                return out1
            elif 'cross' in self.attn:
                if 'both' in self.cd: return out1+out2
                else: return out2
            # else: input3(f"!!unexpected attn: {self.attn}")
        # end else:
        x = self.fc(x)        
        return x

    def forward(self, x: Tensor, feat_clip: Tensor) -> Tensor:
        return self._forward_impl(x, feat_clip)
# end LLM_ResNet

def _llm_resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], pretrained: bool, progress: bool, **kwargs: Any) -> LLM_ResNet:
    model = LLM_ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        if 'fc.weight' in state_dict and agDATA.num_cls != state_dict['fc.weight'].shape[0]:
            del state_dict['fc.weight']
            del state_dict['fc.bias']
        model.load_state_dict(state_dict, strict=False)
    return model

def LLM_resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LLM_ResNet:
    return _llm_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def LLM_next50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LLM_ResNet: # resnext50_32x4d
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _llm_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def run_llm_resnet(attn, cd, fnp_pth, dev, output_cls, fnp_cf_tra, fnp_cf_val, dl_tra, dl_val, is_train=True, epoch_max=agDATA.epoch_cls):
    feat_tra, feat_val = np.load(fnp_cf_tra), np.load(fnp_cf_val)
    write_txt([agLOG.tra_cls, agLOG.val_cls], f"loaded clip features at: {agDIR.last_cls}\n")
    # input(f" inside: clip_tra={len(feat_tra)}, clip_val={len(feat_val)}; dl_tra={len(dl_tra)}, dl_val={len(dl_val)}\n")

    bb_tag = agTAG.bb_cls # use tag to indicate k-fold
    clip_bb_tag = bb_tag.replace('bb','b') # to distinguish from train_cnn[]
    num_cls, epoch_max, batch_size = agDATA.num_cls, agDATA.epoch_cls, agDATA.batch_cls # 64
    optimizer, learning_rate, weight_decay, momentum = agNET.opti, agNET.lr, agNET.wd, agNET.mome

    tb_writer = tb.SummaryWriter(log_dir=agDIR.out_tb_cls, comment='clip_atten', filename_suffix='tb')
    # attn, cd, tag = agNET.attn, agNET.clip_dim, 'SelfAttn'
    fp_pth_last, fp_pth = agDIR.last_cls, agDIR.out_pth_cls,
    if isinstance(feat_tra, np.ndarray): feat_tra = torch.from_numpy(feat_tra).to(dev)
    if isinstance(feat_val, np.ndarray): feat_val = torch.from_numpy(feat_val).to(dev)
    # input(f" inside 2: clip_tra={len(feat_tra)}, clip_val={len(feat_val)}; dl_tra={len(dl_tra)}, dl_val={len(dl_val)}\n")
    acc, loss, e, ii, t1 = 0.0, 0.0, -1, 0, time.time()
    # LLM_ResNet trained
    net = get_clip_net(clip_bb_tag, num_cls=output_cls, attn=attn, cd=cd)
    # net = net.to(dev) # to(dev) in reload_net()
    this_bb = f"{attn}_{cd}_{clip_bb_tag}"
    tag_this_bb = f"{this_bb}{agTAG.bug_cls}" # _{ACT} is NOT used for search and reload .pth
    write_txt([agLOG.tra_cls, agLOG.val_cls], f"* {agNET.bb_cls}({tag_this_bb}) gpu={dev}reloaded_net from: {fnp_pth}")
    if is_train:
        write_txt([agLOG.tra_cls], f"*{tag_this_bb} reload from trained ResNet and start training for run_llm_resnet at {time_str()}")
        net, acc, e, _ = reload_net(net, dev, fnp_pth, log=agLOG.tra_cls, _find=bb_tag)
        criterion = get_criterion()
        opt, sch = get_optimizer(net.parameters(), opti=optimizer, lr=learning_rate, wd=weight_decay, mome=momentum)
        for e in range(1, epoch_max+1):
            net.train()
            yhat_batch, lb_batch, p = [], [], 0
            for i, (images, lb) in enumerate(dl_tra):
                # input(f"size of images, lb, feat = {images.shape}, {lb.shape}, {feat_tra[p:p+len(images)].shape}\n")
                images, lb = images.to(dev), lb.to(dev)
                y_ = net(images, feat_tra[p:p+len(images)])
                p += len(images)
                loss = criterion(y_.squeeze(), lb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ii = ii+1
                yhat_batch.append(y_.argmax(dim=1).detach())
                lb_batch.append(lb.detach())
                if i <= 5 or i % 20 == 0 or i==(len(dl_tra)-1) or (e == epoch_max and i == 1):
                    loss, _lr = loss.cpu().detach().numpy(), opt.param_groups[0]['lr']
                    _s = f" train e[{e}/{epoch_max}], iteration[{i+1}/{len(dl_tra)}]({ii}), acc={acc:.4f}, loss={loss:.4f}, lr={_lr:.5f} {tag_this_bb}"
                    if np.isnan(loss): input3(f"!! loss={loss}\n{_s}")
                    # y_, lb = torch.cat(yhat_batch).cpu().numpy(), torch.cat(lb_batch).cpu().numpy()
                    y_, lb = y_.argmax(dim=1).detach().cpu().numpy(), lb.detach().cpu().numpy()
                    correct = sum(y_ == lb)
                    acc = correct.item()/lb.shape[0]
                    write_txt([agLOG.tra_cls], f"{_s}")
                    tb_writer.add_scalars(f"{tag_this_bb} train iter{TM}", {'acc': acc}, ii)
                    tb_writer.add_scalars(f"{tag_this_bb} train iter{TM}", {'loss': loss}, ii)
                    tb_writer.add_scalars(f"{tag_this_bb} train iter{TM}", {'lr': _lr}, ii)
            # end train in this epoch; calculate epoch accuracy.
            y_, lb = torch.cat(yhat_batch).cpu().numpy(), torch.cat(lb_batch).cpu().numpy()
            correct = sum(y_ == lb)
            acc = correct.item() / lb.shape[0]
            tb_writer.add_scalars(f"{tag_this_bb} epoch acc{TM}", {'train': acc}, e)
            yhat_batch, lb_batch, p = [], [], 0
            with torch.no_grad():
                for images, lb in tqdm(dl_val):
                    images, lb = images.to(dev), lb.to(dev)
                    y_ = net(images, feat_val[p:p+len(images)])
                    p += len(images)
                    yhat_batch.append(y_.argmax(dim=1).detach())
                    lb_batch.append(lb.detach())
                y_, lb = torch.cat(yhat_batch).cpu().numpy(), torch.cat(lb_batch).cpu().numpy()
                correct = sum(y_== lb)
                acc = correct.item() / lb.shape[0]
                write_txt([agLOG.val_cls], f"{tag_this_bb} validating e[{e}/{epoch_max}], acc={acc:.4f}, loss={loss:.4f}")
                tb_writer.add_scalars(f"{tag_this_bb} epoch acc{TM}", {'val': acc}, e)
            if (e == epoch_max) or (e % 10 == 0):
                os.makedirs(fp_pth, exist_ok=True)
                ckpoint = {'net': net.state_dict(), 'acc': np.round(acc, 4), 'epoch': e, 'step': ii}
                if e < epoch_max: torch.save(ckpoint, osp.join(fp_pth, f"{tag_this_bb}_acc={acc:.3f}e={e}step={ii}.pth"))
                else:
                    _fn = f"{tag_this_bb}_acc={acc:.3f}e={e}step={ii}_final.pth"
                    write_txt([agLOG.tra_cls, agLOG.val_cls, agLOG.log_all_cls], f"** run_llm_resnet[{TM}] last .pth fn:{_fn}")
                    torch.save(ckpoint, osp.join(fp_pth, _fn))
                    os.makedirs(fp_pth_last, exist_ok=True)
                    fnp_pth = osp.join(fp_pth_last, _fn) # for return
                    torch.save(ckpoint, fnp_pth)

    else: # not is_train
        _find = [clip_bb_tag, attn] # this_bb if 'keep' in attn else tag_this_bb
        write_txt([agLOG.tra_cls], f" '{tag_this_bb}' skip training and load {_find} from {agDIR.last_cls}, find .pth in: \n {fnp_pth}")
        # net = get_clip_net(bb, num_cls) # do NOT do this!
        net, acc, e, fnp_pth = reload_net(net, dev, fnp_pth, log=agLOG.tra_cls, _find=_find)
    ## both train or not train, just validate again
    write_txt([agLOG.tra_cls], f"*{agTAG.bug_cls} start validating for run_llm_resnet at {time_str()}")
    yhat_batch, lb_batch, p = [], [], 0
    with torch.no_grad():
        for images, lb in tqdm(dl_val): # tqdm(DataLoader(ds_val, batch_size=batch_size, num_workers=2)):
            images, lb = images.to(dev), lb.to(dev)
            y_ = net(images, feat_val[p:p+len(images)])
            p += len(images)
            yhat_batch.append(y_.argmax(dim=1).detach())
            lb_batch.append(lb.detach())
        yhat_batch, lb_batch = torch.cat(yhat_batch).cpu().numpy(), torch.cat(lb_batch).cpu().numpy()
        correct = sum(yhat_batch == lb_batch)
        acc = correct.item() / lb_batch.shape[0]
        write_txt([agLOG.val_cls], f"{tag_this_bb} validating {e}/{epoch_max}, acc={acc:.4f}, loss={loss:.4f}")
    tb_writer.close()
    write_txt([agLOG.tra_cls], f"*{TM} {agTAG.bug_cls} is_train={is_train} run_clip_attention({tag_this_bb}) val acc={acc:.4f}, loss={loss:.4f} end at{time_str()}, cost{time_gap(t1)}")
    return yhat_batch, lb_batch, acc*100, loss, fnp_pth
# end run_llm_resnet

if __name__ == '__main__':
    tt0 = datetime.datetime.now() # only here; other use time.time()

    print(f"- main of llm_resnet.py done at {agBASE.tm}, from {time_str(tt0)}, cost {time_gap(tt0)}")
