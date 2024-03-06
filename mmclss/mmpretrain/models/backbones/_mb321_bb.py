# mb321_bb.py, chengyu wang, 2022-0415, last update:2022-0527
from genericpath import exists
import sys, os

from cv2 import transform
from pt_flag import IMG_C, IMG_H, NUM_CLASS_ABN, TEMP_ROOT, Ii, dict2class, time_strHMS, TM, QUICK
_import_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_import_root)
from pt_flag import SWIN_DICT, FFN_RATIO, OUT_RATIO
from all_include.pt_cnn import ToyCnn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from copy import deepcopy
from typing import Sequence

import numpy as np
import torch, warnings
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
# from mmcv.cnn.bricks.transformer import FFN, PatchEmbed, PatchMerging
from all_include.mm_tsfm import resize_pos_embed, to_2tuple, OurFFN, OurPatchEmbed, OurPatchMerging, OurShiftMSA # 'Our' to avoid 'already registered'
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
# from ..utils import ShiftWindowMSA, resize_pos_embed, to_2tuple
from .base_backbone import BaseBackbone

@BACKBONES.register_module()
class OurToy(ToyCnn): # BaseBackbone, BaseModule
    def __init__(self): # , num_classes=-1, input_channels=3, init_cfg=None
        ToyCnn.__init__(self) # , num_classes=num_classes, input_channels=input_channels
        # BaseBackbone.__init__(self , init_cfg) #

from all_include.pt_tsfm import ClsOurVit, ClsToyVit
@BACKBONES.register_module()
class OurVit(ClsOurVit):
    def __init__(self, img_size=224, patch_size=16, dim=1024, init_cfg=None):
        ClsOurVit.__init__(self, img_size=img_size, patch_size=patch_size, dim=dim, init_cfg=init_cfg)

@BACKBONES.register_module()
class ToyVit(ClsToyVit):
    def __init__(self): #, img_size=224, patch_size=16, dim=1024, init_cfg=None):
        ClsToyVit.__init__(self) #, img_size=img_size, patch_size=patch_size, dim=dim, init_cfg=init_cfg)

### from A:\Syno\0sync\PT\mmclassi\mmcls\models\backbones\swin_transformer.py
class OurSwinBlock(BaseModule):
    def __init__(self, embed_dims, num_heads, window_size=7, shift=False, ffn_ratio=FFN_RATIO, drop_path=0., pad_small_map=False, attn_cfgs=dict(), ffn_cfgs=dict(), norm_cfg=dict(type='LN'), init_cfg=None):
        super(OurSwinBlock, self).__init__(init_cfg)
        _attn_cfgs = { 'embed_dims': embed_dims, 'num_heads': num_heads,
            'shift_size': window_size // 2 if shift else 0,
            'window_size': window_size,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path), 'pad_small_map': pad_small_map, **attn_cfgs }
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = OurShiftMSA(**_attn_cfgs)
        _ffn_cfgs = { 'embed_dims': embed_dims, 'ffn_ratio':ffn_ratio, 'num_fcs': 2, 'ffn_drop': 0, 'dropout_layer': dict(type='DropPath', drop_prob=drop_path), 'act_cfg': dict(type='GELU'), **ffn_cfgs }
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = OurFFN(**_ffn_cfgs)
    def forward(self, Bh2c, hw): # [2, 3136, 96], (56,56)
        def _inner_forward(Bh2c):
            identity = Bh2c # [2, 3136, 96]
            Bh2c = self.norm1(Bh2c) # [2, 3136, 96]
            Bh2c = self.attn(Bh2c, hw) # [2, 3136, 96]
            Bh2c = Bh2c + identity # [2, 3136, 96]

            identity = Bh2c # [2, 3136, 96]
            Bh2c = self.norm2(Bh2c) # [2, 3136, 96]
            Bh2c = self.ffn(Bh2c, identity=identity) # [2, 3136, 96]
            return Bh2c # [2, 3136, 96]
        Bh2c = _inner_forward(Bh2c)
        return Bh2c # [2, 3136, 96], shape NOT changed!
# end class OurSwinBlock
class StageSequence(BaseModule):
    def __init__(self, embed_dims, num_blocks, num_heads, window_size=7, downsample=False, downsample_cfg=dict(), drop_paths=0., block_cfgs=dict(), pad_small_map=False, init_cfg=None):
        super().__init__(init_cfg)
        if not isinstance(drop_paths, Sequence): drop_paths = [drop_paths] * num_blocks
        if not isinstance(block_cfgs, Sequence): block_cfgs = [deepcopy(block_cfgs) for _ in range(num_blocks)]
        self.embed_dims = embed_dims
        self.blocks = ModuleList()
        print(f"* OurSwin StageSequence with {num_blocks} blocks, num_heads={num_heads}, window_size={window_size}, drop_paths={drop_paths}")
        for i in range(num_blocks):
            _block_cfg = { 'embed_dims': embed_dims,
                'num_heads': num_heads,
                'window_size': window_size,
                'shift': False if i % 2 == 0 else True,
                'drop_path': drop_paths[i], 'pad_small_map': pad_small_map, **block_cfgs[i] }
            block = OurSwinBlock(**_block_cfg)
            self.blocks.append(block)
        if downsample:
            _downsample_cfg = {'in_channels': embed_dims, 'out_ratio': OUT_RATIO, 'norm_cfg': dict(type='LN'), **downsample_cfg } # 'out_channels': 'fake', 
            self.downsample = OurPatchMerging(**_downsample_cfg)
        else: self.downsample = None
    def forward(self, Bh2c0, h0w0): # h2=h0*w2; 0:([2, 3136, 96], (56, 56))
        for block in self.blocks: Bh2c0 = block(Bh2c0, h0w0) # shape NOT changed!
        if self.downsample: Bk2c1, h1w1 = self.downsample(Bh2c0, h0w0) # c1=2*c0; out_shape=hw/2=out_shape
        else: Bk2c1, h1w1 = Bh2c0, h0w0 # NOT changed!
        return Bk2c1, h1w1 # k2=h1*w1; 
    @property
    def out_channels(self):
        if self.downsample: return self.downsample.out_channels
        else: return self.embed_dims
# end class StageSequence
class mb321_neck(nn.Module): # GlobalAveragePooling()
    def __init__(self, dim=2):
        super(mb321_neck, self).__init__()
        assert dim in [1, 2, 3], f'GlobalAveragePooling dim only support {1, 2, 3}, get {dim} instead.'
        if dim == 1: self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2: self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else: self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
    def init_weights(self): pass
    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else: raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs # BFeat, [2,768]
# end mb321_neck

@BACKBONES.register_module()
class OurSwin(BaseBackbone): 
    _version, num_extra_tokens = 3, 0
    def __init__(self, arch=SWIN_DICT, img_size=IMG_H, in_channels=IMG_C, num_classes=NUM_CLASS_ABN, drop_rate=0., drop_path_rate=0.1, use_abs_pos_embed=False, interpolate_mode='bicubic', norm_eval=False, pad_small_map=False, norm_cfg=dict(type='LN'), stage_cfgs=dict(), patch_cfg=dict(), init_cfg=None): # patch_size=4, in_channels=3, window_size=7, out_indices=(3, ), 
        super(OurSwin, self).__init__(init_cfg=init_cfg)
        print(f"** init of class OurSwin(BaseModule) ...")
        self.arch_settings = {} # tiny, {'embed_dims': 96, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24]}
        if len(arch) > 1:
            for k, v in arch.items(): self.arch_settings[k] = v
        self.embed_dims = self.arch_settings['embed_dims'] # 96
        self.blocks = self.arch_settings['depths'] # [2, 2, 6, 2]
        self.num_heads = self.arch_settings['num_heads'] # [3, 6, 12, 24]
        patch_size = self.arch_settings['patch_size'] # 4
        window_size = self.arch_settings['window_size'] # 7
        self.num_layers = len(self.blocks) # 4
        self.out_indices = (len(self.blocks)-1,) # 3=4-1
        self.use_abs_pos_embed = use_abs_pos_embed
        self.interpolate_mode = interpolate_mode
        if self.out_indices[0] != self.num_layers-1: input(f"** out_indices({self.out_indices[0]}) != num_layers({self.num_layers-1})\n")
        self.prt = dict2class({'output':True, 'input':QUICK, 'embed':QUICK})
        print(f"self.prt={self.prt}")
        self.neck = mb321_neck()
        self.head = None # nn.Linear(arch['head_channels'], num_classes)

        norm_cfg = None if QUICK else dict(type='LN') # 
        _patch_cfg = dict(in_channels=in_channels, input_size=img_size, embed_dims=self.embed_dims, conv_type='Conv2d', kernel_size=patch_size, stride=patch_size, norm_cfg=norm_cfg,)
        _patch_cfg.update(patch_cfg)
        self.patch_embed = OurPatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size # (56,56)
        if self.use_abs_pos_embed:
            num_patches = self.patch_resolution[0] * self.patch_resolution[1]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
            self._register_load_state_dict_pre_hook(self._prepare_abs_pos_embed)
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.norm_eval = norm_eval

        total_blocks = sum(self.blocks) # 12=[2+2+6+2]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]  # stochastic depth decay rule

        self.stages = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (num_blocks, num_heads) in enumerate(zip(self.blocks, self.num_heads)):
            if isinstance(stage_cfgs, Sequence): stage_cfg = stage_cfgs[i]
            else: stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i < self.num_layers - 1 else False
            _stage_cfg = {'embed_dims': embed_dims[-1], 'num_blocks': num_blocks, 'num_heads': num_heads, 'window_size': window_size, 'downsample': downsample, 'drop_paths': dpr[:num_blocks], 'pad_small_map': pad_small_map, **stage_cfg}

            stage = StageSequence(**_stage_cfg)
            self.stages.append(stage)

            dpr = dpr[num_blocks:]
            embed_dims.append(stage.out_channels)
        for i in self.out_indices: # the last block of the stage.
            if norm_cfg is not None: norm_layer = build_norm_layer(norm_cfg, embed_dims[i + 1])[1]
            else: norm_layer = nn.Identity()
            self.add_module(f'norm{i}', norm_layer)
    def init_weights(self):
        super(OurSwin, self).init_weights()
        if (isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained'): return
        if self.use_abs_pos_embed: trunc_normal_(self.absolute_pos_embed, std=0.02)
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args, **kwargs):
        """load checkpoints."""
        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is OurSwin:
            final_stage_num = len(self.stages) - 1
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if k.startswith('norm.') or k.startswith('backbone.norm.'):
                    convert_key = k.replace('norm.', f'norm{final_stage_num}.')
                    state_dict[convert_key] = state_dict[k]
                    del state_dict[k]
        if (version is None or version < 3) and self.__class__ is OurSwin:
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if 'attn_mask' in k: del state_dict[k]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, *args, **kwargs)
    def train(self, mode=True):
        super(OurSwin, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules(): # trick: eval have effect on BatchNorm only               
                if isinstance(m, _BatchNorm): m.eval()
    def _prepare_abs_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'absolute_pos_embed'
        if name not in state_dict.keys(): return
        ckpt_pos_embed_shape = state_dict[name].shape
        if self.absolute_pos_embed.shape != ckpt_pos_embed_shape:
            from mmcls.utils import get_root_logger
            logger = get_root_logger()
            logger.info(f"Resize the absolute_pos_embed shape from {ckpt_pos_embed_shape} to {self.absolute_pos_embed.shape}.")
            ckpt_pos_embed_shape = to_2tuple(int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name], ckpt_pos_embed_shape, pos_embed_shape, self.interpolate_mode, self.num_extra_tokens)
    def forward(self, BCHW): # [2, 3, 224, 224]
        fp = os.path.join(TEMP_ROOT, 'jpg', f"embed_{TM}")
        if QUICK and self.prt.input:
            self.prt.input = False
            os.makedirs(fp, exist_ok=True)
            ## visualize the original input [B,C,H,W]
            for idx in range(BCHW.shape[0]):
                t = BCHW[idx].detach().cpu()
                for i in range(3): t[i] = t[i].mul(Ii.std[i]) + Ii.mean[i]
                hwc = t.numpy().transpose(1,2,0).astype(np.uint8)
                # plt.imshow(hwc)
                plt.imsave(os.path.join(fp, f"input_{time_strHMS()}_{idx}.jpg"), hwc)
                # plt.clf()

        Bh2c0, hw = self.patch_embed(BCHW) # [2, 3136=56x56, 96=embed_dims], (56, 56); [B, out_h * out_w, embed_dims]
        
        if QUICK and self.prt.embed:
            self.prt.embed = False
            os.makedirs(fp, exist_ok=True)
            # ## visualize the embed [B, h0*w0, c0]
            for idx in range(len(Bh2c0)):
                t = Bh2c0[idx].detach().cpu()
                t = t.reshape(hw[0], hw[1], t.shape[1])
                for i in range(3): t[i] = t[i].mul(Ii.std[i]) + Ii.mean[i]
                u8 = t.numpy().astype(np.uint8) # !! already hwc, don't .transpose(1,2,0)
                n = int(t.shape[2]/3); j=0; s=f"{time_strHMS()}_{idx}"
                plt.figure(figsize=(20,16))
                plt.subplots_adjust(wspace=0.4, hspace=0.5)
                for p in range(1,n+1):
                    hh = u8[:,:, j:j+3]               
                    plt.imsave(os.path.join(fp, f"embed_{s}_{j}.jpg"), hh)                
                    plt.subplot(6,6,p)
                    plt.xlabel(f"{p}", fontsize=12)
                    plt.xticks([]); plt.yticks([])
                    plt.imshow(hh)
                    j += 3
                plt.savefig(os.path.join(fp, f"embed_{s}_{idx}.jpg"))
                plt.savefig(os.path.join(fp, f"embed_{s}_{idx}.pdf"), dpi=200)
                plt.clf()
        if self.use_abs_pos_embed:
            Bh2c0 = Bh2c0 + resize_pos_embed( self.absolute_pos_embed, self.patch_resolution, hw, self.interpolate_mode, self.num_extra_tokens)
        Bh2c = self.drop_after_pos(Bh2c0) # [2, 3136, 96=d0=d]

        Bchw_list = []
        for idx, stage in enumerate(self.stages):# 0:([2, 3136, 96], (56, 56)); 1:([2, 784, 192]), (28, 28)); 2:([2, 196, 384]), (14, 14)); 3:([2, 49, 768]), (7, 7))
            Bh2c, hw = stage(Bh2c, hw) # [2, h_in^w_in=h_in^2, c_in*2]
            # -->0:[2, 3136=h0*w0=56^2,  96=c0=embed_dims]; 56=out=1+(in+2p+k)/s=1+(224+0-4)/4
            # -->1:[2   784=h1*w1=28^2, 192=c0*2=96*2];     28=out=1+(in+2p+k)/s=1+(56+0-2)/2
            # -->2:[2,  196=h2*w1=14^2, 384=c1*2=192*2];    14=out=1+(in+2p+k)/s=1+(28+0-2)/2
            # -->3:[2,   49=h3*w1=7^2,  768=c2*2=384*2];     7=out=1+(in+2p+k)/s=1+(14+0-2)/2
            if idx in self.out_indices: # 3=len(self.stages)-1
                norm_layer = getattr(self, f'norm{idx}') # D768=D3
                Bh2c3 = norm_layer(Bh2c) # [2, 49, 768=D3]
                Bc3hw = Bh2c3.view(-1, *hw, stage.out_channels).permute(0, 3, 1, 2).contiguous() # [2, 768, 7, 7]
                Bchw_list.append(Bc3hw)
        if self.neck:
            outs=self.neck(tuple(Bchw_list)) # [B,in_channels]=[2,768]
            # if self.head:
            #     cls_score = self.head(outs[-1]) # [B,num_classes]=[2,24]
            #     outs = F.softmax(cls_score, dim=1)
        else: outs = tuple(Bchw_list) # [B,in_channels,out_h, out_w]=[2,768,7,7]
        if self.prt.output: print(f"** last stage output:{Bchw_list[0].shape}; outs:{outs[0].shape}\n self.head={self.head}"); self.prt.output = False        
        return outs # outs[0]
# end class OurSwin

def leave_me_here(): print("leave me here, just for a reminder of a ',' ")
