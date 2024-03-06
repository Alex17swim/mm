# # mm_tsfm.py, chengyu wang, 20220518, from: \wpt1.10\Lib\site-packages\mmcv\cnn\bricks\transformer.py
import sys, os
_import_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_import_root)
from pt_flag import GITHUB, print1, input1
import numpy as np
import copy, math, warnings
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc
from itertools import repeat
from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer, build_norm_layer)
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning, to_2tuple)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.utils.weight_init import trunc_normal_
# Avoid BC-breaking of importing MultiScaleDeformableAttention from this file
try: from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention  # noqa F401
except ImportError: warnings.warn(f" check warning in: mmcv\\cnn\\bricks\\transformer.py")

def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)

def build_attention(cfg, default_args=None): return build_from_cfg(cfg, ATTENTION, default_args)
def build_feedforward_network(cfg, default_args=None): return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)
def build_transformer_layer(cfg, default_args=None): return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)
def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)

def _ntuple(n):    
    def parse(x):
        if isinstance(x, collections.abc.Iterable): return x
        return tuple(repeat(x, n))
    return parse
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def resize_pos_embed(pos_embed, src_shape, dst_shape, mode='bicubic', num_extra_tokens=1):
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]: return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, f"The length of `pos_embed` ({L}) doesn't match the expected "  f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the'  '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    return torch.cat((extra_tokens, dst_weight), dim=1)
# end resize_pos_embed

class AdaptivePadding(nn.Module): # Applies padding adaptively to the input.
    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation) # ?? Spacing between kernel elements.
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
    def get_pad_shape(self, input_shape): # pad to: input_size/stride=224/4=56;
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w
    def forward(self, BCHW): # [1, 3, 224, 224]; Add padding to `x`
        pad_h, pad_w = self.get_pad_shape(BCHW.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner': BChw = F.pad(BCHW, [0, pad_w, 0, pad_h])
            elif self.padding == 'same': BChw = F.pad(BCHW, [ pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2 ])
        else: BChw = BCHW
        return BChw
# enc class AdaptivePadding
class OurPatchEmbed(BaseModule): # Image to Patch Embedding.    
    def __init__(self, in_channels=3, embed_dims=768, conv_type='Conv2d', kernel_size=4, stride=4, padding='corner', dilation=1, bias=True, norm_cfg=None, input_size=None, init_cfg=None): # kernel_size=16, stride=16
        super(OurPatchEmbed, self).__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        if stride is None: stride = kernel_size # 4, !! kernel_size=patch_size
        kernel_size = to_2tuple(kernel_size) # 4
        stride = to_2tuple(stride) # (4,4), !! stride=patch_size
        if dilation != 1: input(f"?? dilation({dilation}) is: Spacing between kernel elements. default 1")
        dilation = to_2tuple(dilation)
        print(f"* OurPatchEmbed stride=patch_size=kernel_size={kernel_size}, dilation={dilation}")

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding( kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
            padding = 0 # disable the padding of conv
        else: self.adaptive_padding = None
        padding = to_2tuple(padding)

        print(f"* PatchEmbed projection: in_channels={in_channels}, out_channels=embed_dims={embed_dims}, kernel_size={kernel_size}, stride={stride}")
        self.projection = build_conv_layer(dict(type=conv_type), in_channels=in_channels,  out_channels=embed_dims, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        if norm_cfg is not None: self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else: self.norm = None

        if input_size: # 224, img_size
            input_size = to_2tuple(input_size) # `init_out_size` would be used outside to calculate the num_patches, e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # o=1+(in+2p-k)/s; round down
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None
    def forward(self, BCHW): # (2, 3, 224, 224)        
        if self.adaptive_padding: BChw = self.adaptive_padding(BCHW)
        else: BChw = BCHW

        Bdkk = self.projection(BChw) # [1, 96, 56, 56]
        kk = (Bdkk.shape[2], Bdkk.shape[3]) # out_size: (56, 56)
        Bk2c1 = Bdkk.flatten(2).transpose(1, 2) # [1, 3136, 96]
        if self.norm is not None: Bk2c1 = self.norm(Bk2c1)
        return Bk2c1, kk
# end class OurPatchEmbed
class OurPatchMerging(BaseModule):
    def __init__(self, in_channels, out_ratio=2, norm_cfg=dict(type='LN'), init_cfg=None): # out_channels, 
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels # 96
        self.out_channels = out_ratio*in_channels # 192; out_channels for further use !!set in: _downsample_cfg
        # if stride: stride = stride
        # else: stride = kernel_size
        stride = kernel_size = out_ratio
        padding = 'corner'; dilation=1; bias=False

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding( kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
            padding = 0 # disable the padding of unfold
        else: self.adaptive_padding = None
        padding = to_2tuple(padding)
        self.sampler = nn.Unfold( kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        sample_dim = kernel_size[0] * kernel_size[1] * in_channels
        print(f"** PatchMerging, Unfold: kernel={kernel_size}, pad={padding}, stride={stride}; dilation={dilation}\n in_channels={in_channels}(out=2*in={2*in_channels}), sample_dim={sample_dim}")
        
        if norm_cfg is not None: self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else: self.norm = None
        self.reduction = nn.Linear(sample_dim, self.out_channels, bias=bias)
    def forward(self, Bh2c0, hw): # [2, 3136, 96]=(B, h0*w0, C_in); (56,56)
        B, L, c0 = Bh2c0.shape # 2, 3136=h2=h0*w0, 96
        assert isinstance(hw, Sequence), f'Expect input_size is `Sequence`, but get {hw}'
        h0, w0 = int(np.sqrt(L)), int(np.sqrt(L)) # 56,56
        assert L == h0 * w0, 'input feature has wrong size'
        Bc0h0w0 = Bh2c0.view(B, h0, w0, c0).permute([0, 3, 1, 2]) # [2,96,56,56]; B, c0, h0, w0
        if self.adaptive_padding:
            Bc0h0w0 = self.adaptive_padding(Bc0h0w0) # [2,96,56,56]
            h0, w0 = Bc0h0w0.shape[-2:] # (56, 56)
        # Use nn.Unfold to merge patch. About 25% faster than original method,  but need to modify pretrained model for compatibility. 
        # !!! [2, 384=c0*k*s=96*4, 784=h0/s*w0/s=(56/2)^2]=h0*w0/4=3136/4; set kernel=stride=2
        Bdk2 = self.sampler(Bc0h0w0) # [2, 384, 784]
        
        # out_h1=1+(in_h0+2p-kernel/stride)=h0/2=28; kernel=stride=2, pad=0, dialation=1, 
        h1 = (h0 + 2 * self.sampler.padding[0] - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1) - 1) // self.sampler.stride[0] + 1 # out_h
        w1 = (w0 + 2 * self.sampler.padding[1] - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1) - 1) // self.sampler.stride[1] + 1 # out_w

        kk = (h1, w1) # (28, 28)=output_size=(k,k)=(h1,w1)=(h0/2,w0/2)
        Bk2d = Bdk2.transpose(1, 2) # [2, 784=h1*w1=28*28, 384=c1*4=]; B, h/2*w/2, 4*c
        Bk2d = self.norm(Bk2d) if self.norm else Bk2d # [2, 784, 384]
        Bk2c1 = self.reduction(Bk2d) # [2, 784, 192]
        return Bk2c1, kk # [2, 784=k*k=h1*w1, 192=2*c0=out_channels]; Merged_H, Merged_W
# enc class OurPatchMerging

@FEEDFORWARD_NETWORK.register_module()
class OurFFN(BaseModule):
    @deprecated_api_warning( { 'dropout': 'ffn_drop', 'add_residual': 'add_identity' }, cls_name='OurFFN')
    def __init__(self, embed_dims=256, ffn_ratio=4., num_fcs=2, act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0., dropout_layer=None, add_identity=True, init_cfg=None, **kwargs):
        super(OurFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        feedforward_channels = int(embed_dims * ffn_ratio) # 384=96*4.0
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs # 2
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        print(f"* OurFFN num_fcs={num_fcs}, in_channels={embed_dims}, out_channels={feedforward_channels}")

        layers, in_channels = [], embed_dims
        for _ in range(num_fcs - 1):
            layers.append( Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        print1(f"* FFN layers:{layers}")
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity
    @deprecated_api_warning({'residual': 'identity'}, cls_name='OurFFN')
    def forward(self, Bk2c1, identity=None): # [2, 3136, 96]
        out_Bo2d = self.layers(Bk2c1)
        if not self.add_identity: return self.dropout_layer(out_Bo2d)
        if identity is None: identity = Bk2c1
        return identity + self.dropout_layer(out_Bo2d) # [2, 3136, 96], shape NOT changed!
# end class OurFFN

# @ATTENTION.register_module()
# class OurMultiheadAttention(BaseModule): # A wrapper for ``torch.nn.MultiheadAttention``

# @TRANSFORMER_LAYER.register_module()
# class BaseTransformerLayer(BaseModule): # Base `TransformerLayer` for vision transformer.

# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class TransformerLayerSequence(BaseModule): # Base class for TransformerEncoder and TransformerDecoder in vision transformer.

class OurMSA(BaseModule): # Window based multi-head self-attention (W-MSA) module with relative position bias.
    def __init__(self, embed_dims, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims # 96
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads # arch_settings['num_heads'] # [3, 6, 12, 24]
        head_embed_dims = embed_dims // num_heads # 32=96/3
        self.scale = qk_scale or head_embed_dims**-0.5 # [0.176, ]

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [169, 3]=[(2x7)^2, 3], 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size # 7, 7
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww) # [1, 49]=[1, 7x7]
        rel_position_index = rel_index_coords + rel_index_coords.T # [49, 49]
        rel_position_index = rel_position_index.flip(1).contiguous() # [49, 49]
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias) # 96, 96*3=288
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
    def init_weights(self):
        super(OurMSA, self).init_weights()
        trunc_normal_(self.relative_position_bias_table, std=0.02)
    def forward(self, x, mask=None): # [128, 49, 96], (num_windows*B, N, c), (num_windows, Wh*Ww, Wh*Ww)
        B_, N, c = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4) # ([3, 128, 3, 49, 32]
        q, k, v = qkv[0], qkv[1], qkv[2] # [128, 3, 49, 32], .. make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        print1(f"* MSA shape of x={x.shape}, qkv={qkv.shape}, attn={attn.shape}", 'MSA')

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)  # [49, 49, 3] Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute( 2, 0, 1).contiguous() # [3, 49, 49] nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # [128, 3, 49, 49]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else: attn = self.softmax(attn)
        attn = self.attn_drop(attn) # [128, 3, 49, 49]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, c) # [128, 49, 96=]
        x = self.proj(x) # [128, 49, 96]
        x = self.proj_drop(x)
        print1(f"* MSA output shape x={x.shape}", 'MSA')
        return x # [128, 49, 96] (-inf, 0]
    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)
# end class OurMSA
        
@ATTENTION.register_module()
class OurShiftMSA(BaseModule): # Shift Window Multihead Self-Attention Module.
    def __init__(self, embed_dims, num_heads, window_size, shift_size=0, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0, dropout_layer=dict(type='DropPath', drop_prob=0.), pad_small_map=False, input_resolution=None, auto_pad=None, init_cfg=None):
        super().__init__(init_cfg)
        if input_resolution is not None or auto_pad is not None: warnings.warn( 'The OurShiftMSA in new version has supported auto padding and dynamic input shape in all condition. And the argument `auto_pad` and `input_resolution` have been deprecated.', DeprecationWarning)
        self.shift_size = shift_size # 3
        self.window_size = window_size # 7
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = OurMSA( embed_dims=embed_dims, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, )
        self.drop = build_dropout(dropout_layer)
        self.pad_small_map = pad_small_map
    def forward(self, Bqc, hw): # Bqc=Bk2c1=[2, 3136, 96]; (56, 56)
        B, q, c = Bqc.shape # 2, 3136, 96
        h, w = int(np.sqrt(q)), int(np.sqrt(q)) # hw # 56, 56
        assert q == h * w, f"The query length {q} doesn't match the input shape ({h}, {w})."
        Bhwc = Bqc.view(B, h, w, c) # [2, 56, 56, 96]
        window_size = self.window_size # 7
        shift_size = self.shift_size # 0

        if min(h, w) == window_size: # If not pad small feature map, avoid shifting when the window size is equal to the size of feature map. It's to align with the behavior of the original implementation.
            shift_size = shift_size if self.pad_small_map else 0 # 0
        elif min(h, w) < window_size: # In the original implementation, the window size will be shrunk to the size of feature map. The behavior is different with swin-transformer for downstream tasks. To support dynamic input shape, we don't allow this feature.
            assert self.pad_small_map,  f'The input shape ({h}, {w}) is smaller than the window  size ({window_size}). Please set `pad_small_map=True`, or  decrease the `window_size`.'

        pad_r = (window_size - w % window_size) % window_size
        pad_b = (window_size - h % window_size) % window_size
        Bhwc = F.pad(Bhwc, (0, 0, 0, pad_r, 0, pad_b))

        h_pad, w_pad = Bhwc.shape[1], Bhwc.shape[2]

        # cyclic shift
        if shift_size > 0:
            # input1(f"shift_size={shift_size} (7//2 == 3)")
            Bhwc = torch.roll(Bhwc, shifts=(-shift_size, -shift_size), dims=(1, 2)) if GITHUB else \
                torch.roll(Bhwc, shifts=-shift_size, dims=1)
        attn_mask = self.get_attn_mask((h_pad, w_pad), window_size=window_size, shift_size=shift_size, device=Bhwc.device)
        query_windows = self.window_partition(Bhwc, window_size) # nW*B, window_size, window_size, c
        query_windows = query_windows.view(-1, window_size**2, c) # [128=2*64, 49=7*7, 96=c], nW*B, window_size*window_size, c
        attn_windows = self.w_msa(query_windows, mask=attn_mask) # [128, 49, 96], W-MSA/SW-MSA (nW*B, window_size*window_size, c)
        attn_windows = attn_windows.view(-1, window_size, window_size, c) # merge windows        
        shifted_Bhwc = self.window_reverse(attn_windows, h_pad, w_pad, window_size) # [128, 7, 7, 96], B h' w' c
        # reverse cyclic shift
        if self.shift_size > 0:
            Bhwc = torch.roll(shifted_Bhwc, shifts=(shift_size, shift_size), dims=(1, 2)) if GITHUB else \
                torch.roll(shifted_Bhwc, shifts=shift_size, dims=1)
        else: Bhwc = shifted_Bhwc # [2, 56, 56, 96]

        if h != h_pad or w != w_pad: Bhwc = Bhwc[:, :h, :w, :].contiguous() # [2, 56, 56, 96]
        Bqc = Bhwc.view(B, h * w, c) # [2, 3136, 96]
        Bqc = self.drop(Bqc) 
        return Bqc # [2, 3136, 96], shape NOT changed!
    @staticmethod
    def window_reverse(windows, h, w, window_size):
        B = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(B, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, h, w, -1)
        return x
    @staticmethod
    def window_partition(x, window_size):
        B, h, w, c = x.shape
        x = x.view(B, h // window_size, window_size, w // window_size, window_size, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, c)
        return windows

    @staticmethod
    def get_attn_mask(hw, window_size, shift_size, device=None):
        if shift_size > 0:
            img_mask = torch.zeros(1, *hw, 1, device=device)
            h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            cnt = 0
            if GITHUB:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
            else:
                for h in h_slices: # for w in w_slices: #
                    img_mask[:, h, :, :] = cnt # img_mask[:, :, w, :] = cnt #
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = OurShiftMSA.window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else: attn_mask = None
        return attn_mask
# end class OurShiftMSA