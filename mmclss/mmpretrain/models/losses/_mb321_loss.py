# _mb321_loss.py, Chengyu Wang, 2023-1230, from cross_entropy_loss.py
import torch
import torch.nn as nn
# import torch.nn.functional as F
#
# from mmpretrain.registry import MODELS
# from .utils import weight_reduce_loss

class PGMC_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_train, noise,imgn_train):
        h_x = out_train.size()[2]
        w_x = out_train.size()[3]
        count_h = self._tensor_size(out_train[:, :, 1:, :])
        count_w = self._tensor_size(out_train[:, :, :, 1:])
        h_gmc = torch.pow((out_train[:, :, 1:, :] - out_train[:, :, :h_x - 1, :]), 2).sum()
        w_gmc = torch.pow((out_train[:, :, :, 1:] - out_train[:, :, :, :w_x - 1]), 2).sum()
        gmcloss = h_gmc / count_h + w_gmc / count_w

        SmoothL1Loss = self.smooth_l1_loss(out_train,noise,imgn_train,reduce=True)/ (imgn_train.size()[0]*2)
        pgmc_loss = 0.01 * gmcloss + SmoothL1Loss
        return pgmc_loss

    def smooth_l1_loss(self,input, target,imgn_train,reduce=True):
        diff = torch.abs(input - target);cond = diff.item() < 1; loss = torch.where(cond, 0.5 * diff ** 2, diff - 0.5)
        if reduce:
            return torch.sum(loss)
        return torch.sum(loss, dim=1)

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

