# mm_vis_feat.py, chengyu wang, 2022.05.15, XJTLU, last update on 2022.05.15
''' *: os  $$: gpu  ##: dataset  &: image  ---: model  --: loss  -: directory  '''
import sys, os
_import_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_import_root)
import os.path as osp
from pt_flag import *
from all_include.pt_utils import label_is_21st


import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
from torchvision import models, transforms
import torch.utils.tensorboard as tb
from torch.nn.modules.normalization import LayerNorm
from timm.models.vision_transformer import vit_small_patch16_224_in21k # vit_large_patch16_384
# print(timm.list_models(pretrained=True))
from timm.models import create_model # , resume_checkpoint, convert_splitbn_model

import mmcv
from mmcls.models.backbones.resnet import ResLayer
# from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer
from mmcls.models.backbones import *
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from all_include.mm_tsfm import OurPatchEmbed

def _vis_feat(loaded_model, device, vit_like, fp_in, fp_out, img_h, img_w, max_num=3 if QUICK else 9, _ext='.jpg', _tag='chr7'):
    '''vit_like: '''
    log_list = []
    dev = device if type(device) is str else device.type
    print(f"vis_feat, dev={dev}, size=[{img_h}, {img_w}]\n input={fp_in}\n output={fp_out}")
    if loaded_model: model = loaded_model.to('cpu'); print(f"** load model from given")
    # model = models.resnet50(pretrained=True)
    else: model = create_model('vit_small_patch16_224_in21k', pretrained=True, num_classes=NUM_CLASS_ABN); print(f"** load model from timm")

    # #print(model)
    # fake_img = torch.rand(size=(1, IMG_C, img_h, img_w))
    # fake_img = fake_img.to(dev)
    # print("adding graph to tensorboard ...")
    # tb_writer = tb.SummaryWriter(log_dir=osp.join(fp_out, 'tb'), comment=_tag, filename_suffix='tb')  # except train, as train is large
    # tb_writer.add_graph(model, fake_img)
    # print(f"cd to {fp_out},\n and check strucure with cmd: tensorboard --logdir=tb --port=6007")

    model_weights = [] # we will save the conv layer weights in this list
    model_children = list(model.children()) # get all the model children as list
    
    vis_layers, ct_vis = [], 0 # put all sub block in the list
    bb_type = BB_ABN.lower()
    bb_type = 'vit' if 'vit'.lower() in bb_type else 'swin' if 'swin'.lower() in bb_type else 'res'
    if vit_like: # transformer
        for idx, model_child in enumerate(model_children):
            print(f"No.{idx}, model_child type={type(model_child)}")
            if type(model_child) in [LayerNorm, PatchEmbed, OurPatchEmbed, torch.nn.modules.container.ModuleList]:
                vis_layers.append(model_child)
                ct_vis += 1
            elif type(model_child) == mmcv.runner.base_module.ModuleList:
                sub_children = list(model_child.children())
                for sub_child in sub_children:
                    # if type(sub_child) in [TransformerEncoderLayer, SwinBlockSequence]:
                    vis_layers.append(sub_child); ct_vis += 1                        
                    # vis_layers.append(sub_child.attn) # mmcls.models.utils.attention.MultiheadAttention
                    # vis_layers.append(sub_child.ffn) # mmcv.cnn.bricks.transformer.FFN
                    # ct_vis += 2
            else: # [Dropout, ]
                sub_children = list(model_child.children())
                for sub_child in sub_children:
                    if type(sub_child) == nn.Conv2d:
                        vis_layers.append(sub_child)
                        ct_vis += 1
    else: # cnn
        vis_layers, ct_vis = [], 0 # for i in range(len(model_children)):
        for i in range(len(model_children)): # use i to avoid: 'Bottleneck' object is not iterable
            if type(model_children[i]) == nn.Conv2d:
                ct_vis += 1
                model_weights.append(model_children[i].weight)
                vis_layers.append(model_children[i])
            elif type(model_children[i]) in [nn.Sequential, ResLayer]:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            ct_vis += 1
                            model_weights.append(child.weight)
                            vis_layers.append(child)
        # # for weight, conv in zip(model_weights, vis_layers):
        # #     # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
        # #     print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
        # make_dirs(fp_out)
        # plt.figure(figsize=(20, 17)) # visualize the first conv layer filters
        # for i, filter in enumerate(model_weights[0]):
        #     plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        #     plt.imshow(filter[0, :, :].detach(), cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(osp.join(fp_out, 'filter1.jpg'))
        # plt.pause(1)
        # model_weights[0].shape; len(model_weights)
    for i in get_range([0,1,2,3], len(vis_layers)): print(f"Layer {i:>02d}, type={type(vis_layers[i])}")
    print(f"Total {ct_vis} conv/mlp layers, vit_like(mlp)={vit_like}")
    
    # read and visualize an image
    if fp_in.endswith(('.jpg', '.png')): fp_in = osp.dirname(fp_in)
    _fp_out_this, n_d, n_f, n_i = '',0,0,0
    for root, dirs, files in os.walk(fp_in):
        for dir in dirs:
            n_d += 1

        for file in files:
            n_f += 1
            if not file.endswith(_ext): continue
            if not_in_demo_case(file, osp.basename(root)): # plot all 24 labels in demo case, otherwise only plot 21st # True: #
                if PLT_21: # will plot all 21st, regardless of max_num #
                    _is = label_is_21st(file)
                    if 0 == _is or (0 > _is and n_i > PLT_CAM): continue
                elif n_i > max_num: print(f"** already {n_i} images, return to:\n {_fp_out_this}"); return _fp_out_this
            _dir = '' if len(root) == len(fp_in) else root[len(fp_in):]
            if (_dir != '') and ('output' in _dir): write_txt(log_list, f"~ skip dir:{_dir}, file:{file}"); continue
            _fp_out_this = osp.join(fp_out, _dir.lstrip(RSTRIP))
            os.makedirs(_fp_out_this, exist_ok=True)
            # input1(f"root={root}, file={file}\n _fp_out_this={_fp_out_this}\n", '_vis_feat')
            # dst = osp.join(_fp_out_this, file)
            src = osp.join(root, file)

            img = cv.imread(src)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # plt.imshow(img)
            plt.pause(1)
            # define the transforms
            transform = transforms.Compose( [transforms.ToPILImage(), transforms.Resize((img_h, img_w)), transforms.ToTensor(),] )
            img = np.array(img)
            # apply the transforms
            img = transform(img) # [3, 224, 224]
            img = img.unsqueeze(0) # [1, 3, 224, 224]; unsqueeze to add a batch dimension

            # pass the image through all the layers
            if 'swin' in bb_type:
                Bk2d, kk = vis_layers[0](img)
                results = [Bk2d]
                if isinstance(Bk2d, tuple): input(f"??? Bk2d type={type(Bk2d)}\n")
                for i in range(1, len(vis_layers)):
                    if isinstance(vis_layers[i], LayerNorm): Bk2d = vis_layers[i](Bk2d)
                    else: Bk2d, kk = vis_layers[i](Bk2d, kk)
                    print(f" appending No.{i:>02d} layer, Bk2d.shape={Bk2d.shape}, kk={kk}, type={type(vis_layers[i])}")
                    results.append(Bk2d)
            else: # vit, resnet
                results = [vis_layers[0](img)]
                for i in range(1, len(vis_layers)): # range(1, len(vis_layers)-2 if vit_like else len(vis_layers)):
                    # pass the result from the last layer to the next layer
                    v = results[-1][0] if isinstance(results[-1], tuple) else results[-1]
                    print(f" appending No.{i:>02d} layer, v.shape={v.shape}, type={type(vis_layers[i])}")
                    results.append(vis_layers[i](v))
                    if isinstance(vis_layers[i], LayerNorm) and vit_like: print(f"** break append from vis_layers after LayerNorm"); break
                if vit_like: dim = results[3].shape[2]; feat_size = int(np.sqrt(results[3].shape[1]))
                else: dim = 1024; feat_size = img_h
            outputs = results # make a copy of the `results`
            # visualize 36 features from each layer 
            # (although there are more feature maps in the upper layers)
            n_i += 1            
            all_layers_list = get_range([0, 1, 2, 3], len(outputs)) # change the number of length to control how many layers you want to visualize
            for idx, num_layer in enumerate(all_layers_list):
                plt.figure(figsize=(20, 20))
                if vit_like:
                    layer_viz = outputs[num_layer][0].data
                    layer_viz = layer_viz.squeeze(0).transpose(0, 1)
                    if 'swin' in bb_type:
                        feat_size = int(np.sqrt(layer_viz.shape[-1]))
                        layer_viz = layer_viz.reshape(-1, feat_size, feat_size)
                    else: layer_viz = layer_viz.reshape(dim, feat_size, feat_size)
                else:
                    layer_viz = outputs[num_layer][0, :, :, :]
                    layer_viz = layer_viz.data
                
                #print(layer_viz.size())
                for i, filter in enumerate(layer_viz):
                    if i == 36: break # we will visualize only 8x8 blocks from each layer
                    plt.subplot(6, 6, i + 1)
                    plt.imshow(filter, cmap='gray') # gray, brg
                    plt.axis("off")
                print(f"Saving img {n_i}, No.{idx+1} layer [{num_layer}/{all_layers_list[-1]}] feature maps...")
                plt.savefig(osp.join(_fp_out_this, f'{osp.splitext(file)[0]}_layer_{num_layer}.jpg'))  # change the path to save the feature maps
                #plt.show() # use this line to show the figure in jupyter notebook
                plt.close()
            # end for(num_layer)
            if n_f <= 3 or n_f == len(files) or (n_f%20) == 0 or (n_f%len(files)) == 0: write_txt(log_list, f"-- No.[{n_f:>03d}/{len(files):>03d}], {n_i}th image:{file}")
    write_txt(log_list, f"-- last No.[{n_f:>03d}], {n_i}th  image:{file}")  
    return _fp_out_this
# end _vis_feat
def vis_feat(loaded_model, device, vit_like, fp_in, fp_out, img_h, img_w, max_num=3 if QUICK else 9, _ext='.jpg', _tag='chr7'):
    if not QUICK:
        try: _vis_feat(loaded_model, device, vit_like, fp_in, fp_out, img_h, img_w, max_num, _tag='chr7')
        except Exception as e: print(f"!! vis_feat err:\n{e}")
        # finally: pass
    else: _vis_feat(loaded_model, device, vit_like, fp_in, fp_out, img_h, img_w, max_num, _tag='chr7')
        
if __name__ == '__main__':
    fp_in = osp.join(TEMP_ROOT, 'test', 'vis') # , 'val', 'dog.png'
    fp_out = osp.join(TEMP_ROOT, 'test', 'vis', f"output_vis_feat_{TM}")
    vis_feat(None, 'cpu', True, fp_in, fp_out, 224, 224)
    