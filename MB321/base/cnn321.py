# cnn321.py, Chengyu Wang, 2023-0532, last update on 2023-1102

import sys, os
_import_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_import_root)
# print("cnn321.py _import_root={}".format(_import_root))

from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision import models, transforms
# from torchvision.datasets import CIFAR100

from torch.utils.data import Dataset, DataLoader
import torch.utils.tensorboard as tb
# from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from collections import OrderedDict
import matplotlib
matplotlib.use('TKagg') # for UserWarning: Matplotlib is currently using agg
# import matplotlib.pyplot as plt
# import os, random, tqdm, re
# import os.path as osp
from PIL import Image
from MB321.mm_flag import *
from MB321.base.util321 import *
from MB321.base.data321 import set_transform, ChrDataset, RmbDataset # , RandomDataset

def reload_net(net, dev, fnp_pth='', log='', _find=agNET.bb_cls, tag=agTAG.bug_cls):
    if not fnp_pth:
        _find = ['.pth'] + _find if isinstance(_find, list) else ['.pth', str(_find)]
        fnp_pth = find_sth_easy(agDIR.last_cls, _find, _confirm=False)
        if fnp_pth is None:
            input3("!! skip reload_net[], as not found '.pth'\n") # [args.fnp_log_eval],
            return _find, fnp_pth
    else:
        assert osp.exists(fnp_pth), f"!! can't find: {fnp_pth}"
    print(f"the net have to be assigned as a parameter for reload_net[] since it maybe a self-designed net!\n")
    state_dict = torch.load(fnp_pth, map_location=dev)
    if 'fc.weight' in state_dict and agDATA.num_cls != state_dict['fc.weight'].shape[0]:
        del state_dict['fc.weight']
        del state_dict['fc.bias']
    net.load_state_dict(state_dict['net'], strict=False)
    net = net.to(dev)
    acc = state_dict['acc']
    e = state_dict['epoch'] if 'epoch' in state_dict.keys() else '-1'
    write_txt(log, f"* {tag}, loaded pth {os.path.basename(fnp_pth)}\n with acc={acc}, epoch={e}, starting eval() at {time_str()}...", True)
    if e != agDATA.epoch_cls: print(f" !! reloaded epoch {e} != expect {agDATA.epoch_cls}\n\n")
    return net, acc, e, fnp_pth
# end reload_net

def linear_model():
    x = torch.rand((20,1))*10
    y = 2*x + (5+torch.rand(20,1))
    w = torch.randn((1),requires_grad=True)
    b = torch.zeros((1),requires_grad=True)
    for i in range(0,1000):
        y_ = torch.add(torch.mul(w,x),b)
        loss = (0.5*(y-y_)**2).mean()
        loss.backward()
        w.data.sub_(0.01*w.grad.data)
        b.data.sub_(0.01*b.grad.data)
        w.grad.zero_()
        b.grad.zero_()
        if i % 20 == 0:
            plt.clf() # plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x, y_.data.numpy(),'r-')
            plt.text(1,20,"loss={:.3f}".format(loss.data))
            plt.title("iteration {}".format(i))
            if loss.data.numpy() < 1:
                print("linear done")
                plt.show()
                break
            else:
                plt.pause(0.1)
# end

expansion_layer = nn.Linear(1024, 2048)
def combinef1withf2(clip_feat, bb_feat):
    if isinstance(clip_feat, np.ndarray): clip_feat = torch.from_numpy(clip_feat)
    if isinstance(bb_feat, np.ndarray): bb_feat = torch.from_numpy(bb_feat)
    if clip_feat.dtype != torch.float32: clip_feat = clip_feat.float()
    dim_c = clip_feat.shape[1]
    dim_bb = bb_feat.shape[1]
    clip2 = expansion_layer(clip_feat) # nn.linear(dim_c, dim_bb)  # 扩展1024维特征到2048维
    combined_feature = clip2 + bb_feat
    return combined_feature

class LR(nn.Module): # logistic regression
    def __init__(self, dim_in=2, dim_out=1):
        super(LR, self).__init__()
        self.feat = nn.Linear(dim_in, dim_out)
        self.act = nn.Sigmoid()

    def forward(self,x):
            x = self.feat(x)
            x = self.act(x)
            return x
# end LR
def run_logistic_regression(fnp_pth, dev, output_cls, feat, lb, train=True, max_iter=1000, lr=0.001, e=-1):
    tb_writer = tb.SummaryWriter(log_dir=agDIR.out_tb_cls, comment='logistic_regree', filename_suffix='tb')
    fp_pth, bb_tag, num_cls = agDIR.out_pth_cls, agTAG.bb_cls.replace('bb','b'), agDATA.num_cls
    if isinstance(feat, np.ndarray): feat = torch.from_numpy(feat)
    if isinstance(lb, np.ndarray): lb = torch.from_numpy(lb)
    input_dim = feat.shape[1] # [batch, feature_size] for avgpool
    acc, loss, i, t1 = 0.0, 0.0, 0, time.time()
    net = LR(input_dim, output_cls)
    this_bb = f"LR_{bb_tag}"
    if train:
        net = net.to(dev)
        feat = feat.to(dev); lb = lb.to(dev)
        criterion = get_criterion()
        opt, _ = get_optimizer(net.parameters(), lr=lr)
        write_txt([agLOG.tra_cls], f"*run_logistic_regression[] {agTAG.bug_cls} start training for run_logistic_regression[] at {time_str()}")
        for i in range(0, max_iter):
            y_ = net(feat)
            loss = criterion(y_.squeeze(), lb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 20 == 0 or i < 5:
                np_pred = y_.argmax(dim=1).cpu().numpy()
                correct = sum(np_pred == lb.cpu().numpy())
                acc = correct.item()/lb.shape[0]
                print(f"{bb_tag} LR iteration{i}/{max_iter}, acc={acc:.3f}, loss={loss:.3f}")
                tb_writer.add_scalars(f"{this_bb}_{ACT} iter acc{TM}", {'train': acc}, i+1)
        tb_writer.close()
        os.makedirs(fp_pth, exist_ok=True)
        ckpoint = {'net': net.state_dict(), 'acc': np.round(acc, 4), 'epoch': e, 'step': i}
        _fn = f"{this_bb}acc={acc:.4f}e={e}step={i}.pth"
        write_txt([agLOG.tra_cls, agLOG.val_cls, agLOG.log_all_cls], f"** run_logistic_regression[{TM}] last .pth fn:{_fn}")
        fnp_pth = osp.join(fp_pth, _fn)
        torch.save(ckpoint, fnp_pth)
    else:
        write_txt([agLOG.tra_cls], f"skip training and load {this_bb} from {agDIR.last_cls}, find at: \n {fnp_pth}")
        # net = get_net(bb, num_cls)
        net, acc, e, fnp_pth = reload_net(net, dev, fnp_pth, log=agLOG.tra_cls, _find=this_bb)
    y_ = net(feat.to(dev))  # final check
    np_lb_pred = y_.argmax(dim=1).cpu().numpy()
    correct = sum(np_lb_pred == lb.cpu().numpy())
    acc = correct.item() / lb.shape[0]
    write_txt([agLOG.tra_cls], f"*{TM} run_logistic_regression[] {agTAG.bug_cls} train={train} run_logistic_regression[{this_bb}] acc={acc}, loss={loss}, end at{time_str()}, cost{time_gap(t1)}")
    return np_lb_pred, acc*100, loss, fnp_pth
# end run_logistic_regression

def sigmoid_model(): # logistic regression
    sample_num = 100
    mean = 1.7
    n_data = torch.ones(sample_num,2)
    x0 = torch.normal(mean=mean*n_data,std=1)+1
    y0 = torch.zeros(sample_num)
    x1 = torch.normal(mean=-mean*n_data, std=1)+1
    y1 = torch.ones(sample_num)
    train_x = torch.cat([x0,x1])
    train_y = torch.cat([y0,y1])    

    def logistic_fn():
        net = LR()
        criterion = nn.BCELoss()
        opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        for i in range(0,1000):
            y_ = net(train_x)
            loss = criterion(y_.squeeze(), train_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 20 == 0:
                mask = y_.ge(0.5).float().squeeze()
                correct = (mask == train_y).sum()
                acc = correct.item()/train_y.shape(0)
                w0, w1 = net.feat.weight[0]
                w0, w1 = float(w0.item()), float(w1.item())
                _b = float(net.feat.bias[0])
                _x = torch.arange(-6,6,0.1)
                _y = (-w0*_x-_b)/w1

                plt.clf() # plt.cla()
                plt.scatter(x0.data.numpy()[:,0], x0.data.numpy()[:,1], c='b', label='class 0')
                plt.scatter(x1.data.numpy()[:,0], x1.data.numpy()[:,1], c='r', label='class 1')
                plt.plot(_x,_y)
                plt.xlim(-7,7)
                plt.ylim(-7,7)
                plt.legend(loc='upper right')
                plt.text(-6,-6, "loss={:.3f}".format(loss))
                plt.title("iteration {}, acc={:.3f}".format(i,acc))

                if acc > 0.95:
                    print("--sigmoid done")
                    plt.show()
                    break
                else:
                    plt.pause(0.1)

    # end
    logistic_fn()
# end

def init_paras(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0., std=0.1)
            m.bias.data.zero_()
# end
class LeNetSeq(nn.Module):
    def __init__(self, class_num):
        super(LeNetSeq, self).__init__()
        self.feat = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,)),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2,stride=2)
        }))
        self.conv2 = nn.Conv2d(6,16,(5,))
        self.clsf = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(in_features=16*5*5, out_features=120),
            'relu3': nn.ReLU(inplace=True),
            'fc2': nn.Linear(in_features=120, out_features=80),
            'relu4': nn.ReLU(inplace=True),
            'fc3':nn.Linear(in_features=80, out_features=class_num),
        }))
        init_paras(self)
    # end

    def forward(self,x):
        x = self.feat(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = x.view(x.size(0),-1)
        x = self.clsf(x)
        return x
    # end
# end class

def cnn_model(ds='BIL'):
    if ds in ['MNIST', 'mnist']: data_root="/public/home/alex/1github/data"; ds = 'MNIST'
    elif ds in ['Chr', 'chr', 'BIL', 'bil', 'Bil']: data_root = "/public/home/alex/Docu/Dataset/ichr_bil_data/0528"; ds = 'BIL'
    elif ds in ['RMB', 'rmb']: data_root = os.path.join(agLOG.root, 'RMB_data', 'rmb_split'); ds = "RMB"
    else: data_root = './data'; ds = 'MNIST'
    if not os.path.exists(data_root): input3(f"NOT found dataset:{data_root}")

    mean, std = agDATA.mean, agDATA.std # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # Normalize is NOT mandatory in transforms

    repeat_c = 1; nw = 2; pm=False; size = 224; num_cls = 10; bs = 32 # nw=2 is the fastest for MNIST, BIL,
    if ds == 'MNIST':  # MNIST is 1 channel,  [1,32,32], we need 3 channels [3,32,32]
        repeat_c = 3
        tsfm_tra = transforms.Compose(
            [transforms.Resize((size, size)), T.RandomCrop(size, padding=int(size / 8)),
             transforms.ToTensor()])
        tsfm_val = transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor()])
        ds_train = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=tsfm_tra)
        ds_val = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=tsfm_val)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pm)
        dl_val = torch.utils.data.DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
    else:
        tsfm_tra = transforms.Compose(
            [transforms.Resize((size, size)), T.RandomCrop(size, padding=int(size / 8)), transforms.ToTensor(),
             T.Normalize(mean, std)]) # RandomCrop is NOT mandatory
        tsfm_val = transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor(), T.Normalize(mean, std)])
        if ds == 'RMB':
            num_cls = 2
            dir_train = os.path.join(data_root, 'train')
            dir_val, dir_test = dir_train.replace('train', 'val'), dir_train.replace('train', 'test')
            ds_train = RmbDataset(data_dir=dir_train, tsfm=tsfm_tra)
            ds_val = RmbDataset(data_dir=dir_val, tsfm=tsfm_val)
            dl_train = DataLoader(dataset=ds_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pm, drop_last=False)
            dl_val = DataLoader(dataset=ds_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm, drop_last=False)
        else:
            print(f" Chromosome is a long DNA molecule with part of ...")
            num_cls = 24
            ds_train = ChrDataset(agDIR.tra_cls, tsfm_tra)
            ds_val = ChrDataset(agDIR.val_cls, tsfm_val)
            dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pm)
            dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
    print('size of dataloader train {}, val {}'.format(len(dl_train, ), len(dl_val)))

    # num_classes = 24 if 'chr' in ds_train.data_info[0][0] else 10
    dev = "cuda:3" # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    def _run(dev):
        t1 = time.time()
        net = models.resnext50_32x4d().resnet50(pretrained=False, num_classes=num_cls) # LeNetSeq(class_num=2)
        net.to(dev)
        criterion = nn.CrossEntropyLoss()
        opt = optim.SGD(net.parameters(),lr=0.01, momentum=0.9)
        epoch_max = 100
        curve_train, curve_val, total_iter = list(),list(), 0
        loss, j, tt = 0.0, 0, time.time()
        print(f"start training of {ds} at {time_str()}, number_workers={nw}")
        for epoch in range(epoch_max):
            t_e = time.time()
            loss_train, correct_train, sample_train = 0., 0., 0.
            loss_v, correct_v, sample_v = 0., 0., 0.
            print(f"epoch [{epoch:03d}/{epoch_max:03d}]")
            net.train()
            for step,data in enumerate(dl_train):
                x, y = data
                x, y = x.repeat(1,repeat_c,1,1).to(dev), y.to(dev)
                y_ = net(x)
                loss = criterion(y_,y)
                loss.backward()
                opt.step()
                opt.zero_grad()
                loss_train += loss.item()
                curve_train.append(loss.item())
                _, pred = torch.max(y_.data,1)
                correct_train += (pred == y).squeeze().sum()
                sample_train += y.size(0)
                if step % 50 == 0 or step < 5:
                    loss2 = loss_train/sample_train
                    acc = correct_train/sample_train
                    print(f" Training Epoch {epoch + 1}/{epoch_max}, iteration[{step}/{len(dl_train)}], loss={loss2:.3f}, acc={acc:.3f}; at {time_str()}, cost {time_gap(t_e)}/{time_gap(t1)}")
                    # loss_train = 0

            if len(dl_val) < 1: continue
            t_v = time.time()
            net.eval()
            print(f"start training at {time_str()}")
            for j, data in enumerate(dl_val):
                x, y = data
                x, y = x.to(dev), y.to(dev)
                y_ = net(x)
                loss = criterion(y_,y)
                loss_v += loss.item()
                _, pred = torch.max(y_.data, 1)
                correct_v += (pred == y).squeeze().sum()
                sample_v += y.size(0)
            loss_v = loss_v/sample_v
            curve_val.append(loss.item())
            acc = correct_v/sample_v
            print(f"validating iteration [{j}/{len(dl_val)}], loss={loss_v:.3f}, acc={acc:.3f}\n done at {time_str()}, val cost {time_gap(t_v)}, total {time_gap(t1)} 寇可为，吾亦复可为；寇可往，吾亦复可往")

        plot_x_train = range(len(curve_train))
        plot_x_val = torch.arange(1, len(curve_val)+1)*(len(dl_train))
        plt.plot(plot_x_train, curve_train, label='class 0')
        plt.plot(plot_x_val, curve_val, label='class 1')
        plt.legend(loc='upper right')
        plt.xlabel('iteration'); plt.ylabel('loss value')
        fn = os.path.join(agDIR.temp, 'log', f'everyday_{agBASE.tm}.jpg')
        plt.savefig(fn); plt.show(); print(f'--done: {fn}')
    # end _run
    _run(dev)
# end cnn_model

LABEL = {'1': 0, '100': 1}
class IAdataset(Dataset):
    def __init__(self,data_dir, tsfm ):
        self.label_name = LABEL
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
                    lb = LABEL[dir]
                    data_info.append((fnp, int(lb)))
        return data_info
# end IAdataset

def toy(dev = "cuda:3", data_root="/public/home/alex/1github/data"): # './data'
    print(f"^_^ code from xinghuo.xfyun.cn ^_^")
    if not os.path.exists(data_root): input3(f"NOT found dataset:{data_root}")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.models import resnet50

    size, repeat_c, nw = 224, 3, 2 # nw=2 is the fastest for MNIST, BIL, RMB
    tsfm_tra = transforms.Compose([transforms.Resize((size, size)), T.RandomCrop(224, padding=28), transforms.ToTensor()])  # RandomCrop is NOT mandatory
    tsfm_val = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    train_set = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=tsfm_tra)
    test_set = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=tsfm_val)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=nw)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=nw)

    # 定义ResNet50模型
    model = resnet50(pretrained=False, num_classes=10)
    model = model.to(dev)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    t1 = time.time()
    print(f"start training of {'MNIST'} at {time_str()}, number_workers={nw}")
    # 训练模型
    num_epochs, step = 10, 0
    for epoch in range(num_epochs):
        t_e = time.time()
        model.train()
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.repeat(1,repeat_c,1,1).to(dev), labels.to(dev)
            optimizer.zero_grad()
            # input(f" [inputs.shape, labels.shape] = [{inputs.shape}; {labels.shape}] \n")
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / (step + 1)} at {time_str()}, cost {time_gap(t_e)}/{time_gap(t1)}")
        # 5.23min / 1 epoch for nw=2; pin_memory = False

    print(f"Finished Training at {time_str()}, cost {time_gap(t1)}")

    # 测试模型性能
    model.eval()
    correct = 0
    total = 0
    t_v = time.time()
    print(f"start training at {time_str()}")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f" validating done at {time_str()}, val cost {time_gap(t_v)}, total {time_gap(t1)} \n Accuracy: {100 * correct / total}%\n 寇可为，吾亦复可为；寇可往，吾亦复可往")
 # end toy

def do_backward(y_, lb, _feat, _LossCriterion, _opt_net):  # y_: the predicted yhat; lb: true label
    assert (y_.shape[0] == lb.shape[0]) and (y_.shape[0] == _feat.shape[0]), \
        "! check shape: {}, {}, {}".format(y_.shape, lb.shape, _feat.shape)
    yhat = y_.argmax(dim=1)  # .cpu().numpy()
    loss_dict_tensor = _LossCriterion(y_, lb, _feat)  # y_: the predicted yhat; lb: true label

    _opt_net.zero_grad()
    loss_dict_tensor['ce'].backward()
    _opt_net.step()

    loss_dict_np, _ = tensor_to_xx(loss_dict_tensor)
    return loss_dict_np

# net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Flatten(), nn.Linear(128 * 7 * 7, 1024), nn.ReLU(), nn.Linear(1024, 10))
def get_net(bb_tag=agTAG.bb_cls, num_cls=agDATA.num_cls, _prt=False):
    if 'RN50' in bb_tag: net = models.resnet50(pretrained=True) # , num_classes=num_cls
    elif 'NEXT50' in bb_tag: net = models.resnext50_32x4d(pretrained=True)
    else: input3(f"!!unexpect backbone:{bb_tag}({agNET.bb_cls})"); net = None
    net.fc = torch.nn.Linear(net.fc.in_features, num_cls)
    if _prt: summary(net, (3,224,224), -1, 'cpu')
    return net

def get_criterion():
    return nn.CrossEntropyLoss()

def get_optimizer(params, opti=agNET.opti, lr=agNET.lr, wd=agNET.wd, mome=agNET.mome):
    if 'SGD' == opti:
        opt = optim.SGD(params, lr=lr, weight_decay=wd, momentum=mome)
    elif 'AdamW' == opti:
        opt = optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        opt = optim.SGD(params, lr=lr, weight_decay=wd, momentum=mome)
        print(f"!! unexpected optimizer({opti}), set to default SGD\n\n\n")
    
    # sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: warmup_linear_decay(epoch, learning_rate, epoch_max))
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 60], gamma=0.1)

    return opt, sch

def train_cnn(dev, ds_train, ds_val, b_quick_debug=QUICK, _prt=True): # using agBASE, etc.
    # dev = 'cuda:3'
    bb_tag = agTAG.bb_cls # use tag to indicate k-fold
    num_cls, epoch_max, batch_size = agDATA.num_cls, agDATA.epoch_cls, agDATA.batch_cls # 64
    optimizer, learning_rate, weight_decay, momentum = agNET.opti, agNET.lr, agNET.wd, agNET.mome

    t0 = time.time()
    epoch, total_step = -1, 0
    fp_pth = agDIR.out_pth_cls
    fp_pth_last = agDIR.last_cls
    fnp_pth = ''

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=agGPU.worker_cls, pin_memory=agGPU.pin)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=agGPU.worker_cls, pin_memory=agGPU.pin)
    print(f"size of dataloader train {len(dl_train)}, val {len(dl_val)}")    

    assert (num_cls == 24) == ('chr' in ds_train.data_info[0][0]), f"!!num_cls={num_cls}, ds_train.data_info[0][0]={ds_train.data_info[0][0]}"

    batch_train = len(ds_train.data_info)/batch_size
    batch_train = int(batch_train) if (batch_train%1 == 0) else int(batch_train + 1)
    batch_val = len(ds_val.data_info) / batch_size
    batch_val = int(batch_val) if (batch_val % 1 == 0) else int(batch_val + 1)

    gap_print_train, gap_print_val, gap_save, gap_tb = int(len(dl_train) / 20), int(len(dl_val) / 5), 20, 60 # np.floor

    net = get_net(bb_tag, num_cls)
    net.to(dev)

    # 初始化日志记录器和可视化工具
    tb_writer = tb.SummaryWriter(log_dir=agDIR.out_tb_cls, comment=f'{bb_tag}', filename_suffix='tb')  # except train, as train is large
    fake_img = torch.rand(size=(1, agDATA.img_c, agDATA.img_h, agDATA.img_w))
    fake_img = fake_img.to(dev)
    if_print("adding graph to tensorboard ...", _prt)
    tb_writer.add_graph(net, fake_img)

    # 定义模型和优化器
    criterion = get_criterion()
    opt, sch = get_optimizer(net.parameters(), opti=optimizer, lr=learning_rate, wd=weight_decay, mome=momentum)

    write_txt([agLOG.tra_cls], f"* {bb_tag}, start training model at {time_str()}...")
    acc_best_train, acc_best_val, acc_last_tra, acc_last_val = 0.0, 0.0, 0.0, 0.0
    acc_best_train, acc_best_val, acc_last_tra, acc_last_val = 0.0, 0.0, 0.0, 0.0
    loss_best_train_d, loss_best_val_d = {'ce':0.0}, {'ce':0.0}
    loss_last_tra_d, loss_last_val_d = {'ce':0.0}, {'ce':0.0}
    for epoch in range(1, epoch_max+1):
        t_e = time.time()  # time of start this epoch, also time of start this training
        t_i = t_e  # !!
        e_acc_l_train, e_loss_ld_train = [], {'ce':[]}  # epoch_acc_list_train
        _this_lr = opt.state_dict()['param_groups'][0]['lr']
        write_txt([agLOG.tra_cls], f"\n*train_cnn() {bb_tag} start epoch[{epoch:03d}/{epoch_max:03d}], last val acc={acc_last_val:.5f}, lr={_this_lr:.5f}, time:{time_str()}")
        net.train()
        for step, (x, lb) in enumerate(dl_train):
            total_step += step
            x, lb = x.to(dev), lb.to(dev)
            y_ = net(x)
            #
            yhat = y_.argmax(dim=1)  # .cpu().numpy()
            loss = criterion(y_, lb)  # y_: the predicted yhat; lb: true label

            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()

            # calculate batch mean accuracy, and saved
            _values, pred = torch.max(y_.data, 1)
            assert yhat.any() == pred.any(), f"!!! train yhat={yhat.shape}, pred={pred.shape}\n"
            correct = torch.tensor(pred == lb).clone().detach().squeeze().cpu().sum() # torch.tensor(pred == lb).squeeze().cpu().sum().numpy() #
            _batch_acc = correct / lb.size(0)
            _batch_acc = np.round(_batch_acc, PRINT_DECIMAL)
            e_acc_l_train.append(_batch_acc)
            # calculate batch mean loss dict, and saved
            _bach_loss = np_mean(loss.item())
            e_loss_ld_train['ce'].append(_bach_loss)
            if np.isnan(_bach_loss):
                write_txt([agLOG.tra_cls], f"!!!loss={loss} in step={total_step + 1}, epoch={epoch}")
                tb_writer.close()
                return None

            tb_writer.add_scalars(f"batch mean accuracy{TM}", {'train': _batch_acc}, total_step)
            tb_writer.add_scalars(f"batch mean losses{TM}", {'train':_bach_loss}, total_step)
            tb_writer.add_scalars(f"batch time second{TM}", {'train': np.round(time.time() - t_i, 8)}, total_step)

            if (total_step + 1) % gap_print_train == 0 or (total_step + 1) < 5:
                # print("total_step:{}".format(total_step+1))
                tb_writer.add_scalars("learning rate{}".format(TM), {'lr': _this_lr}, total_step)
                tt = time.time()
                write_txt([agLOG.tra_cls], f" {bb_tag} {'train'} e[{epoch}/{epoch_max}]: step[{step + 1}/{len(dl_train)}], total={total_step}, acc={_batch_acc:.4f}, loss={_bach_loss:.4f}, time b{time_gap(tt - t_i)}, tr/v{time_gap(tt - t_e)}, total{time_gap(tt - t0)}")
                t_i = tt
                if b_quick_debug and step > 3: break
        acc_last_tra = np_mean(e_acc_l_train)
        loss_last_tra_d['ce'] = np_mean(e_loss_ld_train['ce'])
        sch.step(min(agNET.lr, loss_last_tra_d['ce']))  # pick the minor one in case loss llm
        tt = time.time()
        write_txt([agLOG.tra_cls], f"{bb_tag} Epoch[{epoch}/{epoch_max}] train acc={acc_last_tra}, loss={loss_last_tra_d}, {time_str()}, {time_gap(tt - t_e)}, total{time_gap(tt - t0)}, next epoch lr={opt.state_dict()['param_groups'][0]['lr']}, trained done at {time_str()}")
        tb_writer.add_scalars(f"1epoch mean accuracy{TM}", {'tra': acc_last_tra}, epoch)

        t_v = time.time()  # time for start this valid
        t_i = t_v  # !!
        e_acc_l_val = []  # epoch_acc_list_train
        e_loss_ld_val = {'ce':[]}
        x, _steps_train, _steps_val = None, (len(dl_train) - 1), (len(dl_val) - 1)
        _step_map_ratio, _start_val_step = _steps_train / _steps_val, total_step - _steps_train
        print("*train_cnn() starting  eval() ...", end='')
        net.eval()
        # with torch.no_grad():
        for step, data in enumerate(dl_val):
            _val_step = _start_val_step + int(step * _step_map_ratio)  # map steps from val to train
            x, lb = data
            x, lb = x.to(dev), lb.to(dev)
            # _hook_features = features.register_forward_hook(_Cnn.hook_features)
            with torch.no_grad():
                y_ = net(x)
            yhat = y_.argmax(dim=1)  # .cpu().numpy()
            # _loss_d1 = 'not need here'
            # y_, lb = y_.to('cpu'), lb.to('cpu') # !! do NOT do this!!

            # calculate batch mean accuracy, and saved
            _values, pred = torch.max(y_.data, 1)
            assert yhat.any() == pred.any(), f"!!! val yhat={yhat.shape}, pred={pred.shape}\n"
            correct = torch.tensor(pred == lb).squeeze().cpu().sum().numpy()
            _batch_acc = correct / lb.size(0)
            _batch_acc = np.round(_batch_acc, PRINT_DECIMAL)
            e_acc_l_val.append(_batch_acc)

            if (step + 1) % gap_print_val == 0 or step < 3:
                tt = time.time()
                write_txt([agLOG.val_cls],
                          f"{bb_tag} {'val'} e[{epoch}/{epoch_max}]: step[{step + 1}/{len(dl_val)}], total={_val_step}, acc={_batch_acc}, time b{time_gap(tt - t_i)}, tr/v{time_gap(tt - t_v)}, total{time_gap(tt - t0)}")
                t_i = tt
                if b_quick_debug and step > 3: break

        # epoch record for eval
        acc_last_val = np_mean(e_acc_l_val)
        write_txt([agLOG.tra_cls, agLOG.val_cls],
                  f"*train_cnn() {bb_tag} epoch [{epoch}/{epoch_max}] valid mean [train/val]: acc=[{acc_last_tra:.4f}/{acc_last_val}], loss=[{loss_last_tra_d['ce']}/{loss_last_val_d['ce']}]\n time train {time_gap(tt - t_e)}, val{time_gap(tt - t_v)}, total{time_gap(tt - t0)}.") # best: acc=[{acc_best_train}/{acc_best_val}], loss=[{loss_best_train_d['ce']}/{loss_best_val_d['ce']}] \n
        tb_writer.add_scalars(f"1epoch mean accuracy{TM}", {'val': acc_last_val}, epoch)
        if epoch % gap_save == 0 or epoch == epoch_max or (acc_last_val - acc_best_val) > 0.01:
            acc_best_val = acc_last_val if (acc_last_val - acc_best_val) > 0.01 else acc_best_val
            os.makedirs(fp_pth, exist_ok=True)
            ckpoint = {'net': net.state_dict(), 'acc': np.round(acc_last_val,4), 'epoch': epoch, 'step':total_step}
            if epoch < epoch_max: torch.save(ckpoint, osp.join(fp_pth, f"{bb_tag}acc={acc_last_val:.3f}e={epoch:04d}.pth"))
            else: # name with '_final' and save one more
                _fn = f"{bb_tag}acc={acc_last_val:.3f}e={epoch:03d}_final.pth"
                write_txt([agLOG.tra_cls, agLOG.val_cls, agLOG.log_all_cls], f"** train_cnn[{TM}] last .pth fn:{_fn}")
                torch.save(ckpoint, osp.join(fp_pth, _fn))
                fnp_pth = osp.join(fp_pth_last, _fn) # for return
                os.makedirs(fp_pth_last, exist_ok=True)
                torch.save(ckpoint, fnp_pth)
    tb_writer.close()
    write_txt([agLOG.tra_cls, agLOG.val_cls, agLOG.log_all_cls], f"*train_cnn() {bb_tag}, llm_cnn -> train_cnn() done with acc={acc_last_val}, epoch={epoch}, at {time_str()}, total time {time_gap(time.time() - t0)},\n output:{agDIR.out_cls}\n")
    return fnp_pth
# end train_cnn

def eval_cnn(dev, fnp_pth, ds_val, _find_in_last='', b_quick_debug=QUICK, _prt=True): # using agBASE, etc.
    t_v = time.time() # time for start this valid
    t_i = t_v  # !!
    e_acc_l_val = []  # epoch_acc_list_train
    e_loss_ld_val = {'ce': []}

    write_txt([agLOG.val_cls], f"*start eval at:{time_str()}")
    bb_tag = agTAG.bb_cls # use tag to indicate k-fold
    num_cls = agDATA.num_cls
    batch_size = agDATA.batch_cls # 64

    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=agGPU.worker_cls, pin_memory=agGPU.pin)
    gap_print_val = int(len(dl_val) / 5)
    print(f"size of dataloader val {len(dl_val)}")

    if not fnp_pth:
        _base = ['.pth', bb_tag]
        if type(_find_in_last) is str: _find_in_last = [_find_in_last]
        fnp_pth = find_sth_easy(agDIR.last_cls, _base + _find_in_last, _confirm=False)
        if fnp_pth is None:
            input3("!! skip eval_mm, as not found '.pth'\n") # [args.fnp_log_eval], 
            return _find_in_last, fnp_pth
    else:
        assert osp.exists(fnp_pth), f"!! can't find: {fnp_pth}"

    net = get_net(bb_tag, num_cls)
    state_dict = torch.load(fnp_pth, map_location=dev)
    if 'fc.weight' in state_dict and agDATA.num_cls != state_dict['fc.weight'].shape[0]:
        del state_dict['fc.weight']
        del state_dict['fc.bias']
    net.load_state_dict(state_dict['net'], strict=False)
    write_txt([agLOG.val_cls], f"* {bb_tag} loaded pth {os.path.basename(fnp_pth)}\n with acc={state_dict['acc']}, starting eval() at {time_str()}...", True)

    net.to(dev)
    net.eval()
    # with torch.no_grad():
    for step, data in enumerate(dl_val):
        x, lb = data
        x, lb = x.to(dev), lb.to(dev)
        # _hook_features = features.register_forward_hook(_Cnn.hook_features)
        with torch.no_grad():
            y_ = net(x)
        yhat = y_.argmax(dim=1)  # .cpu().numpy()
        # _loss_d1 = 'not need here'
        # y_, lb = y_.to('cpu'), lb.to('cpu') # !! do NOT do this!!

        # calculate batch mean accuracy, and saved
        _values, pred = torch.max(y_.data, 1)
        correct = torch.tensor(pred == lb).squeeze().cpu().sum().numpy()
        _batch_acc = correct / lb.size(0)
        _batch_acc = np.round(_batch_acc, PRINT_DECIMAL)
        e_acc_l_val.append(_batch_acc)

        if (step + 1) % gap_print_val == 0 or step < 3:
            tt = time.time()
            write_txt([agLOG.val_cls],
                        f" Only {'val'}: step[{step + 1}/{len(dl_val)}], acc={_batch_acc}, time b{time_gap(tt - t_i)}, tr/v{time_gap(tt - t_v)}")
            t_i = tt
            if b_quick_debug and step > 3: break

    # epoch record for eval
    acc_last_val = np_mean(e_acc_l_val)
    write_txt([agLOG.val_cls, agLOG.log_all_cls], f"*eval_cnn() {bb_tag} Only Valid mean val: acc={acc_last_val}, cost{time_gap(t_v)}")
# end eval_cnn


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

if __name__ == '__main__':
    tt0 = datetime.datetime.now() # only here; other use time.time()
    # linear_model()
    # sigmoid_model()
    # cnn_model('BIL')
    # toy()
    dev = 'cuda:3'

    tsfm_tra = set_transform(True)
    tsfm_val = set_transform(False)
    ds_train = ChrDataset(agDIR.tra_cls, tsfm_tra)
    ds_val = ChrDataset(agDIR.val_cls, tsfm_val)
    fnp_pth = train_cnn(dev, ds_train, ds_val)
    eval_cnn(dev, fnp_pth, ds_val)

    print(f"- main of llm_cnn.py done at {agBASE.tm}, from {time_str(tt0)}, cost {time_gap(tt0)}")
