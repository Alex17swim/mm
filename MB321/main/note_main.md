# task
## test without ground truth annotation
* install and active the mmlab environment
* pip install labelme

# Data
## k-fold
In sz_data, 85 cases contributes 85/841=10% --> k=10, 10-fold
1. separate 841 data into 10 sub-set
2. pick 1 sub-set for 1-fold. as "make abn250", the ichr is picked first, then mchr:
   2.1 cd abn_ichr_sz_data/k_fold
   2.2 manually split 'abn0_841' into 10 folders, each 84 cases except f10 with 85
   2.3 
3. do it 10 times.

## excel
*行转列：Ctrl + F  --> 查找目标：填写指定的内容 --> 替换为：\r\n --> 查找模式：正则表达式
列转行：Ctrl + F  --> 查找目标：\r\n --> 替换为：不填写或填写指定的内容--> 查找模式：正则表达式
单击替换或全部替换按钮*

# coding
## conv
* bchw = [-1=batch_size, in_channels=c=3, in=h=224, in=w=224]
**所有 summary(net, (3,224,224), -1, 'cpu') 打印输出维度含义均如此**
* nn.Conv2d(in_channels, out_channels, kernel_size, stride)
*out_dim需要计算:d=[(in+2p-k)/stride]+1* **这里的in不是in_channels=3, 而是in=224**
*in_channels来自上一层任意输出; out_channels来自本层任意指定;*
torch.nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1) # 这里的in和out_dim都没有显示!