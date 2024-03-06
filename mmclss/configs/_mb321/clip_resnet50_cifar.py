print(f" *here is mm_cfg: clip_resnet50_cifar.py ")
_base_ = [
    # '../_base_/models/resnet50.py',
    'mb321_base_cls.py'
    ] # _base_

from MB321.mm_flag import agDATA # , agNET, agMM, agDIR, agTAG
num_cls = agDATA.num_cls



print(f"clip_resnet50_cifar.py, ")
