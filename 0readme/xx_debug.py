import torch
import mmcv
from mmengine.visualization import Visualizer

load_from = '/public/home/alex/1github/mm/mmclss/configs/_mb321/clip_resnet50_cifar.py'
fnp_img = 'a:/1github/data/img/lena.png' # 'docs/en/_static/image/cat_and_dog.png'

# https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/en/_static/image/cat_and_dog.png
image = mmcv.imread(fnp_img, channel_order='rgb')
visualizer = Visualizer(image=image)
# single bbox formatted as [xyxy]
visualizer.draw_bboxes(torch.tensor([72, 13, 179, 147]))
# draw multiple bboxes
visualizer.draw_bboxes(torch.tensor([[33, 120, 209, 220], [72, 13, 179, 147]]))
visualizer.show()