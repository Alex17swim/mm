* for windows:
virtualenv -p "C:\Program Files\Python\Python38\python.exe" w_mm231107
cd A:\env\w_mm231107\Scripts\
.\activate
(w_mm231107) cd A:\env\whl\911win

* for ubuntu:
virtualenv -p /usr/bin/python3 mm231107
source ~/env/gpt/bin/activate
(w_mm231107) cd ~/env/whl/911ubuntu

pip install torch-1.10.0+cu113-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.11.1+cu113-cp38-cp38-linux_x86_64.whl 
pip install ftfy regex tqdm scikit-learn matplotlib tensorboard torchsummary

(w_mm231107) pip install git+https://github.com/openai/CLIP.git
or
(w_mm231107) cd A:\env\whl\CLIP-main@20231107
(w_mm231107) python setup.py install

# Debug
## 0906
* nw=2
Epoch 1, Loss: 0.608996759014825 at 0906_153145, cost : 5.23 minutes/: 5.23 minutes
Epoch 2, Loss: 0.085598953546832 at 0906_153701, cost : 5.25 minutes/: 10.50 minutes
  
* nw=0
Epoch 1, Loss: 0.573245075930655 at 0906_161406, cost : 32.85 minutes/: 32.85 minutes
Epoch 2, Loss: 0.0830588623555998 at 0906_164936, cost : 35.48 minutes/: 1.14 hours 

* UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.pause(0.1)
