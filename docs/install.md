Make sure you have installed cuda > 10.0
```shell
conda create -n LD python=3.7 -y
conda activate LD
conda install pytorch=1.5 cudatoolkit=10.1 torchvision -c pytorch
pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html
pip install -v -e .
```
