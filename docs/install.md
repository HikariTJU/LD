Make sure you have installed cuda > 10.0
```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
conda install pytorch=1.5 cudatoolkit=10.1 torchvision -c pytorch
pip install mmcv-full==1.2.7+torch1.5.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
pip install -v -e .
```
