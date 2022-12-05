Make sure you have installed cuda > 11.0
```shell
conda create -n LD python=3.7 -y
conda activate LD
conda install pytorch=1.7 cudatoolkit=11 torchvision -c pytorch
pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
pip install -v -e .
```
