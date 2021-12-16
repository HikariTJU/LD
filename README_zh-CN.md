# Localization Distillation for Dense Object Detection 

### 本代码基于mmDetection框架.

这是我们论文的代码实现
 - [Localization Distillation for Dense Object Detection](https://arxiv.org/abs/2102.12252)

[2021.3.30] LD被集成至[MMDetection V2](https://github.com/open-mmlab/mmdetection/tree/master/configs/ld), 感谢 [@jshilong
](https://github.com/jshilong), [@Johnson-Wang](https://github.com/Johnson-Wang)与[@ZwwWayne](https://github.com/ZwwWayne).

LD是知识蒸馏在目标定位任务上的拓展，利用教师所学习到的定位分布来指导学生的定位学习。

LD可稳定提高密集检测器约2.0 AP，几乎无附加计算消耗！

## 简介

[English](README.md) | 简体中文

知识蒸馏(KD)在目标检测中具有强大的学习紧凑模型的能力。以往的目标检测KD方法大多聚焦于在特定区域内进行深层特征模仿，而非在分类头上进行logit模仿，因为后者提取定位信息的效率低。
本文通过重新定制定位头的知识蒸馏过程，提出了一种新的定位蒸馏(LD)方法，可以有效地将定位知识从教师传递给学生。
此外，我们还启发性地引入了有价值定位区域(VLR)的概念，有助于在特定区域内有选择性地蒸馏语义知识和定位知识。
结合这两个新的组件，我们首次展示了**logit模仿可以优于特征模仿，并且对于蒸馏目标检测器来说，定位知识的蒸馏要比语义知识更重要、更有效。**
我们的蒸馏方案简单有效，可以很容易地应用于不同的密集目标检测器。
实验表明，我们的LD可以在不牺牲推理速度的情况下，在COCO基准上，将训练1x的GFocalV1-ResNet-50的AP从40.1提高到42.1。

<img src="LD.png" height="220" align="middle"/>


## 环境安装

参见[INSTALL.md](docs/install.md), 里面包含了安装和数据集准备。我们使用的是Pytorch=1.5.1与cudatoolkits=10.1。

## 快速入门

参见[GETTING_STARTED.md](docs/getting_started.md), 包含了MMDetection的基本使用方法。

## 训练

```python
# 假设你已经在本项目代码的根目录下,
# 假设你已经激活了所需要的虚拟环境,
# 且COCO数据集已放置于'data/coco/'下。

./tools/dist_train.sh configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py 8
```

#### 学习率调整

`lr=(samples_per_gpu * num_gpu) / 16 * 0.01`

对于使用2块GPUS，mini-batch大小为6张图，你应修改config文件如下:

```python
optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=3,
```

对于使用8块GPUS，mini-batch大小为16张图，你应修改config文件如下:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=2,
```

#### 特征模仿方法

我们提供了一些特征模仿方法的实现，包括了FitNets `fitnet`, DeFeat `decouple`, Fine-Grained `finegrain`, GI location `gibox`。
```python
    bbox_head=dict(
        loss_im=dict(type='IMLoss', loss_weight=2.0),
        imitation_method='finegrained'  # gibox, finegrain, decouple, fitnet
    )
```
## 转换模型

如果你发现蒸馏训练后的模型大小很大，你可以转换模型，参见[publish_model.py](tools/model_converters/publish_model.py)

```python
python tools/model_converters/publish_model.py your_model.pth your_new_model.pth
```

## 速度测试 (FPS)

```python
CUDA_VISIBLE_DEVICES=0 python3 ./tools/benchmark.py configs/ld/ld_gflv1_r101_r50_fpn_coco_1x.py work_dirs/ld_gflv1_r101_r50_fpn_coco_1x/epoch_24.pth
```

## 评估

```python
./tools/dist_test.sh configs/ld/ld_gflv1_r101_r50_fpn_coco_1x.py work_dirs/ld_gflv1_r101_r50_fpn_coco_1x/epoch_24.pth 8 --eval bbox
```

<details open>
<summary>COCO</summary>
 
- **LD for Lightweight Detectors**
 
  Evaluate on the main distillation region only.
  |  Teacher  | Student | Training schedule  | AP (val) | AP50 (val) | AP75 (val) | AP (test-dev) | AP50 (test-dev) | AP75 (test-dev) | AR100 (test-dev) |
  | :-------: | :-----: | :---------------:  | :------: | :--------: | :--------: | :-----------: | :-------------: | :-------------: | :--------------: |
  |    --     |  R-18   |        1x         |   35.8   |    53.1    |    38.2    |     36.0      |      53.4       |      38.7       |       55.3       |
  |   R-101   |  R-18   |        1x         |   36.5   |    52.9    |    39.3    |     36.8      |      53.5       |      39.9       |       56.6       |
  |    --     |  R-34   |        1x         |   38.9   |    56.6    |    42.2    |     39.2      |      56.9       |      42.3       |       58.0       |
  |   R-101   |  R-34   |        1x         |   39.8   |    56.6    |    43.1    |     40.0      |      57.1       |      43.5       |       59.3       |
  |    --     |  R-50   |        1x         |   40.1   |    58.2    |    43.1    |     40.5      |      58.8       |      43.9       |       59.0       |
  |   R-101   |  R-50   |        1x         |   41.1   |    58.7    |    44.9    |     41.2      |      58.8       |      44.7       |       59.8       |
  |    --     |  R-101  |        2x         |   44.6   |    62.9    |    48.4    |     45.0      |      63.6       |      48.9       |       62.3       |
  | R-101-DCN |  R-101  |        2x         |   45.4   |    63.1    |    49.5    |     45.6      |      63.7       |      49.8       |       63.3       |
 
- **Self-LD**
 
  Evaluate on the main distillation region only.
  |     Teacher     |     Student     | Training schedule | AP (val) | AP50 (val) | AP75 (val) |
  | :-------------: | :-------------: | :---------------: | :------: | :--------: | :--------: |
  |       --        |      R-18       |        1x         |   35.8   |    53.1    |    38.2    |
  |      R-18       |      R-18       |        1x         |   36.1   |    52.9    |    38.5    |
  |       --        |      R-50       |        1x         |   40.1   |    58.2    |    43.1    |
  |      R-50       |      R-50       |        1x         |   40.6   |    58.2    |    43.8    |
  |       --        | X-101-32x4d-DCN |        1x         |   46.9   |    65.4    |    51.1    |
  | X-101-32x4d-DCN | X-101-32x4d-DCN |        1x         |   47.5   |    65.8    |    51.8    |

- **Logit Mimicking vs. Feature Imitation**
 
  **Ours** = Main KD + Main LD + VLR LD. ''Main'' denotes the main distillation region, ''VLR'' denotes the valuable localization region.
  | Method | Training schedule  | AP (val) | AP50 (val) | AP75 (val) | APs (val) | APm (val) | APl (val) |
  | :-------: | :---------------:  | :------: | :--------: | :--------: | :-----------: | :-------------: | :-------------: |
  |       --        |        1x         |   40.1   |    58.2    |    43.1    |     23.3      |      44.4       |      52.5       |
  |     FitNets     |        1x         |   40.7   |    58.6    |    44.0    |     23.7      |      44.4       |      53.2       |
  |  Inside GT Box  |        1x         |   40.7   |    58.6    |    44.2    |     23.1      |      44.5       |      53.5       |
  |   Main Region   |        1x         |   41.1   |    58.7    |    44.4    |     24.1      |      44.6       |      53.6       |
  |   Fine-Grained  |        1x         |   41.1   |    58.8    |    44.8    |     23.3      |      45.4       |      53.1       |
  |      DeFeat     |        1x         |   40.8   |    58.6    |    44.2    |     24.3      |      44.6       |      53.7       |
  |   GI Imitation  |        1x         |   41.5   |    59.6    |    45.2    |     24.3      |      45.7       |      53.6       |
  |    **Ours**     |        1x         |   42.1   |    60.3    |    45.6    |     24.5      |      46.2       |      54.8       |
</details>

```python
./tools/dist_test.sh configs/ld/ld_gflv1_r101_r18_fpn_voc.py work_dirs/ld_gflv1_r101_r18_fpn_voc/epoch_4.pth 8 --eval mAP
```
<details open>
<summary>PASCAL VOC</summary>
 
- **LD for Lightweight Detectors**
 
  Evaluate on the main distillation region only.
  |  Teacher  | Student | Training Epochs  |  AP   | AP50  | AP75  |
  | :-------: | :-----: | :-------------:  | :---: | :---: | :---: |
  |    --     |  R-18   |        4        | 51.8  | 75.8  | 56.3  |
  |   R-101   |  R-18   |        4        | 53.0  | 75.9  | 57.6  |
  |    --     |  R-50   |        4        | 55.8  | 79.0  | 60.7  |
  |   R-101   |  R-50   |        4        | 56.1  | 78.5  | 61.2  |
  |    --     |  R-34   |        4        | 55.7  | 78.9  | 60.6  |
  | R-101-DCN |  R-34   |        4        | 56.7  | 78.4  | 62.1  |
  |    --     |  R-101  |        4        | 57.6  | 80.4  | 62.7  |
  | R-101-DCN |  R-101  |        4        | 58.4  | 80.2  | 63.7  |

  This is an example of evaluation results (R-101→R-18).
  ```
  +-------------+------+-------+--------+-------+
  | class       | gts  | dets  | recall | ap    |
  +-------------+------+-------+--------+-------+
  | aeroplane   | 285  | 4154  | 0.081  | 0.030 |
  | bicycle     | 337  | 7124  | 0.125  | 0.108 |
  | bird        | 459  | 5326  | 0.096  | 0.018 |
  | boat        | 263  | 8307  | 0.065  | 0.034 |
  | bottle      | 469  | 10203 | 0.051  | 0.045 |
  | bus         | 213  | 4098  | 0.315  | 0.247 |
  | car         | 1201 | 16563 | 0.193  | 0.131 |
  | cat         | 358  | 4878  | 0.254  | 0.128 |
  | chair       | 756  | 32655 | 0.053  | 0.027 |
  | cow         | 244  | 4576  | 0.131  | 0.109 |
  | diningtable | 206  | 13542 | 0.150  | 0.117 |
  | dog         | 489  | 6446  | 0.196  | 0.076 |
  | horse       | 348  | 5855  | 0.144  | 0.036 |
  | motorbike   | 325  | 6733  | 0.052  | 0.017 |
  | person      | 4528 | 51959 | 0.099  | 0.037 |
  | pottedplant | 480  | 12979 | 0.031  | 0.009 |
  | sheep       | 242  | 4706  | 0.132  | 0.060 |
  | sofa        | 239  | 9640  | 0.192  | 0.060 |
  | train       | 282  | 4986  | 0.142  | 0.042 |
  | tvmonitor   | 308  | 7922  | 0.078  | 0.045 |
  +-------------+------+-------+--------+-------+
  | mAP         |      |       |        | 0.069 |
  +-------------+------+-------+--------+-------+
  AP:  0.530091167986393
  ['AP50: 0.759393', 'AP55: 0.744544', 'AP60: 0.724239', 'AP65: 0.693551', 'AP70: 0.639848', 'AP75: 0.576284', 'AP80: 0.489098', 'AP85: 0.378586', 'AP90: 0.226534', 'AP95:   0.068834']
  {'mAP': 0.7593928575515747}
  ```
</details>
 
#### Note:
 - 更多的实验细节可参阅[GFocalV1](https://github.com/implus/GFocal), [GFocalV2](https://github.com/implus/GFocalV2) and [mmdetection](https://github.com/open-mmlab/mmdetection).

## 预训练权重

#### VOC 07+12
##### GFocal V1 

[pan.baidu](https://pan.baidu.com/s/16s-ae8GyWNZFPO_vyGqmUA) pw: ufc8, teacher R101

[pan.baidu](https://pan.baidu.com/s/1PseEYuQG_WTCSuqoMSIcYQ) pw: 5qra, teacher R101DCN

[pan.baidu](https://pan.baidu.com/s/1Jd1UjfuxLi8MGD1JKruWxw) pw: 1bd3, Main LD R101→R18, box AP = 53.0

[pan.baidu](https://pan.baidu.com/s/13QWRpSEvgPx_Y7_06iuBEQ) pw: thuw, Main LD R101DCN→R34, box AP = 56.5

[pan.baidu](https://pan.baidu.com/s/10IZRaFc1FfqoSAmr7iJPcg) pw: mp8t, Main LD R101DCN→R101, box AP = 58.4

[GoogleDrive](https://drive.google.com/file/d/1RoHEnCiGkCve_g9Fi5DBBjOmScX1CrXI/view?usp=sharing) Main LD + VLR LD + VLR KD R101→R18, box AP = 54.0

[GoogleDrive](https://drive.google.com/file/d/1hMgJs5rCN-PwyLlfU_GP3w-b4UX4_8UP/view?usp=sharing) Main LD + VLR LD + VLR KD + GI imitation R101→R18, box AP = 54.4

#### COCO
##### GFocal V1 

[pan.baidu](https://pan.baidu.com/s/1C34l6cMf5f1AViKGYXmeKw) pw: hj8d, Main LD R101→R18 1x, box AP = 36.5

[pan.baidu](https://pan.baidu.com/s/16OLOcPBqgEf8Mh3ljOKPVA) pw: bvzz, Main LD R101→R50 1x, box AP = 41.1

[GoogleDrive](https://drive.google.com/file/d/1gBnE0MsfxsXfSZHLF6LGdL-I-tcOVdeU/view?usp=sharing) Main KD + Main LD + VLR LD R101→R18 1x, box AP = 37.5

[GoogleDrive](https://drive.google.com/file/d/100otwiOF2w7iO18t4ACxGmABOYHvhSLG/view?usp=sharing) Main KD + Main LD + VLR LD R101→R34 1x, box AP = 41.0

[GoogleDrive](https://drive.google.com/file/d/1MPSuJ3TpM5mJk1m4cFjJpKVQOeTU12nP/view?usp=sharing) Main KD + Main LD + VLR LD R101→R50 1x, box AP = 42.1

[GoogleDrive](https://drive.google.com/file/d/1ryn5bYEf5yx1UXVHQ3iUucnecarM8T3m/view?usp=sharing) Main KD + Main LD + VLR LD + GI imitation R101→R50, box AP = 42.4

##### GFocal V2

[GoogleDrive](https://drive.google.com/file/d/136MMvGSf-qANEE9xpbvYfnw8tRlscqHf/view?usp=sharing) Main KD + Main LD + VLR LD R101→R50 1x, box AP = 42.7

[GoogleDrive](https://drive.google.com/file/d/1n4rFLacovKm6PbhKK1QTBJK2scmqvmOl/view?usp=sharing) &#124; [Training log](https://drive.google.com/file/d/1n4rFLacovKm6PbhKK1QTBJK2scmqvmOl/view?usp=sharing) Main KD + Main LD + VLR LD R101-DCN→R101 2x, box AP (test-dev) = 47.1

[GoogleDrive](https://drive.google.com/file/d/1_Y6Mm_SlaBgY1oSyz8Vaobl7tmMyOJQ9/view?usp=sharing) &#124; [Training log](https://drive.google.com/file/d/1I8Fmor_t0YhV9txwDLbuLUYODoowjdmm/view?usp=sharing) Main KD + Main LD + VLR LD Res2Net101-DCN→X101-32x4d-DCN 2x, box AP (test-dev) = 50.5
#### 其他的教师模型，可于[GFocalV1](https://github.com/implus/GFocal), [GFocalV2](https://github.com/implus/GFocalV2)以及[mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/configs/gfl/README.md)中下载。

## Score voting Cluster-DIoU-NMS
我们提供了Score voting [Cluster-DIoU-NMS](https://github.com/Zzh-tju/CIoU), 这是一种加速版的score voting NMS, 并且与DIoU-NMS相结合。
对于GFocalV1与GFocalV2, Score voting Cluster-DIoU-NMS可带来0.1-0.3 AP提升, 0.2-0.5 AP75提升, <=0.4 AP50下降, 以及<=1.5 FPS下降。相关config文件改动如下:

```
# Score voting Cluster-DIoU-NMS
test_cfg = dict(
nms=dict(type='voting_cluster_diounms', iou_threshold=0.6),

# Original NMS
test_cfg = dict(
nms=dict(type='nms', iou_threshold=0.6),
```

## 引用

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 LD。

```
@Article{zheng2021LD,
  title={Localization Distillation for Dense Object Detection},
  author= {Zheng, Zhaohui and Ye, Rongguang and Wang, Ping and Ren, Dongwei and Zuo, Wangmeng and Hou, Qibin and Cheng, Mingming},
  journal={arXiv:2102.12252},
  year={2021}
}
```
