## 介绍

本代码  A Multi-Scale Localization and Feature Refinement Network for Camouflaged Object Detection  论文代码。下面按照目录结构，训练和测试介绍。

采用目前四个主流的COD公开数据集：CAMO、COD10K、CHAMELEON和NC4K

## 目录结构

> .
> |-- MyTesting.py		测试代码
> |-- MyTrain_Val.py		训练代码
> |-- eval_sod.py		评估代码
> |-- lib
> |   |-- EDNet_v2.py		网络代码
> |   |-- Modules_v2.py		模块代码
> |   |-- backbone		主干网络代码
> |   |-- ops		多头可变注意力机制相关包
> |-- pretrain		主干网络预训练权重
> |   |-- resnet50_ram-a26f946b.pth
> |-- readme.md
> |-- snapshot		运行生成的权重
> |   |-- res2net50
> |   |-- resnet50
> |-- utils		一些工具类
>     |-- FeatureViz.py
>     |-- MyFeatureVisulization.py
>     |-- cod10k_subclass_split.py
>     |-- data_val.py		Dataset的代码
>     |-- dataloader.py
>     |-- fps.py
>     |-- generate_LaTeX.py
>     |-- heatmap.py
>     |-- pytorch_jittor_convert.py
>     |-- tif2png.py
>     |-- utils.py		一些工具类

## 训练

运行 `python MyTrain_Val.py`,可以自行修改一些超参数

```python
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=800, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
parser.add_argument('--trainsize', type=int, default=576, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--train_root', type=str, default='/data0/hcm/dataset/COD/TrainDataset/',
                    help='the training rgb images root')
parser.add_argument('--val_root', type=str, default='/data0/hcm/dataset/COD/TestDataset/COD10K/',
                    help='the test rgb images root')
parser.add_argument('--save_path', type=str, 
                    default='./snapshot/EDNet_V2_576/',
                    help='the path to save model and log')
opt = parser.parse_args()
```

## 测试

运行`python MyTesting.py`.

## 评估

运行 `python eval_sod.py`