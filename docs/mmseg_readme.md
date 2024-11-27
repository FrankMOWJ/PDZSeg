## Overview

There are over 5 steps to reproduce the baseline mentioned in our paper.

- Prepare the dataset in a format compatible with MMSegmentation.

- Initialize the dataset definition under `mmseg/datasets`.

- Define the data loading process under `configs/base/datasets`.

- Modify the model parameters as needed in the configuration files under `configs/`.

- Train with `tools/train.py`.

## Get MMsegmentation

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
```

## Prepare the dataset 

The original dataset contains 3 classes, we have to first convert 3-classes to 2 classes. please refer to `utils/create_dataset_mmseg.py`.

## Define ESD_SEG dataset class

Create a file named `ESD_SEG.py` under `mmsegmentation/mmseg/datasets/`, copy the following code in the file.

```python
# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ESD_SEG(BaseSegDataset):
    """ESD_SEG dataset.

    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """
    METAINFO = dict(
        classes=('background', 'cuttingArea'),
        palette=[[0, 0, 0], [255, 0, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args)
```

## Define data loading

Create a file named `ESD_SEG.py` under `mmsegmentation/configs/_base_/datasets/` and copy the following code into the file.

```python
# dataset settings
# make sure that the type name should be the same with the name of the dataset class in the above step.
dataset_type = 'ESD_SEG'
# TODO
data_root = '/PATH/TO/DATASET' 

crop_size = (532, 532)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = val_evaluator

```

##  Modify the model parameters

Take training `Fast-SCNN` for example, create a file `fast_scnn_8xb4-ESD_SEG-532x532.py` under `mmsegmentation/configs/fastscnn/`.

```python
_base_ = [
    '../_base_/models/fast_scnn.py', '../_base_/datasets/ESD_SEG.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (532, 532)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(data_preprocessor=data_preprocessor,
             decode_head=dict(
                type='DepthwiseSeparableFCNHead',
                in_channels=128,
                channels=128,
                concat_input=False,
                num_classes=2,
                in_index=-1,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1)),
            auxiliary_head=[
                dict(
                    type='FCNHead',
                    in_channels=128,
                    channels=32,
                    num_convs=1,
                    num_classes=2,
                    in_index=-2,
                    norm_cfg=norm_cfg,
                    concat_input=False,
                    align_corners=False,
                    loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
                dict(
                    type='FCNHead',
                    in_channels=64,
                    channels=32,
                    num_convs=1,
                    num_classes=2,
                    in_index=-3,
                    norm_cfg=norm_cfg,
                    concat_input=False,
                    align_corners=False,
                    loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
            ])
# Re-config the data sampler.
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-5)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
```

## Train

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py [CONFIG_FILE_PATH]
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/fastscnn/fast_scnn_8xb4-ESD_SEG-532x532.py
```

## Test and visualization

The training log and checkpoint will be saved at `mmsegmentation/work_dirs`, to test on test set, use the following demand.

```shell
CUDA_VISIBLE_DEVICES=0 python tools/test.py [CONFIG_FILE_PATH] [CHECKPOINT_PATH]
```

if you want to visualization the results, you can use the code in `mmsegmentation/demo/image_demo.py`.
