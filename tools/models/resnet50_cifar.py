# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    'mmcls::resnet/resnet18_8xb16_cifar10.py',
]

model = dict(
    _delete_=True,
    type='mmrazor.FXModelWrapper',
    model = _base_.model
)