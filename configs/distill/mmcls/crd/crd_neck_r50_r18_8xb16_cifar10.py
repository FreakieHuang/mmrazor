_base_ = [
    'mmcls::_base_/datasets/cifar10_bs16.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        bgr_to_rgb=True),
    architecture=dict(
        cfg_path='mmcls::resnet/resnet18_8xb16_cifar10.py', pretrained=False),
    teacher=dict(
        cfg_path='mmcls::resnet/resnet50_8xb16_cifar10.py', pretrained=True),
    teacher_ckpt='resnet50_b16x8_cifar10_20210528-f54bfad9.pth',
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            neck=dict(type='ModuleOutputs', source='neck.gap'),
            data_samples=dict(type='ModuleInputs', source='')),
        teacher_recorders=dict(
            neck=dict(type='ModuleOutputs', source='neck.gap')),
        distill_losses=dict(loss_crd=dict(type='CRDLoss', loss_weight=0.8)),
        connectors=dict(
            loss_crd_stu=dict(type='CRDConnector', dim_in=512, dim_out=128),
            loss_crd_tea=dict(type='CRDConnector', dim_in=2048, dim_out=128)),
        loss_forward_mappings=dict(
            loss_crd=dict(
                s_feats=dict(from_student=True, recorder='neck'),
                t_feats=dict(from_student=False, recorder='neck'),
                data_samples=dict(
                    from_student=True, recorder='data_samples', data_idx=1)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

# change `CIFAR10` dataset to `CRD_CIFAR10` dataset.
dataset_type = 'CRD_CIFAR10'
train_pipeline = [
    dict(_scope_='mmcls', type='RandomCrop', crop_size=32, padding=4),
    dict(_scope_='mmcls', type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(_scope_='mmrazor', type='PackCRDClsInputs'),
]

test_pipeline = [
    dict(_scope_='mmrazor', type='PackCRDClsInputs'),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=16,
    num_workers=2,
    dataset=dict(
        _scope_='mmrazor',
        type=dataset_type,
        data_prefix='data/cifar10',
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    _delete_=True,
    batch_size=16,
    num_workers=2,
    dataset=dict(
        _scope_='mmrazor',
        type=dataset_type,
        data_prefix='data/cifar10/',
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
