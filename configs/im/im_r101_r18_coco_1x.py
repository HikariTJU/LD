_base_ = ['../ld/ld_r18_gflv1_r101_fpn_coco_1x.py']
model = dict(
    output_feature=True,
    bbox_head=dict(
        type='IMHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
        loss_im=dict(type='IMLoss', loss_weight=0.2),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
)
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
)
optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001)