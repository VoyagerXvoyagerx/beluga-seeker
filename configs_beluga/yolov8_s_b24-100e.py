_base_ = 'mmyolo::yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

# ======================= Modified parameters =====================
work_dir = 'work_dirs/yolov8_s_sam_repeat_10_BUF_7_1218_0137'
# -----data related-----
data_root = './datasets/beluga/'
train_ann_file = 'annotations/split_sam_BUF_7/train.json'
train_data_prefix = 'images/'
val_ann_file = 'annotations/split_sam_BUF_7/val.json'
val_data_prefix = 'images/'
test_ann_file = 'annotations/split_sam_BUF_7/test.json'
test_data_prefix = 'images/'
class_name = ('certain whale', 'uncertain whale', 'harp seal')
# num_classes = len(class_name)
num_classes = 3
metainfo = dict(
    classes=class_name,
    palette=[(250, 165, 30), (255, 255, 0), (0, 255, 0)])
# Batch size of a single GPU during training
train_batch_size_per_gpu = 24   # 11940 MB
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 20
RepeatDataset_times = 10

# -----model related-----

# -----train val related-----
# base_lr_default * (your_bs / default_bs (8x16)) for SGD
base_lr = _base_.base_lr * train_batch_size_per_gpu / (8 * 16)
max_epochs = 100
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'  # noqa

# default_hooks
save_epoch_intervals = 10
logger_interval = 200    # record 1 piece of log for each epoch  
max_keep_ckpts = 5

# train_cfg
val_interval = 2
val_begin = 20

tta_model = None
tta_pipeline = None

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend')
                  ])

# ===================== Unmodified in most cases ==================
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        # prior_generator=dict(base_sizes=anchors),
        loss_cls=dict(loss_weight=0.5 *
                      (num_classes / 80 * 3 / _base_.num_det_layers))),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes)
    ))

train_dataloader = dict(
    collate_fn=dict(_delete_=True, type='yolov5_collate'),   # add https://github.com/open-mmlab/mmyolo/issues/886
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),   # add
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=RepeatDataset_times,
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file=train_ann_file,
            data_prefix=dict(img=train_data_prefix),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)))

val_dataloader = dict(
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),   # add
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)))

test_dataloader = dict(
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),   # add
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img=test_data_prefix)))

optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        # save_param_scheduler=None,  # for yolov5
        interval=save_epoch_intervals,
        max_keep_ckpts=max_keep_ckpts,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=logger_interval))

val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = dict(ann_file=data_root + test_ann_file)

train_cfg = dict(
    max_epochs=max_epochs, val_begin=val_begin, val_interval=val_interval)
