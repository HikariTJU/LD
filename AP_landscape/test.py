import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config1', help='test config file path')
    parser.add_argument('config2', help='test config file path')
    parser.add_argument('checkpoint1', help='checkpoint file')
    parser.add_argument('checkpoint2', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg1 = Config.fromfile(args.config1)
    cfg2 = Config.fromfile(args.config2)
    if args.cfg_options is not None:
        cfg1.merge_from_dict(args.cfg_options)
        cfg2.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg1.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg1['custom_imports'])
        import_modules_from_strings(**cfg2['custom_imports'])
    # set cudnn_benchmark
    if cfg1.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg1.model.pretrained = None
    cfg2.model.pretrained = None
    if cfg1.model.get('neck'):
        if isinstance(cfg1.model.neck, list):
            for neck_cfg in cfg1.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg1.model.neck.get('rfp_backbone'):
            if cfg1.model.neck.rfp_backbone.get('pretrained'):
                cfg1.model.neck.rfp_backbone.pretrained = None
    if cfg2.model.get('neck'):
        if isinstance(cfg2.model.neck, list):
            for neck_cfg in cfg2.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg2.model.neck.get('rfp_backbone'):
            if cfg2.model.neck.rfp_backbone.get('pretrained'):
                cfg2.model.neck.rfp_backbone.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg1.data.test, dict):
        cfg1.data.test.test_mode = True
        samples_per_gpu = cfg1.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg1.data.test.pipeline = replace_ImageToTensor(
                cfg1.data.test.pipeline)
    elif isinstance(cfg1.data.test, list):
        for ds_cfg in cfg1.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg1.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg1.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if isinstance(cfg2.data.test, dict):
        cfg2.data.test.test_mode = True
        samples_per_gpu = cfg2.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg2.data.test.pipeline = replace_ImageToTensor(
                cfg2.data.test.pipeline)
    elif isinstance(cfg2.data.test, list):
        for ds_cfg in cfg2.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg2.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg2.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg1.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg1.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg1.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg1.model.train_cfg = None
    cfg2.model.train_cfg = None
    model1 = build_detector(cfg1.model, test_cfg=cfg1.get('test_cfg'))
    model2 = build_detector(cfg2.model, test_cfg=cfg2.get('test_cfg'))
    fp16_cfg = cfg1.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model1)
        wrap_fp16_model(model2)
    checkpoint1 = load_checkpoint(model1, args.checkpoint1, map_location='cpu')
    checkpoint2 = load_checkpoint(model2, args.checkpoint2, map_location='cpu')
    if args.fuse_conv_bn:
        model1 = fuse_conv_bn(model1)
        model2 = fuse_conv_bn(model2)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint1.get('meta', {}):
        model1.CLASSES = checkpoint1['meta']['CLASSES']
        model2.CLASSES = checkpoint1['meta']['CLASSES']
    else:
        model1.CLASSES = dataset.CLASSES
        model2.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)

    else:
        model1 = MMDistributedDataParallel(
            model1.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        model2 = MMDistributedDataParallel(
            model2.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model1, data_loader, model2, args.gpu_collect)
        #outputs = multi_gpu_test(model1, data_loader, args.tmpdir, args.gpu_collect)
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg1.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
