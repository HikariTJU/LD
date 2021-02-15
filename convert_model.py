import torch
from mmcv.parallel import MMDistributedDataParallel
import mmcv
from mmcv.runner import load_checkpoint, save_checkpoint
from mmdet.core import get_classes
from mmdet.models import build_detector

def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = get_classes('coco')
    model.eval()
    return model


pth = 'your_checkpoint.pth'	

model = init_detector('configs/gfl/gfl_r18_fpn_1x_coco.py ',pth)	# corresponding config file

save_checkpoint(model, pth)
