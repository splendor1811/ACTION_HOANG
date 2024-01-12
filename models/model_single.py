import functools
import torch.nn as nn
from abc import ABCMeta
from .backbone.cnns.resnet3d import ResNet3d
from .head_block.simple_head import I3DHead


def rgeattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))

class Recognizer3D(nn.Module, metaclass=ABCMeta):
    """Base class for recognizers.

        All recognizers should subclass it.
        All subclass should overwrite:

        - Methods:``forward_train``, supporting to forward when training.
        - Methods:``forward_test``, supporting to forward when testing.

        Args:
            backbone (dict): Backbone modules to extract feature.
            cls_head (dict | None): Classification head to process feature. Default: None.
            train_cfg (dict): Config for training. Default: {}.
            test_cfg (dict): Config for testing. Default: {}.
        """

    def __init__(self,
                 backbone,
                 cls_head=None,
                 train_cfg=dict(),
                 test_cfg=dict()):
        super().__init__()
        self.backbone = ResNet3d(conv1_kernel=(1, 7, 7),
                                 inflate=tuple(backbone['inflate']),
                                 in_channels=backbone['in_channels'],
                                 base_channels=backbone['base_channels'],
                                 num_stages=backbone['num_stages'],
                                 out_indices=tuple(backbone['out_indices']),
                                 stage_blocks=tuple(backbone['stage_blocks']),
                                 conv1_stride=tuple(backbone['conv1_stride']),
                                 pool1_stride=tuple(backbone['pool1_stride']),
                                 spatial_strides=tuple(backbone['spatial_strides']),
                                 temporal_strides=tuple(backbone['temporal_strides'])
                                 )  # conv1_kernel=(1, 7, 7), inflate=(0, 0, 1, 1))
        self.cls_head = I3DHead(cls_head['num_classes'], cls_head['in_channels'], cls_head['dropout'])

    def forward(self, img):
        # print("Image shape: ",img.shape)
        out = self.backbone(img)
        # print("Output Backbone: ",out.shape)
        out = self.cls_head(out)

        return out