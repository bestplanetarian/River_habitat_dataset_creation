import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models.xception import get_xception
from .deeplabv3 import _ASPP
from .fcn import _FCNHead
from ..nn import _ConvBNReLU

__all__ = ['DeepLabV3Plus', 'get_deeplabv3_plus', 'get_deeplabv3_plus_xception_voc']

class DeepLabV3Plus(nn.Module):
    r"""DeepLabV3Plus
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'xception').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """

    def __init__(self, nclass, backbone='xception', aux=True, pretrained_base=True, dilated=True, **kwargs):
        super(DeepLabV3Plus, self).__init__()
        self.aux = aux
        self.nclass = nclass
        output_stride = 8 if dilated else 32

        #self.pretrained = get_xception(pretrained=pretrained_base, output_stride=output_stride, **kwargs)
        self.pretrained = resnext50_32x4d(pretrained=pretrained_base, dilated=(output_stride == 8), **kwargs)


        # deeplabv3 plus
        self.head = _DeepLabHead(nclass, c1_channels=256, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)
        
        

    def base_forward(self, x):
        # Entry flow
        # stem
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)

    # stages
        c1 = self.pretrained.layer1(x)   # low-level (256 ch)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)  # mid/aux (1024 ch)
        c4 = self.pretrained.layer4(c3)  # ASPP in (2048 ch)

    return c1, c3, c4

    def forward(self, x):
        size = x.size()[2:]
        c1, c3, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4, c1)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)

