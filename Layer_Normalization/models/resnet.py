"""
This file is modified from the original implementation of PyTorch.
Check https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
The original file has license BSD-3.0.
"""


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, conv1x1


class BasicBlockGroupnormAfter(BasicBlock):
    """Normalization after ReLU"""

    def __init__(self, inplanes, planes, *args, stride=1, **kwargs):
        super().__init__(inplanes, planes, *args, stride, **kwargs)
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.bn2(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        norm_method,
        group_number,
        block,
        num_blocks,
        num_classes=10,
        affine=True,
        return_feature=False,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.group_number = group_number
        self.affine = affine
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_method = norm_method
        if norm_method == "batch_norm":
            self.bn1 = nn.BatchNorm2d(64, affine)
        elif "group_norm" in norm_method:
            self.gn1 = nn.GroupNorm(group_number, 64, affine)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if num_classes == 200:
            feature_dim = 4  # To adapt the model for TinyImageNet
        elif num_classes == 7:
            feature_dim = 49  # To adapt the model for PACS dataset
        else:
            feature_dim = 1  # To adapt the model for CIFAR datasets
        self.w = nn.Linear(feature_dim * 512 * 1, num_classes)
        self.return_feature = return_feature

    def forward(self, x):
        if self.norm_method == "batch_norm":
            out = F.relu(self.bn1(self.conv1(x)))
        elif self.norm_method == "group_norm":
            out = F.relu(self.gn1(self.conv1(x)))
        elif self.norm_method == "group_norm_after":
            out = self.gn1(F.relu(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.reshape(out.size(0), -1)
        if self.norm_method == "fn":
            out = F.normalize(out, p=2, dim=1) * torch.sqrt(torch.tensor(out.shape[1]))
        feature = out
        out = self.w(out)

        return out, feature

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            downsample = None
            if stride != 1 or self.in_planes != planes * 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_planes, planes,
                            kernel_size=1, stride=stride, bias=False),
                )
            if "group_norm" in self.norm_method:
                layers.append(
                    block(
                        self.in_planes, planes, groups=self.group_number, stride=stride, downsample=downsample
                    )
                )
            elif "batch_norm" in self.norm_method:
                layers.append(block(self.in_planes, planes, stride, self.affine))
            else:
                layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * 1
        return nn.Sequential(*layers)


def resnet18(num_classes, norm_method, affine, group_number=1, **kwargs):
    if norm_method == "no_norm" or norm_method == "fn":
        return ResNet(
            norm_method,
            group_number=0,
            block=partial(
                BasicBlock,
                norm_layer=nn.Sequential(),
            ),
            num_blocks=[2, 2, 2, 2],
            num_classes=num_classes,
            **kwargs
        )

    elif norm_method == "group_norm":
        return ResNet(
            norm_method,
            group_number=group_number,
            block=partial(
                BasicBlock,
                norm_layer=partial(nn.GroupNorm, group_number, affine=affine),
            ),
            num_blocks=[2, 2, 2, 2],
            num_classes=num_classes,
            affine=affine,
        )

    elif norm_method == "group_norm_after":
        return ResNet(
            norm_method,
            group_number=group_number,
            block=partial(
                BasicBlockGroupnormAfter,
                norm_layer=partial(nn.GroupNorm, group_number, affine=affine),
            ),
            num_blocks=[2, 2, 2, 2],
            num_classes=num_classes,
            affine=affine,
        )

    elif norm_method == "batch_norm":
        return ResNet(
            norm_method,
            group_number=0,
            block=partial(
                BasicBlock,
                norm_layer=nn.BatchNorm2d(affine=affine),
            ),
            num_blocks=[2, 2, 2, 2],
            num_classes=num_classes,
            affine=affine,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    pass
