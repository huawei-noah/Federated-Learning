'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim

class CNN(nn.Module):
    def __init__(self, optimizer=optim.SGD, lr: float = 0.01, seed=None, **kwargs):
        super(CNN, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        input_shape = kwargs.get("input_shape", (None, 1, 28, 28))
        output_size = kwargs.get("output_size", 10)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[1],
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output  # return x for visualization


class ResNet(torch.nn.Module):
    def __init__(
        self,
        optimizer=optim.Adam,
        lr: float = 5e-5,
        seed=None,
        dropout=0.0,
        groupnorm=False,
        **kwargs
    ):
        super(ResNet, self).__init__()

        pretrain = kwargs.get("pretrain", True)
        input_shape = kwargs.get("input_shape", (None, 3, 224, 224))
        output_size = kwargs.get("output_size", 10)
        resnet50 = kwargs.get("resnet50", False)

        if not resnet50:
            self.network = torchvision.models.resnet18(pretrained=pretrain)
        else:
            self.network = torchvision.models.resnet50(pretrained=pretrain)

        if input_shape[1] != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                input_shape[1],
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

            for i in range(input_shape[1]):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        del self.network.fc
        self.network.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.CrossEntropyLoss()

        if seed is not None:
            torch.manual_seed(seed)

        self.out = torch.nn.Linear(512 * (1 + 3 * resnet50), output_size)
        self.network.train(True)
        if groupnorm:
            self.replace_batchnorm(self)
        else:
            self.freeze_bn()

    def forward(self, x):
        return self.out(self.dropout(self.network(x)))

    def train(self, mode=True):
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def replace_batchnorm(self, module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(
                    module,
                    name,
                    nn.GroupNorm(
                        num_groups=2, num_channels=child.num_features, eps=1e-05
                    ),
                )
            else:
                self.replace_batchnorm(child)