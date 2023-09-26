'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import os
from copy import deepcopy
from typing import Union, Optional, List

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

augmentation = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

weak_augmentation = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PACS:
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, transform=transform):
        self.dir = os.path.join(root, "PACS/")
        environments = sorted([f.name for f in os.scandir(self.dir) if f.is_dir()])

        self.datasets = []
        for i, environment in enumerate(environments):
            path = os.path.join(self.dir, environment)
            self.datasets.append(customImageFolder(path, transform=transform))

        self.input_shape = (
            3,
            224,
            224,
        )
        self.num_classes = len(self.datasets[-1].classes)

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class customImageFolder(ImageFolder):
    def __getitem__(self, idx):
        X, y = list(), list()
        if isinstance(idx, slice):
            for i in range(super().__len__())[idx]:
                x_, y_ = super().__getitem__(i)
                X.append(x_), y.append(y_)

            return torch.stack(X), torch.tensor(y)
        elif isinstance(idx, int):
            return super().__getitem__(idx)

        for i in idx:
            x_, y_ = super().__getitem__(i)
            X.append(x_), y.append(y_)

        return torch.stack(X), torch.tensor(y)

    def filter_by_id(self, idx):
        copy = deepcopy(self)
        copy.imgs = [x for i, x in enumerate(copy.imgs) if i in idx]
        copy.samples = [x for i, x in enumerate(copy.samples) if i in idx]
        copy.targets = [x for i, x in enumerate(copy.targets) if i in idx]
        return copy

    def set_transform(self, transform):
        self.transform = transform
