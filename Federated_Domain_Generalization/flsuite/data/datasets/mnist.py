'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

# Code extracted from Facebook's DomainBed project
# https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py

import os
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms.functional import rotate


class MultipleDomainDataset:
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return TensorDataset(*self.datasets[index])

    def __len__(self):
        return len(self.datasets)


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(
        self, root, environments, dataset_transform, input_shape, num_classes, seed=None
    ):
        super().__init__()
        if root is None:
            raise ValueError("Data directory not specified!")

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat(
            (original_dataset_tr.data, original_dataset_te.data)
        )

        original_labels = torch.cat(
            (original_dataset_tr.targets, original_dataset_te.targets)
        )

        if seed is not None:
            torch.manual_seed(seed)

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i :: len(environments)]  # start:stop:step
            labels = original_labels[i :: len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


# RotatedMNIST can be sliced because of TensorDataset
# MNIST cannot be sliced
class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ["0", "15", "30", "45", "60", "75"]

    def __init__(self, root, seed=None):
        super(RotatedMNIST, self).__init__(
            root,
            list(map(int, RotatedMNIST.ENVIRONMENTS)),
            self.rotate_dataset,
            (
                1,
                28,
                28,
            ),
            10,
            seed,
        )

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(
                    lambda x: rotate(
                        x,
                        angle,
                        fill=(0,),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                ),
                transforms.ToTensor(),
            ]
        )

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)
