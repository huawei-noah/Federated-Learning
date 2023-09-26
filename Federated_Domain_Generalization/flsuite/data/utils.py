'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

from collections import namedtuple
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataset import TensorDataset

datatuple = namedtuple("datatuple", field_names=["train", "test"])


class CustomDataset(Dataset):
    """TensorDataset with support of transforms"""

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.tensors = (self.X, self.y)
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.X[index]), self.y[index]
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.size(0)

    @classmethod
    def parse_values(cls, values, transform=None):
        if isinstance(values, TensorDataset):
            return cls(values.tensors[0], values.tensors[1], transform)
        elif isinstance(values, (tuple, list)):
            return cls(values[0], values[1], transform)
        elif isinstance(values, cls):
            if transform:
                return cls(values.X, values.y, transform)
            return cls(values.X, values.y, values.transform)
        raise ValueError("Invalid input")


class DataLoaderWrapper:
    def __init__(self, dataloaders, draws=1):
        self._buffer = dataloaders
        self.loaders = [iter(x) for x in dataloaders]
        self.draws = draws

    def __iter__(self):
        full_batch = list()
        for i, x in enumerate(self.loaders):
            for _ in range(self.draws):
                full_batch.append(self._draw_batch(i))
        yield full_batch

    def _draw_batch(self, i):
        try:
            batch = next(self.loaders[i])

        except StopIteration:
            self.loaders[i] = iter(self._buffer[i])
            batch = next(self.loaders[i])

        return batch

    def __len__(self):
        return len(self.loaders)

    @classmethod
    def wrap_single_dataloader(cls, dataloader):
        wrapper = cls([dataloader])
        wrapper.dataset = dataloader.dataset
        return wrapper


def build_dataloaders(
    datasets: list,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    transform: Optional[Callable] = None,
    sampler: Optional[List[Sampler]] = None,
    seed: int = None,
    **kwargs
) -> List[DataLoader]:
    shuffle = False if sampler else shuffle
    sampler = sampler if sampler else [None] * len(datasets)
    dataloaders = list()
    for i, x in enumerate(datasets):
        if isinstance(x, (list, tuple)):
            x = CustomDataset.parse_values(x)
        generator = torch.Generator()
        if seed:
            generator.manual_seed(seed)

        dataloaders.append(
            DataLoader(
                x,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=False,
                shuffle=shuffle,
                collate_fn=transform,
                sampler=sampler[i],
                generator=generator,
                **kwargs
            )
        )
    return dataloaders


def merge_datasets(datasets):
    if isinstance(datasets[0], TensorDataset):
        datasets = [(x[:][0], x[:][1]) for x in datasets]

    X = torch.cat([x[0] for x in datasets])
    y = torch.cat([x[1] for x in datasets])
    return (X, y)
