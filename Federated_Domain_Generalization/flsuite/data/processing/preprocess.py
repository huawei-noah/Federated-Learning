'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

from typing import Optional, Union

import torch
import numpy as np
from .. import utils


def train_test_split(sample_set, frac=0.2, seed: Optional[int] = None):
    total = len(sample_set)

    np.random.seed(seed)
    idx = np.random.choice(np.arange(total), total, replace=False)
    train_num = int(total * frac)

    train_set = sample_set[idx[train_num:]]
    test_set = sample_set[idx[:train_num]]

    return train_set, test_set


def non_iid_split(
    dataset,
    sizes: Union[int, list],
    beta: int = 10,
    seed: Optional[int] = None,
    labels: Optional[Union[np.array, torch.Tensor]] = None,
):
    if isinstance(sizes, int):
        sizes = [sizes] * (len(dataset) // sizes)

    sizes = np.array(sizes)
    if labels is None:
        labels = dataset[:][1]

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
        unique_labels = labels.unique().numpy()
    else:
        unique_labels = np.array(list(set(labels)))

    labels = np.array(labels).astype(int)

    n_classes = len(unique_labels)
    prior = np.ones(n_classes) / n_classes
    np.random.seed(seed=seed)
    sizes = (
        (np.random.dirichlet(prior * beta, len(sizes)) * sizes[:, np.newaxis])
        .round()
        .astype(int)
    )

    splits = list()
    class_idx = {k: np.where(labels == k)[0] for k in unique_labels}
    class_idx_pos = {k: 0 for k in unique_labels}
    for size in sizes:
        X = list()
        for i, k in enumerate(unique_labels):
            start = class_idx_pos[k]
            indexes = class_idx[k][start : start + size[i]]
            for j in indexes:
                X.append(dataset[int(j)])
            class_idx_pos[k] += size[i]
        splits.append(X)
    return splits


def iid_split(dataset, sizes: Union[int, list], seed: Optional[int] = None):
    if isinstance(sizes, int):
        sizes = [sizes] * (len(dataset) // sizes)

    start = 0
    splits = list()
    np.random.seed(seed=seed)
    idx = np.random.choice(np.arange(len(dataset)), len(dataset), replace=False)
    for size in sizes:
        X, y = dataset[idx[start : start + size]]
        splits.append(utils.CustomDataset(X, y))
        start += size
    return splits
