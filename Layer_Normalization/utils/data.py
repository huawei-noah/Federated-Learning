'''
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the BSD 3.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3.0 License for more details.
'''


import itertools
import pickle
import logging
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch


def label_stat(dataset, num_classes=10):      # label distribution for the data
    counts = np.zeros(num_classes, dtype=int)
    for _, data in enumerate(dataset):
        counts[data[1]] += 1
    return counts


def label_stats(datasets, num_clients=10):
    return [label_stat(datasets[i]) for i in range(num_clients)]


def get_labels(in_file, num_clients):
    with open(in_file, 'rb') as f_in:
        in_data = pickle.load(f_in)
    labels = [in_data[i][0][1] for i in range(num_clients)]
    return labels


class TransformDataset(Dataset):
    """Dataset"""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def get_data_loaders(in_file, out_file, batch_size, num_workers,\
        num_clients, transform_train, transform_test, num_classes):

    with open(in_file, 'rb') as f_in:
        in_data = pickle.load(f_in)
    with open(out_file, 'rb') as f_out:
        out_data = pickle.load(f_out)

    if isinstance(in_data, dict):
        in_data = [x for x in in_data.values()]
        out_data = [x for x in out_data.values()]

    client_class = [x[0][1] for x in in_data]

    label_dist = {}
    for user in range(num_clients):
        dist = [0 for _ in range(num_classes)]
        for label in in_data[user]:
            dist[label[1]] += 1
        label_dist[user] = torch.tensor(dist)

    counts_train = np.array([len(in_data[i]) for i in range(num_clients)])
    counts_test = np.array([len(out_data[i]) for i in range(num_clients)])
    logging.info(f'| total train samples:  %s', np.sum(counts_train))
    logging.info(f'| total test samples:   %s', np.sum(counts_test))
    logging.info(f'| total samples:        %s', np.sum(counts_train) + np.sum(counts_test))
    logging.info(f'| client train samples: %s', counts_train)

    weights_train = list(counts_train / np.sum(counts_train))

    train_loaders = [
        DataLoader(
            dataset=TransformDataset(in_data[i], transform=transform_train),
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        ) for i in range(num_clients)
    ]

    test_loaders = [
        DataLoader(
            dataset=TransformDataset(out_data[i], transform=transform_test),
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=False,
        ) for i in range(num_clients)
    ]

    global_test_loaders = DataLoader(
        dataset=TransformDataset(ConcatDataset(out_data),
                                    transform=transform_test),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
    ) 

    return train_loaders, test_loaders, global_test_loaders, weights_train, client_class, label_dist

    

