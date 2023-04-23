'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

# borrowed from 
# https://github.com/facebookresearch/DomainBed/blob/main/domainbed/lib/fast_data_loader.py

# Infinite dataloader so that we can just keep iterating

import torch

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

from torch.utils.data import RandomSampler, BatchSampler, DataLoader
                
class InfiniteDataLoader():
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        sampler = RandomSampler(dataset, replacement=True)

        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        self._infinite_iterator = iter(DataLoader(dataset, num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError
 

class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = BatchSampler(
            RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):  # for _ in range(len(self)/10)
            yield next(self._iterator)

    def __len__(self):
        return self._length
