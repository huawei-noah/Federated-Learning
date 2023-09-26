'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import numpy as np
import torch
from typing import Optional
from flsuite.data.utils import CustomDataset

def sample_class_prob(n_classes: int, n_high: int = 0, exclude: Optional[list] = None, scale: int = 50):

    if exclude is None:
        exclude = list()
    elif isinstance(exclude, int):
        exclude = [exclude]

    prob = np.zeros(n_classes)
    n_valid = n_classes-len(exclude)
    
    beta = np.ones(n_valid)
    beta[:n_valid-n_high] = 0.1
    np.random.shuffle(beta)
    
    dirichlet = np.random.dirichlet(beta*scale)
    dirichlet_idx = list(set(range(n_classes))-set(exclude))
    prob[dirichlet_idx] = dirichlet

    return prob

def subsample_dataset(dataset, class_prob, size):
    y_train = dataset.tensors[1].numpy()
    y_prob = class_prob[y_train]
    y_prob /= y_prob.sum()
    idx = np.random.choice(np.arange(len(y_train)), size=size, replace=False, p=y_prob).tolist()
    return CustomDataset.parse_values(dataset[idx])

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)