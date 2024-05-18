'''
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the BSD 3.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3.0 License for more details.
'''


import sys
from typing import List

from torch import nn, tensor
import torch.nn.functional as F
import torch

sys.path.append("..")


class CNN_CIFAR(nn.Module):
    def __init__(self, norm_method, device, num_classes=10, tau=1, feature_dim=192, num_groups=1,
                 affine_group_norm=False, before_activation=False, return_feature=False, bias=False):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, bias=bias)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, bias=bias)
        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=384, bias=bias)
        self.fc2 = nn.Linear(in_features=384, out_features=feature_dim, bias=bias)
        self.w = nn.Linear(feature_dim, num_classes, bias=bias)
        self.tau = tau
        self.gn0 = nn.GroupNorm(num_groups, num_channels=64, affine=affine_group_norm).to(device)
        self.gn1 = nn.GroupNorm(num_groups, num_channels=64, affine=affine_group_norm).to(device)
        self.gn2 = nn.GroupNorm(num_groups, num_channels=384, affine=affine_group_norm).to(device)
        self.gn3 = nn.GroupNorm(num_groups, feature_dim, affine=affine_group_norm).to(device)
        self.bn0 = nn.BatchNorm2d(num_features=64, affine=affine_group_norm).to(device)
        self.bn1 = nn.BatchNorm2d(num_features=64, affine=affine_group_norm).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=384, affine=affine_group_norm).to(device)
        self.bn3 = nn.BatchNorm1d(feature_dim, affine=affine_group_norm).to(device)
        self.group_norms = {0: self.gn0, 1: self.gn1, 2: self.gn2, 3: self.gn3}
        self.batch_norms = {0: self.bn0.to(device), 1: self.bn1.to(device), 2: self.bn2.to(device), 3: self.bn3.to(device)}

        self.before_activation = before_activation
        self.gamma = nn.Parameter(data=tensor(
            feature_dim ** (1 / 2)), requires_grad=True)
        self.normalization(norm_method.replace(" ", "").strip('][').split(','))
        self.return_feature = return_feature
        
    def forward(self, x):
        if self.before_activation:
            x = self.feature_norm_before_activation(x)
        else:
            x = self.feature_norm_after_activation(x)
        feature = x
        x = self.w(x)
        
        return x / self.tau, feature

    def zero_mean(self, x):
        """zero means the input"""

        # for feedforward layers
        if len(x.shape) == 2: 
            mean = torch.mean(x, dim=1,)
            return torch.sub(x, mean[:, None])
        # for conv layers
        elif len(x.shape) == 4:
            mean = torch.mean(x, dim=(1, 2, 3), keepdim=False)
            return torch.sub(x, mean[:, None, None, None])
        else:
            raise NotImplementedError

    def l2_norm(self, x, variance_scale=False):
        """l2_normalize the input"""

        # for feedforward layers
        if len(x.shape) == 2:
            if variance_scale:
                return F.normalize(x) * torch.sqrt(tensor(x.shape[1]))
            else:
                return F.normalize(x)
        elif len(x.shape) == 4:
            if variance_scale:
                return F.normalize(x, dim=(1, 2, 3)) * torch.sqrt(tensor(x.shape[1] * x.shape[2] * x.shape[3]))
            else:
                return F.normalize(x, dim=(1, 2, 3))
        else:
            raise NotImplementedError
        
    def normalization(self, method: List[str]) -> List:
        """returns list of function. Each function performs normalization per layer"""
        function_per_layer = [lambda x :x for _ in range(len(method))]

        # Loop throuh the methods for each layer
        for layer_num, method_layer in enumerate(method):
            for letter in method_layer:
                letter = letter.lower()
                if letter == 's':
                    # shifts the vector
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num] : self.zero_mean(f(x))
                elif letter == 'n':
                    # l2 norm the vector
                    function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num] : self.l2_norm(f(x))
                elif letter == 'v':
                    # l2 norm the vector with variance scale
                    function_per_layer[layer_num] =\
                        lambda x, f=function_per_layer[layer_num] : self.l2_norm(f(x), variance_scale=True)
                elif letter == 'c':
                    # multipy the layer by the numbers after c
                    multiplier = ''.join([i for i in method_layer if i.isdigit()])
                    function_per_layer[layer_num] =\
                        lambda x, f=function_per_layer[layer_num] : f(x) * tensor(int(multiplier))
                elif letter == 'g':
                    # group normalization
                    function_per_layer[layer_num] =\
                        lambda x, f=function_per_layer[layer_num], layer_num=layer_num : \
                                self.group_norms[layer_num](f(x))
                elif letter == 'b':
                    # batch normalization
                    function_per_layer[layer_num] =\
                        lambda x, f=function_per_layer[layer_num], layer_num=layer_num : \
                                self.batch_norms[layer_num](f(x))
                elif letter == 'x':
                    # learnable parameter only for last layer 3
                    if layer_num == 3:
                        function_per_layer[layer_num] = lambda x, f=function_per_layer[layer_num] : self.gamma(f(x))
                    else: 
                        raise NotImplementedError
                elif letter.isdigit():
                    # pass if there is a number in the letters
                    continue
                else:
                    raise NotImplementedError
        self.function_per_layer = function_per_layer
        return function_per_layer

    def feature_norm_before_activation(self, x):
        """Normalize vectors before activation"""
        x = self.pool(F.relu(self.function_per_layer[0](self.conv1(x))))
        x = self.pool(F.relu(self.function_per_layer[1](self.conv2(x))))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.function_per_layer[2](self.fc1(x)))
        x = F.relu(self.function_per_layer[3](self.fc2(x)))
        return x

    def feature_norm_after_activation(self, x):
        """Normalize vectors after activation"""
        x = self.pool(self.function_per_layer[0](F.relu(self.conv1(x))))
        x = self.pool(self.function_per_layer[1](F.relu(self.conv2(x))))
        x = x.view(-1, 64 * 5 * 5)
        x = self.function_per_layer[2](F.relu(self.fc1(x)))
        x = self.function_per_layer[3](F.relu(self.fc2(x)))
        return x


class CNN_BN(nn.Module):
    def __init__(self, num_classes=10, tau=1, feature_dim=192, normalize_feature=False, normalize_w=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_features=64, affine=False)
        self.bn2 = nn.BatchNorm2d(num_features=64, affine=False)
        self.bn3 = nn.BatchNorm1d(num_features=384, affine=False)
        self.bn4 = nn.BatchNorm1d(feature_dim, affine=False)   # adding bn for the last layer has some problems
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=384)
        self.fc2 = nn.Linear(in_features=384, out_features=feature_dim)
        self.w = nn.Linear(feature_dim, num_classes)
        self.normalize_feature = normalize_feature
        self.tau = tau

    def forward(self, x):
        x = self.feature(x)
        x = self.w(x)
        return x / self.tau
    
    def feature(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = F.relu(self.fc2(x))
        x = self.bn4(x)

        return x
