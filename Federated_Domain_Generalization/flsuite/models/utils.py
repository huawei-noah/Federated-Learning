'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

from typing import Optional
from . import models


def model_loader(dataset: str, num_clients: int, seed: Optional[int] = None, **kwargs):
    if dataset == "RotatedMNIST" or dataset == "RMNIST":
        return [models.CNN(seed=seed, **kwargs) for _ in range(num_clients)]
    elif dataset == "PACS":
        return [models.ResNet(seed=seed, **kwargs) for _ in range(num_clients)]
    elif dataset == "OfficeHome":
        return [
            models.ResNet(seed=seed, output_size=65, **kwargs)
            for _ in range(num_clients)
        ]
    else:
        raise LookupError("Dataset not found")
