'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import torch.optim as optim

from .afl import afl
from .fed_avg import fed_avg
from . import trainers


def fed_irm(models, *args, **kwargs):
    models = [
        trainers.IRM.bind_to(x, optimizer=optim.SGD(x.parameters(), lr=0.001))
        for x in models
    ]
    return fed_avg(models, *args, **kwargs)
