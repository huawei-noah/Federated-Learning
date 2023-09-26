'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

from .afl import afl
from .fed_avg import fed_avg
from .individual_train import individual_train
from .fed_prox import fed_prox
from .var_min import var_min
from .fed_irm import fed_irm
from . import trainers
