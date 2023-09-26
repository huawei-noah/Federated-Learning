'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import logging
import sys

logging.basicConfig(
    format="%(levelname)s: %(message)s", level=logging.ERROR, stream=sys.stdout
)

from . import algs
from . import data
from . import models
from . import utils
