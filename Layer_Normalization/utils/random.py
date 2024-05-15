import random

import torch
import numpy as np


def fix_randomness(seed):
    # fix randomness
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
