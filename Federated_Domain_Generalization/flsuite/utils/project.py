import numpy as np


def project(y):
    """algorithm comes from:
    https://arxiv.org/pdf/1309.1541.pdf
    """
    u = sorted(y, reverse=True)
    x = []
    rho = 0
    for i in range(len(y)):
        if (u[i] + (1.0 / (i + 1)) * (1 - np.sum(np.asarray(u)[:i]))) > 0:
            rho = i + 1
    lambda_ = (1.0 / rho) * (1 - np.sum(np.asarray(u)[:rho]))
    for i in range(len(y)):
        x.append(max(y[i] + lambda_, 0))
    return x
