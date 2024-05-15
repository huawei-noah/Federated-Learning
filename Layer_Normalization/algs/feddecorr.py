"""
This file is borrowed from the original implementation of FedDecorr loss.
Check https://github.com/Yujun-Shi/FedCLS/
Reference:
[1] Shi, Yujun, et al. "Towards understanding and mitigating dimensional collapse
    in heterogeneous federated learning." arXiv preprint arXiv:2210.00226 (2022).
"""


import torch
import torch.nn as nn


class FedDecorrLoss(nn.Module):
    """Implementation of FedDecorr Loss
    https://arxiv.org/pdf/2210.00226.pdf

    from the original implementation : https://github.com/Yujun-Shi/FedCLS/
    """

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        n, c = x.shape
        if n == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / n

        return loss

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape

        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
