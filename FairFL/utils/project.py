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
import quadprog
from utils.aggregate import product_models
from qpsolvers import solve_qp

def project(y):
    ''' algorithm comes from:
    https://arxiv.org/pdf/1309.1541.pdf
    '''
    u = np.array(sorted(y, reverse=True))
    x = []
    rho = 0.
    sum_ = 0
    for i in range(len(y)):
        sum_ += u[i]
        if (u[i] + (1.0/(i + 1)) * (1 - sum_)) > 0:
            rho = i + 1
    lambda_ = (1.0/rho) * (1 - np.sum(u[:rho]))
    for i in range(len(y)):
        x.append(max(y[i] + lambda_, 0))
    return x

def project_stable(y):  # solve 1/2 x^T P x + q^T x, s.t., G x <= h and A x == b
    n = len(y)
    P = np.eye(n) 
    q = -np.array(y)
    G = -np.eye(n)
    h = np.zeros(n)
    A = np.ones(n)
    b = np.array([1.0])
    return list(solve_qp(P, q, G, h, A, b))   


def solve_centered_w(U, epsilon):
    """
        utils from FedMGDA repo
        U is a list of normalized gradients (stored as state_dict()) from n users
    """
    n = len(U)
    K = np.eye(n,dtype=float)
    for i in range(n):
        for j in range(n):
            K[i,j] = product_models(U[i], U[j])

    Q = 0.5 *(K + K.T)
    p = np.zeros(n,dtype=float)
    a = np.ones(n,dtype=float).reshape(-1,1)
    Id = np.eye(n,dtype=float)
    neg_Id = -1. * np.eye(n,dtype=float)
    lower_b = (1./n - epsilon) * np.ones(n,dtype=float)
    upper_b = (-1./n - epsilon) * np.ones(n,dtype=float)
    A = np.concatenate((a,Id,Id,neg_Id),axis=1)
    b = np.zeros(n+1)
    b[0] = 1.
    b_concat = np.concatenate((b,lower_b,upper_b))
    alpha = quadprog.solve_qp(Q,p,A,b_concat,meq=1)[0]
    #print('weights of FedMGDA: ', alpha)
    return alpha
