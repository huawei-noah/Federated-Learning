'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import sys
sys.path.append("..")

import torch
import numpy as np
from utils.aggregate import aggregate, aggregate_lr, sum_models, zero_model, \
                            assign_models, avg_models, add_models, scale_model, sub_models, norm2_model
from copy import deepcopy
from utils.eval import losses
from utils.project import project, project_stable, solve_centered_w
from utils.io import round_list
import logging

def individual_train(train_loader, loss_func, optimizer, model, device, epochs): 
    
    model.to(device)
    for epoch in range(epochs):
        for images, target in train_loader:
            images = images.to(device)
            target = target.to(device)
            outputs = model(images).to(device)
            model.zero_grad()
            loss = loss_func(outputs, target).to(device)
            loss.backward()
            optimizer.step() 
    return model

ALGS = ['FedAvg', 'AFL', 'PropFair', 'qFedAvg', 'GiFair', 'FedMGDA', 'TERM']

class FedAvg():  # abstract class for FL algorithms
    '''FedAvg: https://arxiv.org/abs/1602.05629v2'''
    """subclass should implement the following
    --aggregate(): the server aggregation of models
    --local_updates(): the update of each client"""
    def __init__(self, models, optimizers, num_clients, num_local_epochs, loss_func):
        super(FedAvg, self).__init__()
        self.num_clients = num_clients
        self.num_local_epochs = num_local_epochs
        self.models = models
        self.optimizers = optimizers
        self.loss_func = loss_func
        self.losses = None
    
    def aggregate(self, weights=None):
        aggregate(self.models, weights=weights)
    
    def local_updates(self, train_loaders, device):
        loss_func = self.loss_func
        #self.losses = losses(self.models, train_loaders, self.loss_func, device)
        for i in range(self.num_clients):
            individual_train(train_loaders[i], loss_func, self.optimizers[i], self.models[i], \
                         device=device, epochs=self.num_local_epochs)    
            
            
            
class AFL(FedAvg):
    '''AFL: https://arxiv.org/abs/1902.00146'''
    def __init__(self, models, optimizers, num_clients, num_local_epochs, \
                 loss_func, lambda_, step_size_lambda=0.1):
        super(AFL, self).__init__(models, optimizers, num_clients, num_local_epochs, loss_func)
        self.lambda_ = lambda_
        self.step_size_lambda = step_size_lambda
    
    def aggregate(self, weights):
        # update lamdba   
        y = np.array(self.lambda_) + self.step_size_lambda * np.array(self.losses)
        self.lambda_ = project(y)
        super(AFL, self).aggregate(weights=self.lambda_)
        
    def local_updates(self, train_loaders, device):
        loss_func = self.loss_func
        self.losses = losses(self.models, train_loaders, self.loss_func, device)
        for i in range(self.num_clients):
            individual_train(train_loaders[i], loss_func, self.optimizers[i], self.models[i], \
                         device=device, epochs=self.num_local_epochs)
        
        
        
class PropFair(FedAvg):
    '''PropFair: https://arxiv.org/abs/2202.01666'''
    def __init__(self, models, optimizers, num_clients, num_local_epochs, \
             loss_func, base, epsilon = 0.2, huber=False):
        super(PropFair, self).__init__(models, optimizers, \
                                       num_clients, num_local_epochs, loss_func)
        self.base = base
        self.epsilon = epsilon
        self.huber = huber
        
    def local_updates(self, train_loaders, device):
        
        def log_loss(output, target, base=self.base):
            ce_loss = self.loss_func(output, target)
            base = torch.tensor(base).to(device)
            if base - ce_loss < self.epsilon:           
                # for the bad performing batches, we enforce a constant to avoid divergence
                if not self.huber:
                    return ce_loss/base
                else:
                    return ce_loss/self.epsilon
            else:
                return -torch.log(1 - ce_loss/base)
        
        for i in range(self.num_clients):
            individual_train(train_loaders[i], log_loss, self.optimizers[i], self.models[i], \
                         device=device, epochs=self.num_local_epochs)
            

class qFedAvg(FedAvg):
    '''qFedAvg: https://arxiv.org/abs/1905.10497'''
    def __init__(self, models, optimizers, num_clients, num_local_epochs, \
             loss_func, Lipschitz, q=1.0):
        super(qFedAvg, self).__init__(models, optimizers, num_clients, \
                                      num_local_epochs, loss_func)
        self.q = q
        self.Lipschitz = Lipschitz
        self.old_models = deepcopy(self.models[0])
        self.losses = None
        
    def local_updates(self, train_loaders, device):
        self.old_model = deepcopy(self.models[0]).to(device)
        self.losses = losses(self.models, train_loaders, self.loss_func, device)
        super(qFedAvg, self).local_updates(train_loaders, device)

    
    def aggregate(self, weights):
        delta_w = [scale_model(sub_models(self.old_model, model), self.Lipschitz)\
                   for model in self.models]
        Delta = [scale_model(delta_w[i], (self.losses[i] ** self.q)) for i in range(len(delta_w))]
        h = [self.q * (self.losses[i] ** (self.q - 1)) * norm2_model(delta_w[i]) + \
               self.Lipschitz * (self.losses[i] ** self.q) for i in range(len(delta_w))]
        new_model = sub_models(self.old_model, scale_model(sum_models(Delta), 1.0 / sum(h)))
        assign_models(self.models, new_model)

        
class FedMGDA(FedAvg):
    '''FedMGDA+: https://ieeexplore.ieee.org/document/9762229'''
    def __init__(self, models, optimizers, num_clients, num_local_epochs, \
             loss_func, epsilon=1.0, global_lr=1.0):
        super(FedMGDA, self).__init__(models, optimizers, num_clients, num_local_epochs, loss_func)
        self.old_models = deepcopy(self.models[0])
        self.losses = None
        self.epsilon = epsilon
        self.global_lr = global_lr
        
    def local_updates(self, train_loaders, device):
        self.old_model = deepcopy(self.models[0]).to(device)
        super(FedMGDA, self).local_updates(train_loaders, device)
    
    def aggregate(self, weights):
        delta_w = [sub_models(self.old_model, model) for model in self.models]    # compute pseudo-gradient
        norms = [torch.sqrt(norm2_model(delta)) for delta in delta_w]     # compute pseudo-grad norms
        # normalized pseudo-gradients
        delta_w = [scale_model(delta_w[i], 1 / norms[i]) for i in range(self.num_clients)]
        lambda_star = list(solve_centered_w(delta_w, epsilon=self.epsilon))
        avg_delta = avg_models(delta_w, weights=lambda_star)   # d_t as in FedMGDA+
        new_model = add_models(self.old_model, avg_delta, alpha=-self.global_lr)
        assign_models(self.models, new_model)
        
class GiFair(FedAvg):
    '''GiFair: https://arxiv.org/pdf/2108.02741.pdf'''
    def __init__(self, models, optimizers, num_clients, num_local_epochs, \
             loss_func, weights, coeff=0.1):  # weights_new = weights + coeff * self.lambda_max * r
        super(GiFair, self).__init__(models, optimizers, num_clients, \
                                      num_local_epochs, loss_func)
        weights = np.array(weights)
        self.lambda_max = min(weights/(num_clients - 1))
        self.r = np.linspace(-(num_clients - 1), num_clients - 1, num=num_clients)
        self.weights_gi = np.zeros(num_clients)
        self.coeff = coeff
         
    def aggregate(self, weights):
        self.weights_gi[np.argsort(self.losses)] = self.r
        new_weights = [weights[i] + self.coeff * self.lambda_max * self.weights_gi[i] \
                       for i in range(self.num_clients)]
        super(GiFair, self).aggregate(weights=new_weights)
        
    def local_updates(self, train_loaders, device):
        loss_func = self.loss_func
        self.losses = losses(self.models, train_loaders, self.loss_func, device)
        for i in range(self.num_clients):
            individual_train(train_loaders[i], loss_func, self.optimizers[i], self.models[i], \
                         device=device, epochs=self.num_local_epochs)        
        
class TERM(FedAvg):
    '''TERM: https://openreview.net/forum?id=K5YasWXZT3O'''
    def __init__(self, models, optimizers, num_clients, num_local_epochs, \
             loss_func, alpha=0.01):
        super(TERM, self).__init__(models, optimizers, num_clients, \
                                      num_local_epochs, loss_func)
        self.alpha = alpha 
         
    def aggregate(self, weights):
        weights_term = [np.exp(self.alpha * self.losses[i]) * weights[i] \
                        for i in range(self.num_clients)]
        weights_term = list(weights_term / np.sum(weights_term))
        super(TERM, self).aggregate(weights=weights_term)
        
        
    def local_updates(self, train_loaders, device):
        loss_func = self.loss_func
        self.losses = losses(self.models, train_loaders, self.loss_func, device)
        for i in range(self.num_clients):
            individual_train(train_loaders[i], loss_func, self.optimizers[i], self.models[i], \
                         device=device, epochs=self.num_local_epochs)
