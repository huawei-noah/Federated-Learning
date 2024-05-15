'''
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the BSD 3.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3.0 License for more details.
'''


import copy
import sys

import torch
import numpy as np
from torch import optim, nn

sys.path.append("..")
from utils.aggregate import aggregate, aggregate_momentum, zero_model, \
                            assign_models, assign_model, avg_models, add_models, \
                            model_to_params, sub_models, model_to_numpy, numpy_to_model
from algs.client_train import client_train_s, client_train_prox, client_train_scaffold, client_train_fedlc, \
                                 client_train_feddecorr, client_train_fedrs
from algs.feddecorr import FedDecorrLoss


class FedAvg():  # abstract class for FL algorithms
    '''FedAvg: https://arxiv.org/abs/1602.05629v2

    subclass should implement the following
        --aggregate(): the server aggregation of models
        --local_updates(): the update of each client
    '''

    def __init__(self, models, optimizers, num_clients, num_local_steps, loss_func, tau=1.0):
        super(FedAvg, self).__init__()
        self.num_clients = num_clients
        self.num_local_steps = num_local_steps
        self.models = models
        self.optimizers = optimizers
        self.loss_func = loss_func
        self.losses = None
        self.tau = tau
        assign_models(models, models[0])

    def aggregate(self, weights=None):
        aggregate(self.models, weights=weights)

    def local_updates(self, train_loaders, device, normalize_w=False, clip_w=None, **kwargs):

        for i in range(self.num_clients):
            client_train_s(
            train_loader=train_loaders[i],
            loss_func=self.loss_func,
            optimizer=self.optimizers[i],
            model=self.models[i],
            device=device,
            steps=self.num_local_steps,
            normalize_w=normalize_w,
            clip_w=clip_w,
            **kwargs)
        return


class FedAwS(FedAvg):
    '''FedAwS: https://arxiv.org/abs/2004.10342'''
    def __init__(self, models, optimizers, num_clients, num_local_steps, loss_func, labels, sp=1, sp_lr=0.01,):
        super(FedAwS, self).__init__(models, optimizers, num_clients, num_local_steps, loss_func)
        self.sp = sp
        self.sp_lr = sp_lr
        self.labels = labels

    @staticmethod
    def neg_loss(w_1, w_2):
        # spreadout loss
        return torch.nn.ReLU()(torch.dot(w_1, w_2)) ** 2

    def reg_sp(self, weights):
        # from FedAwS    
        num_classes = weights.shape[0]
        sum_ = torch.tensor(0.)
        for i in range(num_classes):
            for j in range(num_classes):
                if j is not i:
                    sum_ += self.neg_loss(weights[i], weights[j]).to('cpu')
        return sum_

    def aggregate(self, weights):
        labels = self.labels
        avg = avg_models(self.models, weights=weights)
        optimizer = optim.SGD(avg.parameters(), lr=self.sp_lr, momentum=0.0)
        assign_models(self.models, avg)


class FedAvgM(FedAvg):  # abstract class for FL algorithms
    '''FedAvgM: https://arxiv.org/pdf/1909.06335.pdf'''
    def __init__(self, models, optimizers, \
                 num_clients, num_local_steps, \
                loss_func, momentum=0.9):
        super(FedAvgM, self).__init__(models, optimizers, num_clients, \
                                      num_local_steps, loss_func)
        self.old_model = models[0]  # place-holder, it doesn't work
        assign_model(self.old_model, models[0])
        self.server_mom = zero_model(models[0])
        self.momentum = momentum
        
    def aggregate(self, weights=None):
        aggregate_momentum(self.old_model, self.server_mom, \
                           self.models, weights=weights, global_lr=1.0, \
                 momentum_coeff=self.momentum)
        
    def local_updates(self, train_loaders, device, normalize_w=False, clip_w=None, **kwargs):
        assign_model(self.old_model, self.models[0])  # old_model = models[0]
        return super(FedAvgM, self).local_updates(train_loaders, device, normalize_w=normalize_w, clip_w=clip_w)
    
        
class FedProx(FedAvg):
    def __init__(self, models, optimizers, num_clients, num_local_steps, loss_func, mu=1.0):
        super(FedProx, self).__init__(models, optimizers, num_clients, num_local_steps, loss_func)
        self.mu = mu
        self.old_model = copy.deepcopy(models[0])
        
    def local_updates(self, train_loaders, device, normalize_w=False, *args, **kwargs):
        self.old_model.to(device)
        for i in range(self.num_clients):
            client_train_prox(train_loaders[i], self.loss_func, self.optimizers[i], self.models[i],
                self.old_model, device=device, steps=self.num_local_steps, \
                mu=self.mu, normalize_w=normalize_w,)  
        return

    def aggregate(self, weights=None):
        aggregate(self.models, weights=weights)
        assign_model(self.old_model, self.models[0])
                
                
class SCAFFOLD(FedAvg):
    def __init__(self, models, controls, server_control, \
                 optimizers, num_clients, num_local_steps, loss_func, lr):
        super(SCAFFOLD, self).__init__(models, optimizers, num_clients, \
                                      num_local_steps, loss_func)
        self.lr = lr   # local learning rate
        self.old_model = copy.deepcopy(models[0])
        self.delta_cs = models = [copy.deepcopy(models[0]) for _ in range(num_clients)]
        self.server_control = server_control
        self.controls = controls
        
    def aggregate(self, weights=None):
        aggregate(self.models, weights=weights)
        assign_model(self.old_model, self.models[0])  # old_model = models[0]
        self.server_control = add_models(self.server_control, \
                                         avg_models(self.delta_cs, weights=weights))

    def local_updates(self, train_loaders, device, *args, **kwargs):
        loss_func = self.loss_func
        for i in range(self.num_clients):
            self.delta_cs[i] = client_train_scaffold(
                train_loaders[i], loss_func, self.optimizers[i],
                self.models[i], self.old_model, self.controls[i], self.server_control,
                device=device, steps=self.num_local_steps, lr=self.lr)

        return


class FedBN(FedAvg):
    # average all the model parameters except BN layers

    def aggregate(self, weights=None):
        avg = avg_models(self.models, weights=weights)
        for model in self.models:
            for i, (name, param1) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    param2 = model_to_params(avg)[i]
                    param1.data = copy.deepcopy(param2.data)


class FedLC(FedAvg):
    '''FedLC: 
    https://arxiv.org/pdf/2209.00189.pdf
    
    '''
    def __init__(self, models, optimizers, num_clients, num_local_steps, loss_func, label_margin):
        super(FedLC, self).__init__(models, optimizers,
                                    num_clients, num_local_steps, loss_func)
        self.label_margin = label_margin

    def local_updates(self, train_loaders, device, **args):
        for i in range(self.num_clients):
            client_train_fedlc(train_loaders[i], self.loss_func, self.optimizers[i], self.models[i],
                               device=device, steps=self.num_local_steps,
                               label_margin=self.label_margin[i])
        return


class FedRS(FedAvg):
    '''FedRS: 
    https://dl.acm.org/doi/pdf/10.1145/3447548.3467254
    
    '''
    def __init__(self, models, optimizers, num_clients, num_local_steps, loss_func, class_access):
        super(FedRs, self).__init__(models, optimizers,
                                    num_clients, num_local_steps, loss_func)
        self.class_access = class_access

    def local_updates(self, train_loaders, device, **args):
        for i in range(self.num_clients):
            client_train_fedrs(train_loaders[i], self.loss_func, self.optimizers[i], self.models[i],
                               device=device, steps=self.num_local_steps,
                               class_access=self.class_access[i])
        return


class FedYogi(FedAvg):
    ''' Implementation of FedYogi from the FedOpt methods
        https://arxiv.org/pdf/2003.00295.pdf
    '''
    def __init__(self, models, optimizers, num_clients, num_local_steps, loss_func, device):
        super(FedYogi, self).__init__(models, optimizers, num_clients, 
                                      num_local_steps, loss_func)
        self.device = device
        self.server_model = copy.deepcopy(models[0])
        self.eta = 1e-2  # server learning rate
        self.beta_1 = 0.9
        self.beta_2 = 0.99
        self.tau = 1e-3
        self.server_model.to('cpu')
        model_np = model_to_numpy(self.server_model)
        self.v = [1e-6 * np.ones_like(x) for x in model_np]
        self.m = [np.zeros_like(x) for x in model_np]
        self.server_model.to(device)

    def aggregate(self, weights=None):
        delta = [sub_models(client_model, self.server_model) for client_model in self.models]
        aggregate(delta, weights=weights)
        avg_delta = model_to_numpy(delta[0].to('cpu'))
        self.m = [
            np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
            for x, y in zip(self.m, avg_delta)
        ]
        self.v = [
            x - (1.0 - self.beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
            for x, y in zip(self.v, avg_delta)
        ]
        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(model_to_numpy(self.server_model.to('cpu')), self.m, self.v)
        ]
        numpy_to_model(new_weights, self.server_model)
        self.server_model.to(self.device)
        [assign_model(model, self.server_model) for model in self.models]



class FedDecorr(FedAvg):
    ''' Implementation of FedDecorr
        https://arxiv.org/pdf/2210.00226.pdf
    '''
    def __init__(self, models, optimizers, num_clients, num_local_steps, loss_func, device):
        super(FedDecorr, self).__init__(models, optimizers, num_clients, 
                                        num_local_steps, loss_func)
        self.feddecorr_coef = 0.1
        self.eps = 1e-8
        self.feddecorr = FedDecorrLoss()

    def local_updates(self, train_loaders, device, **args):
        for i in range(self.num_clients):
            client_train_feddecorr(self.feddecorr, train_loaders[i], self.loss_func, self.optimizers[i], self.models[i],
                                device=device, steps=self.num_local_steps,
                                feddecorr_coef=self.feddecorr_coef)
        return
        
        

