'''
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the BSD 3.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3.0 License for more details.
'''

from copy import deepcopy

import torch
import numpy as np



def model_to_params(model):
    return [param.data for param in model.parameters()]


def model_to_numpy(model):
    return [param.data.numpy() for param in model.parameters()]


def numpy_to_model(ndarray_list, model):
    for idx, param in enumerate(model.parameters()):
        param.data.copy_(torch.tensor(ndarray_list[idx]))


def norm2_model(model):

    params = model_to_params(model)
    sum_ = 0.0
    for param in params:
        sum_ += torch.norm(param) ** 2
    return sum_


def zero_model(model):
    zero = deepcopy(model)
    for _, param in enumerate(zero.parameters()):
        param.data = torch.zeros_like(param.data)
    return zero


def scale_model(model, scale):
    scaled = deepcopy(model)
    for i, param in enumerate(scaled.parameters()):
        model_param = model_to_params(model)[i]
        param.data = scale * model_param.data
    return scaled


def add_models(model1, model2, alpha=1.0):
    # obtain model1 + alpha * model2 for two models of the same size
    addition = deepcopy(model1)
    for i, param_add in enumerate(addition.parameters()):
        param1, param2 = model_to_params(model1)[i], model_to_params(model2)[i] 
        with torch.no_grad():
            param_add.data = param1.data + alpha * param2.data
    return addition


def sub_models(model1, model2):
    # obtain model1 - model2 for two models of the same size
    subtract = deepcopy(model1)
    for i, param_sub in enumerate(subtract.parameters()):
        param1, param2 = model_to_params(model1)[i], model_to_params(model2)[i] 
        with torch.no_grad():
            param_sub.data = param1.data - param2.data
    return subtract


def product_models(model1, model2):
    # obtain model1 - model2 for two models of the same size
    prod = 0.0
    for _, _ in enumerate(model1.parameters()):
        param1, param2 = model_to_params(model1)[i], model_to_params(model2)[i] 
        with torch.no_grad():
            prod += torch.dot(param1.data.view(-1), param2.data.view(-1))
    return prod


def assign_model(model1, model2):
    # state_dict() shows all the parameters while model.parameters() does not show running_avg of BN layers
    with torch.no_grad():
        for key in model2.state_dict().keys():
            model1.state_dict()[key].data.copy_(model2.state_dict()[key])
        return


def assign_models(models, new_model):
    ''' assign the new_model into a list of models'''
    for model in models:
        assign_model(model, new_model)
    return


def avg_models(models, weights=None):
    '''take a list of models and average, weights: a list of numbers summing up to 1'''
    if not weights:
        total = len(models)
        weights = [1.0 / total] * total
    avg = zero_model(models[0])
    
    # state_dict() shows all the parameters while model.parameters() does not show running_avg of BN layers
    for key in models[0].state_dict().keys():
        # num_batches_tracked is a non trainable LongTensor and
        # num_batches_tracked are the same for all clients for the given datasets
        if 'num_batches_tracked' in key:
            avg.state_dict()[key].data.copy_(models[0].state_dict()[key])
        else:
            temp = torch.zeros_like(avg.state_dict()[key])
            for client_idx, _ in enumerate(models):
                temp += weights[client_idx] * models[client_idx].state_dict()[key]
            avg.state_dict()[key].data.copy_(temp)
    return avg


def sum_models(models):
    '''take a list of models and average, weights: a list of numbers summing up to 1'''
    weights = [1.0] * len(models)
    return avg_models(models, weights=weights)


def aggregate(models, weights=None):   # FedAvg
    avg = avg_models(models, weights=weights)
    assign_models(models, avg)
    return


def aggregate_lr(old_model, models, weights=None, global_lr=1.0): # FedAvg
    '''return old_model + global_lr * Delta, where Delta is aggregation of local updates'''
    with torch.no_grad():
        delta = [sub_models(model, old_model) for model in models]
        avg_delta = avg_models(delta, weights=weights)
        new_model = add_models(old_model, avg_delta, alpha=global_lr)
        assign_models(models, new_model)
    return


def aggregate_momentum(old_model, server_momentum, models, weights=None, global_lr=1.0, \
                 momentum_coeff=0.9): # FedAvg
    '''
    return old_model + global_lr * Delta + 0.9 momentum, where Delta is aggregation of local updates
    Polyak's momentum it changes momentum and models
    '''
    with torch.no_grad():
        delta = [sub_models(old_model, model) for model in models]
        avg_delta = avg_models(delta, weights=weights)
        avg_delta = scale_model(avg_delta, global_lr)
        assign_model(server_momentum, add_models(avg_delta, server_momentum, momentum_coeff))
        new_model = sub_models(old_model, server_momentum)
        assign_models(models, new_model)
    return
