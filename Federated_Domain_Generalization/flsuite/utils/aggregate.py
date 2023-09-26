import torch
import numpy as np
from copy import deepcopy

""" basic """


def model_to_params(model):
    return [param.data for param in model.parameters()]


"""one model operation"""


def zero_model(model):
    zero = deepcopy(model)
    with torch.no_grad():
        for i, param in enumerate(zero.parameters()):
            param.data = torch.zeros_like(param.data)
    return zero


def scale_model(model, scale):
    with torch.no_grad():
        scaled = deepcopy(model)
        for i, param in enumerate(scaled.parameters()):
            model_param = model_to_params(model)[i]
            param.data = scale * model_param.data
    return scaled


"""two model operation"""


def add_models(model1, model2, alpha=1.0):
    # obtain model1 + alpha * model2 for two models of the same size
    addition = deepcopy(model1)
    layers = len(model_to_params(model1))
    for i, param_add in enumerate(addition.parameters()):
        param1, param2 = model_to_params(model1)[i], model_to_params(model2)[i]
        with torch.no_grad():
            param_add.data = param1.data + alpha * param2.data
    return addition


def sub_models(model1, model2):
    # obtain model1 - model2 for two models of the same size
    subtract = deepcopy(model1)
    layers = len(model_to_params(model1))
    for i, param_sub in enumerate(subtract.parameters()):
        param1, param2 = model_to_params(model1)[i], model_to_params(model2)[i]
        with torch.no_grad():
            param_sub.data = param1.data - param2.data
    return subtract


def assign_model(model1, model2):
    for i, param1 in enumerate(model1.parameters()):
        param2 = model_to_params(model2)[i]
        with torch.no_grad():
            param1.data = deepcopy(param2.data)
    return model1


"""model list operation"""


def avg_models(models, weights=None):
    """take a list of models and average, weights: a list of numbers summing up to 1"""
    if weights == None:
        total = len(models)
        weights = [1.0 / total] * total

    with torch.no_grad():
        avg = zero_model(models[0])
        for index, model in enumerate(models):
            for i, param in enumerate(avg.parameters()):
                model_param = model_to_params(model)[i]
                param.data += model_param * weights[index]
    return avg


def assign_models(models, new_model):
    """assign the new_model into a list of models"""
    for model in models:
        assign_model(model, new_model)
    return models


"""aggregation"""


def aggregate(models, weights=None):  # FedAvg
    avg = avg_models(models, weights=weights)
    assign_models(models, avg)
    return avg


def aggregate_lr(old_model, models, weights=None, global_lr=1.0):  # FedAvg
    """return old_model + global_lr * Delta, where Delta is aggregation of local updates"""
    with torch.no_grad():
        Delta = [sub_models(model, old_model) for model in models]
        avg_Delta = avg_models(Delta, weights=weights)
        new_model = add_models(old_model, avg_Delta, alpha=global_lr)
        assign_models(new_model, models)
    return


def aggregate_momentum(
    old_model, server_momentum, models, weights=None, global_lr=1.0, momentum_coeff=0.9
):  # FedAvg
    """return old_model + global_lr * Delta + 0.9 momentum, where Delta is aggregation of local updates
    Polyak's momentum"""
    with torch.no_grad():
        Delta = [sub_models(model, old_model) for model in models]
        avg_Delta = avg_models(Delta, weights=weights)
        avg_Delta = scale_model(avg_Delta, global_lr)
        server_momentum = add_models(avg_Delta, server_momentum, momentum_coeff)
        new_model = add_models(old_model, avg_Delta)
        assign_models(new_model, models)
    return
