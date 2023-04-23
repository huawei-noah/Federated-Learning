'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import torch
from tqdm import tqdm
import numpy as np
from torchmetrics import Accuracy

def topk_acc(model, loader, device, topk=1):
    correct = 0
    total = 0
    model.to(device)
    accuracy_k = Accuracy(top_k=topk)
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            target = target.to('cpu')
            pred = model(images).to('cpu')
            correct += accuracy_k(pred, target).item() * target.numel()
            total += target.numel()
        acc = correct / total
    return acc * 100


def accuracy(model, loader, device, show=True):
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        if show:
            t = tqdm(loader)
        else:
            t = loader
        for images, target in t:
            images = images.to(device)
            target = target.to(device)
            correct += (model(images).argmax(dim=1) == target).sum().item()
            total += target.numel()
            acc = correct / total
            if show:
                t.set_description(f'test acc: {acc*100:.2f}%')
    return acc * 100


def loss(model, loader, loss_fn, device, show=True):
    loss_total = 0.
    total = 0
    model.to(device)
    
    with torch.no_grad():
        if show:
            t = tqdm(loader)
        else:
            t = loader
        for images, target in t:
            images = images.to(device)
            target = target.to(device)
            #target = torch.nn.functional.one_hot(target, num_classes=10).type(torch.cuda.FloatTensor)
            outputs = model(images).to(device)
            loss_total += loss_fn(outputs, target) * len(target)
            total += len(target)
        
        loss_avg = loss_total / total
    return loss_avg.item()


def topk_accuracies(models, loaders, device, topk=1):
    num_clients = len(loaders)
    accs = []
    for i in range(num_clients):
        model, loader = models[i], loaders[i]
        acc = topk_acc(model, loader, device, topk=topk)
        accs.append(acc)
    return np.array(accs)


def accuracies(models, loaders, device):
    num_clients = len(loaders)
    accs = []
    for i in range(num_clients):
        model, loader = models[i], loaders[i]
        acc = accuracy(model, loader, device, show=False)
        accs.append(acc)
    return np.array(accs)

def losses(models, loaders, loss_fn, device):
    num_clients = len(models)
    losses_ = []
    for i in range(num_clients):
        model, loader = models[i], loaders[i]
        loss_ = loss(model, loader, loss_fn, device, show=False)
        losses_.append(loss_)
    return np.array(losses_)