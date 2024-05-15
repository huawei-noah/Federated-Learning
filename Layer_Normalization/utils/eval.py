'''
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the BSD 3.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3.0 License for more details.
'''

import torch
from tqdm import tqdm
import numpy as np


def accuracy_loss(model, loader, loss_fn, device, show=True):
    loss_total = 0.
    correct = 0
    total = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        if show:
            t = tqdm(loader)
        else:
            t = loader
        for images, target, *offsets in t:
            images = images.to(device)
            target = target.to(device)
            if offsets:
                offsets = offsets[0].to(device)
                outputs = model(images, offsets)
            else:
                outputs = model(images)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            outputs.to(device)
            correct += (outputs.argmax(1) == target).sum().item()
            loss_total += loss_fn(outputs, target) * len(target)
            total += target.numel()
            acc = correct / total
            if show:
                t.set_description(f'test acc: {acc*100:.2f}%')
        loss_avg = loss_total / total
    return acc * 100, loss_avg.item()


def accuracies_losses(models, loaders, loss_fn, device):
    num_clients = len(loaders)
    accs, losses_ = [], []
    for i in range(num_clients):
        model, loader = models[i], loaders[i]
        acc, loss_ = accuracy_loss(model, loader, loss_fn, device, show=False)
        accs.append(acc)
        losses_.append(loss_)
    return np.array(accs), np.array(losses_)


def mean_std(accs):
    return np.mean(accs), np.std(accs)


def max_min(accs):
    return np.max(accs) - np.min(accs)


def best_p(accs, percent=0.2):
    sorted_ = np.sort(accs)
    num = len(accs)
    best = sorted_[num - int(num * percent):]
    return mean_std(best)


def worst_p(accs, percent=0.2):
    sorted_ = np.sort(accs)
    num = len(accs)
    worst = sorted_[:int(num * percent)]
    return mean_std(worst)
