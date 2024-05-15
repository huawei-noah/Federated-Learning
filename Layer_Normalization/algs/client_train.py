'''
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the BSD 3.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3.0 License for more details.
'''

import os
import copy
from copy import deepcopy

from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F

from utils.eval import accuracy_loss
from utils.aggregate import sub_models, assign_model, norm2_model
from utils.io import to_csv



def clip(w, c):
    with torch.no_grad():
        scale = torch.minimum(c / w.norm(dim=1), torch.ones(1).to(w.device))
        w.mul_(scale[None].T)


def normalize(w):
    with torch.no_grad():
        w = nn.Parameter(F.normalize(w, p=2, dim=1))


def client_train_s(train_loader, loss_func, optimizer, model, device, steps=1, normalize_w=None, clip_w=None, **kwargs):
    '''the train_loader has to be Infinite_Dataloader'''
    ''' norm means whether to normalize the last layer'''
    model.train()
    model = model.to(device)
    step = 0
    while step < steps:
        for images, target, in train_loader:
            model.zero_grad()
            optimizer.zero_grad()
            images = images.to(device)
            if 'batch_transform' in kwargs:
                images = kwargs['batch_transform'](images)
            target = target.to(device)
            outputs = model(images) [0]
            loss_ = loss_func(outputs, target)
            loss_.backward()
            optimizer.step()
            step += 1

            if normalize_w:
                normalize(model.w.weight)

            if clip_w is not None:
                clip(model.w.weight, clip_w)

            if step == steps:
                break
    return


def client_train_prox(train_loader, loss_func, optimizer, model, old_model, device, \
                    steps, mu=1.0, normalize_w=False):
    '''the train_loader has to be Infinite_Dataloader'''
    ''' normalize_w means whether to normalize the last layer'''
    step = 0
    model.train()
    model.to(device)
    old_model.to(device)
    for param in old_model.parameters():
        param.requires_grad = False
    while step < steps:
        for images, target in train_loader:
            step += 1
            images, target = images.to(device), target.to(device)
            outputs = model(images).to(device)
            model.zero_grad()
            loss_ = loss_func(outputs, target).to(device)
            prox_term = norm2_model(sub_models(model, old_model))
            for w, w_t in zip(model.parameters(), old_model.parameters()):
                prox_term += (w - w_t).norm(2) ** 2
#             print('loss: ', round(loss_.item(), 2), 'prox_term: ', round(prox_term.item(), 10))
            loss_ += (mu / 2.0) * prox_term
            loss_.backward()
            optimizer.step()
            if normalize_w:           # after each step, normalize the last layer
                normalize(model.w.weight)
            if step == steps:
                break
    return


def client_train_scaffold(train_loader, loss_func, optimizer, model, old_model, control, \
                      server_control, device, steps, lr): 
    '''the train_loader has to be Infinite_Dataloader'''
    step = 0
    model.train()
    model.to(device)
    old_model.to(device)
    control.to(device)
    server_control.to(device)
    while step < steps:
        for images, target in train_loader:
            step += 1
            images = images.to(device)
            target = target.to(device)
            outputs = model(images).to(device)
            model.zero_grad()
            loss_ = loss_func(outputs, target).to(device)
            loss_.backward()
            optimizer.step()   # y_i = y_i - eta_l * g_i(y_i)
            with torch.no_grad():
                for param_model, param_control, param_server \
                        in zip(model.parameters(), control.parameters(), server_control.parameters()):
                    param_model += lr * (param_control - param_server)   # y_i = y_i + eta_l (c_i - c)
            if step >= steps:
                break
    with torch.no_grad():
        temp_control = copy.deepcopy(control)
        for param_temp_control, param_control, param_server, param_old_model, \
                                            param_model in zip(temp_control.parameters(), \
                            control.parameters(), server_control.parameters(), old_model.parameters(), \
                                                                  model.parameters()):
            param_temp_control = param_control - param_server + \
                    (param_old_model - param_model) / (steps * lr)  # c_i^+ = c_i - c + (x - y_i) /(K eta_l)
        delta_c = sub_models(temp_control, control)
        assign_model(control, temp_control)
    return delta_c


def client_train(train_loader, loss_func, optimizer, model, test_loader, device, \
                    client_id, epochs, output_dir, show=True, save=True): 
    
    output_dir = os.path.join(output_dir, f'client_{client_id}')
    model.train()
    model.to(device)
    if save:
        os.makedirs(output_dir, exist_ok=True)
        csv_file = os.path.join(output_dir, f'client_{client_id}_log.csv')
        to_csv(csv_file, ['epoch', 'loss', 'test acc'], mode='w')
    
    # use tqdm to monitor progress
    for epoch in range(epochs):
        if show:
            t = tqdm(train_loader)
        else:
            t = train_loader
        for images, target in t:
            images = images.to(device)
            target = target.to(device)
            outputs = model(images).to(device)
            model.zero_grad()
            loss_ = loss_func(outputs, target).to(device)
            loss_.backward()
            optimizer.step()
            if show:
                t.set_description(f'epoch: {epoch}, client: {client_id}, loss: {loss_:.6f}')
        acc, loss_ = accuracy_loss(model, test_loader, loss_func, device, show=show)
        if save:
            to_csv(csv_file, [epoch, loss.item(), acc], mode='a')
    if save:
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), \
                'optimizer' : optimizer.state_dict()}, output_dir + f'/model_{client_id}_last.pth')

    return model, loss_


def client_train_w(train_loader, loss_func, optimizer, model, device, \
                    epoch=1, steps=10, normalize_w=True, show=True, save=True): 
    '''
    the train_loader has to be Infinite_Dataloader
    norm means whether to normalize the last layer
    ''' 
    model.train()
    model.to(device)
    for _ in range(epoch):
        step = 0
        for images, targets in train_loader:
            step += 1
            weights = deepcopy(model.w.weight).to(device)
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images).to(device)
            model.zero_grad()
            loss_ = loss_func(outputs, targets).to(device)
            loss_.backward()
            optimizer.step()
            with torch.no_grad():
                for i in range(model.w.weight.shape[0]):
                    if i is not label:
                        model.w.weight[i] = weights[i]
                    else:
                        continue
            if normalize_w:           # after each step, normalize the last layer
                normalize(model.w.weight)
            if step > steps:
                break
    return


def client_train_fedlc(train_loader, loss_func, optimizer, model, device, steps, label_margin):
    model.train()
    model = model.to(device)
    step = 0
    while step < steps:
        for images, target in train_loader:
            model.zero_grad()
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)
            outputs = model(images).to(device)
            outputs -= label_margin.expand(target.shape[0], -1).to(device)
            loss_ = loss_func(outputs, target).to(device)
            loss_.backward()
            optimizer.step()
            step += 1
            if step == steps:
                break
    return


def client_train_fedrs(train_loader, loss_func, optimizer, model, device, steps, class_access):
    model.train()
    model = model.to(device)
    step = 0
    while step < steps:
        for images, target in train_loader:
            model.zero_grad()
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)
            outputs = model(images).to(device)
            outputs *= class_access.expand(target.shape[0], -1).to(device)
            loss_ = loss_func(outputs, target).to(device)
            loss_.backward()
            optimizer.step()
            step += 1
            if step == steps:
                break
    return


def client_train_feddecorr(feddecorr, train_loader, loss_func, optimizer, model, device, steps, feddecorr_coef):
    model.train()
    model = model.to(device)
    step = 0
    while step < steps:
        for images, target in train_loader:

            model.zero_grad()
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)
            outputs, features = model(images)
            outputs.to(device)
            loss_ = loss_func(outputs, target).to(device)
            loss_feddecorr = feddecorr(features)
            loss_ += feddecorr_coef * loss_feddecorr
            loss_.backward()
            optimizer.step()
            step += 1

            if step == steps:
                break
    return

if __name__ == "__main__":
    ...