'''
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the BSD 3.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3.0 License for more details.
'''

import sys
import statistics
import csv
import json
import logging
import os
import pickle
import math

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from utils.eval import accuracies_losses, mean_std
from utils.aggregate import norm2_model, zero_model, sub_models


def log_global_epoch(alg, args, train_loaders, test_loaders, loss_func, device, train_writer, \
                         test_writer, f, t,):
    if t % args.log_interval == 0: # or t >= args.num_rounds - 100:   # computing every round in 100 is too expensive
        logging.info(f'| global round: %s', t)

        #etest
        train_metrics = compute_metrics(alg.models, train_loaders, loss_func, device)
        test_metrics = compute_metrics(alg.models, test_loaders, loss_func, device)

        log_metrics('train', *train_metrics, train_writer, t)
        log_metrics('test', *test_metrics, test_writer, t)

        w_norms = alg.models[0].w.weight.norm(dim=1).detach().cpu().numpy()
        model_norm = math.sqrt(norm2_model(alg.models[0]))

        for i, w_norm in enumerate(w_norms):
            train_writer.add_scalar(f'w_norm/class_{i}', w_norm, t)
        train_writer.add_scalar('w_norm/entire_matrix', np.sqrt((w_norms**2).sum()), t)
        train_writer.add_scalar('model_norm', model_norm, t)

        #log to csv
        train_losses, train_accs, train_acc_mean, _ = train_metrics
        test_losses, test_accs, test_acc_mean, _ = test_metrics
        f.write(f'{t},\"{train_losses}\",\"{train_accs}\",{train_acc_mean},' +
                    f'\"{test_losses}\",\"{test_accs}\",{test_acc_mean},\"{w_norms.tolist()}\"\n')

 
        f.flush()
        return


def compute_metrics(models, loaders, loss_func, device):
    accs, loss = accuracies_losses(models, loaders, loss_func, device)
    losses_ = [round(x, 3) for x in loss]
    accs = [round(x, 3) for x in accs]
    acc_mean, acc_std = (round(x, 3) for x in mean_std(accs))
    loss_mean, loss_std = (round(x, 3) for x in mean_std(losses_))

    return losses_, accs, acc_mean, acc_std


def log_metrics(prefix, losses_, accs, acc_mean, acc_std, writer, t):

    logging.info(f'{prefix} losses: {losses_}')
    logging.info(f'{prefix} accs: {accs}')
    logging.info(f'{prefix} acc mean (std): {acc_mean} ({acc_std})')

    num_clients = len(losses_)
    writer.add_scalar('acc', acc_mean, t)
    writer.add_scalar('loss', statistics.mean(losses_), t)
    for i in range(num_clients):
        writer.add_scalar(f'acc/client_dist_{i}', accs[i], t)
        writer.add_scalar(f'loss/client_dist_{i}', losses_[i], t)
        
        
def to_csv(csv_file, row, mode='w'):
    with open(csv_file, mode) as f:
        writer = csv.writer(f)
        writer.writerow(row)



