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
import csv
import json
import os
import numpy as np

def save_acc_loss(json_file, t, acc, loss):
    result = {}
    result['epoch'] = t
    result['accs'] = list(acc)
    result['losses'] = list(loss)
    with open(json_file, 'a') as f:
        f.write(json.dumps(result, sort_keys=True) + '\n')


def best_p(accs, percent=0.2):
    sorted_ = np.sort(accs)
    num = len(accs)
    best = sorted_[num - int(num * percent):]
    return mean_std(best)[0]

def worst_p(accs, percent=0.2):
    sorted_ = np.sort(accs)
    num = len(accs)
    worst = sorted_[:int(num * percent)]
    return mean_std(worst)[0]


def mean_std(accs):
    return np.mean(accs), np.std(accs)


class Tee:  
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

def to_csv(csv_file, row, mode='w'):
    with open(csv_file, mode) as f:
        writer = csv.writer(f)
        writer.writerow(row)
        
 # present things in a nice format 

def print_acc(list_):
    for elem in list_:
        print(f'{elem * 100:.2f}%', end='\t')
    print('\n')
    
def round_list(list_, dec=4):
    return [round(elem, dec) for elem in list_]


def read_json(file):
    results = []
    with open(file) as f:
        for line in f:
            j_content = json.loads(line)
            results.append(j_content)
    return results

def print_mean_std(mean, std):
    return f'{mean}' + '$_{' + f'\pm {std}' + '}$'

def plus_minus(alg, epoch, means, stds, worsts, worsts_p, bests, bests_p):
    mean, std = round_list(list(mean_std(means)), dec=2)
    mean_1, std_1 = round_list(list(mean_std(stds)), dec=2)
    mean_2, std_2 = round_list(list(mean_std(worsts)), dec=2)
    mean_3, std_3 = round_list(list(mean_std(worsts_p)), dec=2)
    mean_4, std_4 = round_list(list(mean_std(bests)), dec=2)
    mean_5, std_5 = round_list(list(mean_std(bests_p)), dec=2)
    print(alg + '& ' + str(epoch) + ' & ' + print_mean_std(mean, std) + ' & ' + \
          print_mean_std(mean_1, std_1) + ' & ' + print_mean_std(mean_2, std_2) + ' & ' +\
          print_mean_std(mean_3, std_3) + ' & ' + print_mean_std(mean_4, std_4) + ' & ' + \
          print_mean_std(mean_5, std_5))
    return

def find_best_time(folder, alg, seed, limit):
    file = os.path.join(folder, alg, f'seed_{seed}', 'log.json')
    results = read_json(file)
    best_acc = -np.inf
    for i in range(len(results)):
        if i > limit:
            break
        accs = results[i]['accs']
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_i, best_acc = i, mean_acc
    return best_i, best_acc


def alg_to_stats(alg='FedAvg', root='./results/CIFAR10/split_0.8', time=-1):
    means, stds, worsts, worsts_p, bests, bests_p = [], [], [], [], [], []
    for seed in range(3):
        log = os.path.join(root, alg, f'seed_{seed}', 'log.json')
        epoch = read_json(log)[time]['epoch']
        accs = read_json(log)[time]['accs']
        mean, std = mean_std(accs)
        worst = np.min(accs)
        worstp = worst_p(accs, percent=0.1)
        best = np.max(accs)
        bestp = best_p(accs, percent=0.1)
        means.append(mean)
        stds.append(std)
        worsts.append(worst)
        worsts_p.append(worstp)
        bests.append(best)
        bests_p.append(bestp)
    plus_minus(alg, epoch, means, stds, worsts, worsts_p, bests, bests_p)
    return
