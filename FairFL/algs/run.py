'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

# training
import sys
sys.path.append("..")

import argparse
import os
import torch
import random
from copy import deepcopy
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pickle
import threading
from tqdm import tqdm
import json
from utils.io import Tee, to_csv, mean_std, print_acc, round_list
from utils.eval import accuracy, accuracies, losses
from utils.aggregate import aggregate, aggregate_lr, zero_model, aggregate_momentum
from models.models import CNN, CNNCifar, RNN_Shakespeare, RNN_StackOverflow
from models.resnet import resnet18
from utils.save import save_acc_loss
from algorithms import *
import logging


root = '..' 

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--name', type=str, default='')
parser.add_argument('--device', type=str, default='7')
parser.add_argument('--data_dir', type=str, default='iid-4')
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--algorithm', type=str, default='qFedAvg')
parser.add_argument('--num_clients', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=0, help='for data loader')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--num_local_epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--save_epoch', type=int, default=10)
parser.add_argument('--q', type=float, default=0.1, help='for qFFL')
parser.add_argument('--alpha', type=float, default=0.01, help='for TERM')
parser.add_argument('--base', type=float, default=2.0, help='for PropFair')
parser.add_argument('--huber', type=bool, default=False, help='for PropFair')
parser.add_argument('--epsilon', type=float, default=0.2, help='for FedMGDA+')
parser.add_argument('--coeff', type=float, default=0.1, help='for GiFair')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--pretrain', type=str, default=None, help='the path to the pretrained model')
parser.add_argument('--transform', type=bool, default=False)
parser.add_argument('--pretrained', type=bool, default=False)
args = parser.parse_args()
# fix randomness

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8' 
torch.use_deterministic_algorithms(True)   
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


if args.algorithm == 'qFedAvg':
    output_dir = os.path.join(root, 'results', args.dataset, args.data_dir, args.algorithm + f'-{args.q}',  f'seed_{args.seed}_{args.name}')
elif args.algorithm == 'TERM':
    output_dir = os.path.join(root, 'results', args.dataset, args.data_dir, args.algorithm + f'-{args.alpha}',  f'seed_{args.seed}_{args.name}')
elif args.algorithm == 'PropFair' and not args.huber:
    output_dir = os.path.join(root, 'results', args.dataset, args.data_dir, args.algorithm + f'-{args.base}',  f'seed_{args.seed}_{args.name}')
elif args.algorithm == 'PropFair' and args.huber:
    output_dir = os.path.join(root, 'results', args.dataset, args.data_dir, args.algorithm + f'-huber-{args.base}',  f'seed_{args.seed}_{args.name}')
elif args.algorithm == 'GiFair':
    output_dir = os.path.join(root, 'results', args.dataset, args.data_dir, args.algorithm + f'-{args.coeff}',  f'seed_{args.seed}_{args.name}')
elif args.algorithm == 'FedMGDA':
    output_dir = os.path.join(root, 'results', args.dataset, args.data_dir, args.algorithm + f'-{args.epsilon}',  f'seed_{args.seed}_{args.name}')  
else:
    output_dir = os.path.join(root, 'results', args.dataset, args.data_dir, args.algorithm,  f'seed_{args.seed}_{args.name}')
args.data_dir = os.path.join(root, 'data', args.dataset, args.data_dir)
log_file = os.path.join(output_dir, 'log.txt')
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode="a", format='%(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


with open(os.path.join(output_dir, 'args.json'), 'w') as fp:
    json.dump(vars(args), fp)

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # use the first GPU
else:
    device = torch.device('cpu')

in_file = os.path.join(args.data_dir, 'in.pickle')
out_file = os.path.join(args.data_dir, 'out.pickle')

# if args.dataset == 'Shakespeare':
#     with open(in_file, 'rb') as f_in:    # we used torch to save Shakespeare
#         in_data = torch.load(f_in)
#     with open(out_file, 'rb') as f_out:
#         out_data = torch.load(f_out)  
# else:
with open(in_file, 'rb') as f_in:   # we used pickle to save vision datasets
    in_data = pickle.load(f_in)
with open(out_file, 'rb') as f_out:
    out_data = pickle.load(f_out)  

weights = np.array([len(in_data[i]) for i in range(args.num_clients)])
weights_test = np.array([len(out_data[i]) for i in range(args.num_clients)])
weights = list(weights / np.sum(weights))


# data augmentation
"""Transformation"""
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
#    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

"""Dataset"""
class CIFARDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


train_loaders = [DataLoader(
    dataset=CIFARDataset(in_data[i], transform=transform_train if args.transform else None),
    batch_size=args.batch_size,
    num_workers=args.num_workers, drop_last=False, pin_memory=True, shuffle=True)
    for i in range(args.num_clients)]

test_loaders = [DataLoader(
    dataset=CIFARDataset(out_data[i], transform=transform_test if args.transform else None),
    batch_size=args.batch_size,
    num_workers=args.num_workers, drop_last=False, pin_memory=True, shuffle=True)
    for i in range(args.num_clients)]


if args.dataset == 'MNIST':
    models = [CNN() for _ in range(args.num_clients)]
elif args.dataset == 'CIFAR10':
    models = [resnet18(num_classes=10, pretrained=args.pretrained)  for _ in range(args.num_clients)]
elif args.dataset == 'CIFAR100':
    models = [resnet18(num_classes=100, pretrained=args.pretrained)  for _ in range(args.num_clients)]
elif args.dataset == 'TinyImageNet':
    models = [resnet18(num_classes=200, pretrained=args.pretrained)  for _ in range(args.num_clients)]
elif args.dataset == 'FEMNIST':
    models = [CNN_FEMNIST() for _ in range(args.num_clients)]      
elif args.dataset == 'Shakespeare':
    models = [RNN_Shakespeare()  for _ in range(args.num_clients)]
elif args.dataset == 'StackOverflow':
    models = [RNN_StackOverflow()  for _ in range(args.num_clients)]     

# loss functions, optimizer
loss_func = nn.CrossEntropyLoss()
optimizers = [optim.SGD(model.parameters(), lr = args.learning_rate, \
                        momentum=0.0) for model in models]

json_file = os.path.join(output_dir, 'log.json')

# checkpoint
if args.pretrain is not None:
    model_path = args.pretrain
    print('loading pretrained model')
else:
    model_path = output_dir  + f'/model_last.pth'
if os.path.exists(model_path):
    start_epoch = torch.load(model_path)['epoch']
    for model in models:
        model.load_state_dict(torch.load(model_path)['state_dict'])
else:
    start_epoch = 0
    with open(json_file, 'w') as f:
        f.write('')
    logging.info(vars(args))
    logging.info(f'output_dir: {output_dir}')
    logging.info(f'total train samples: {np.sum([len(in_data[i]) for i in range(args.num_clients)])}')
    logging.info(f'total test samples: {np.sum(weights_test)}')
    logging.info(f'total samples: {np.sum([len(in_data[i]) for i in range(args.num_clients)]) +np.sum(weights_test)}')
    logging.info(f'samples: {[len(in_data[i]) for i in range(args.num_clients)]}')


if args.algorithm == 'FedAvg':
    alg = FedAvg(models, optimizers, args.num_clients, args.num_local_epochs, loss_func)
elif args.algorithm == 'AFL':
    alg = AFL(models, optimizers, args.num_clients, args.num_local_epochs, loss_func, \
              lambda_=weights, step_size_lambda=0.1)
elif args.algorithm == 'PropFair':
    alg = PropFair(models, optimizers, args.num_clients, args.num_local_epochs, \
             loss_func, base=args.base, epsilon=0.2, huber=args.huber)
elif args.algorithm == 'qFedAvg':
    alg = qFedAvg(models, optimizers, args.num_clients, args.num_local_epochs, \
             loss_func, Lipschitz = 1 / args.learning_rate, q=args.q)
elif args.algorithm == "TERM":
    alg = TERM(models, optimizers, args.num_clients, args.num_local_epochs, loss_func, alpha=args.alpha)
elif args.algorithm == "FedMGDA":
    alg = FedMGDA(models, optimizers, args.num_clients, args.num_local_epochs, loss_func, \
                  epsilon=args.epsilon, global_lr=1.0)
elif args.algorithm == "GiFair":
    alg = GiFair(models, optimizers, args.num_clients, args.num_local_epochs, loss_func, weights, \
                 coeff=args.coeff)
else:
    raise NotImplemented

mean_accs = []
for t in range(start_epoch + 1, args.num_epochs + 1):
    alg.local_updates(train_loaders, device)
    alg.aggregate(weights=weights)
    if t % args.save_epoch == 0:
        logging.info(f'| global epoch: {t}')
        torch.save({'epoch': t, 'state_dict': models[0].state_dict()}, \
            output_dir  + f'/model_last.pth')
        test_accs = accuracies(alg.models, test_loaders, device)
        train_accs = accuracies(alg.models, train_loaders, device)
        losses_ = losses(alg.models, train_loaders, loss_func, device)
        _, std = mean_std(test_accs)
        mean = np.dot(test_accs, weights)
        _, train_std = mean_std(train_accs)
        train_mean = np.dot(train_accs, weights)
        _, loss_std = mean_std(losses_)
        mean_loss = np.dot(losses_, weights)
        mean_accs.append(mean)
        logging.info(f'losses: {round_list(losses_)}')
        logging.info(f'test acc: {round(mean, 2)}({round(std, 2)}), worst: {round(min(test_accs), 2)} train loss: {round(mean_loss, 4)}({round(loss_std, 4)}), train_acc: {round(train_mean, 2)}({round(train_std, 2)})')
        logging.info(f'test accs: {[round(i, 3) for i in test_accs]}')
        save_acc_loss(json_file, t, test_accs, losses_)
    else:
        print(f'| global epoch: {t}')

acc_file = "mean_acc.pkl".format(args.dataset, args.seed)
acc_file = os.path.join(output_dir, acc_file)

with open(acc_file, 'wb') as f_out:
    pickle.dump(mean_accs, f_out) 
