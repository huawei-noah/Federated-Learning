'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

import sys
path_ = os.getcwd()
sys.path.append(path_)

import flsuite
import flsuite.data as data
import flsuite.utils as utils
from flsuite.algs import fed_avg
from flsuite.algs.trainers import ERM

num_clients = 5
batch_size = 64

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--local_steps", type=int)
parser.add_argument("-r", "--rounds", type=int)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-d", "--data", type=str, default="original")
parser.add_argument("--gpu", type=int, default=0)

def experiment(args):
    print(args)

    path = f"./data/experiments/rounds_and_steps/{args.data}_{args.rounds}_{args.local_steps}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    dataset = data.datasets.RotatedMNIST("./data/datasets", seed=args.seed)
    train_sets = [
        data.processing.train_test_split(shard, 0.1, args.seed)
        for shard in dataset.datasets[:num_clients]
    ]
    train_sets, val_sets = list(zip(*train_sets))
    test_set = dataset.datasets[-1]

    all_train_loader = data.build_dataloaders(
        [data.merge_datasets(train_sets)], batch_size, shuffle=False
    )[0]
    val_loaders = data.build_dataloaders(val_sets, batch_size, shuffle=False)
    test_loader = data.build_dataloaders([test_set], batch_size, shuffle=False)[0]

    global_val = [*val_loaders, test_loader]
    local_val = [(loader, test_loader) for loader in val_loaders]

    if args.data != "original":
        if args.data == "da_15":
            transform = transforms.RandomRotation(15)
            train_sets = [
                data.utils.CustomDataset.parse_values(x, transform) for x in train_sets
            ]
        elif args.data == "da_30":
            transform = transforms.RandomRotation(30)
            train_sets = [
                data.utils.CustomDataset.parse_values(x, transform) for x in train_sets
            ]
        elif args.data == "da_45":
            transform = transforms.RandomRotation(45)
            train_sets = [
                data.utils.CustomDataset.parse_values(x, transform) for x in train_sets
            ]
        elif args.data == "da_60":
            transform = transforms.RandomRotation(60)
            train_sets = [
                data.utils.CustomDataset.parse_values(x, transform) for x in train_sets
            ]
        elif args.data == "da_blur":
            transform = transforms.GaussianBlur(5)
            train_sets = [
                data.utils.CustomDataset.parse_values(x, transform) for x in train_sets
            ]
        else:
            raise ValueError

    torch.manual_seed(args.seed)
    train_loaders = data.build_dataloaders(train_sets, batch_size)

    models = flsuite.models.model_loader("RMNIST", num_clients, args.seed)
    models = [
        ERM.bind_to(x, optimizer=optim.Adam(x.parameters(), lr=0.001)) for x in models
    ]
    global_model = fed_avg(
        models,
        train_loaders,
        args.rounds,
        args.local_steps,
        local_validation=local_val,
        local_eval_steps=10,
        global_validation=global_val,
        device=device,
        save=path,
        verbose=1,
    )

    print("Train accuracy: %.3f" % utils.eval.accuracy(global_model, all_train_loader))
    print(
        "Validation accuracy: %.3f"
        % np.mean(utils.eval.accuracies(global_model, val_loaders, device))
    )
    print("Test accuracy: %.3f" % utils.eval.accuracy(global_model, test_loader))

if __name__ == "__main__":
    experiment(parser.parse_args())