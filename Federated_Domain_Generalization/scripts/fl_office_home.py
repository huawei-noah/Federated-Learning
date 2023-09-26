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
import re
import os

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import ConcatDataset

import sys
path_ = os.getcwd()
sys.path.append(path_)

import flsuite
import flsuite.data as data
import flsuite.utils as utils
import flsuite.algs as algs
from flsuite.algs.trainers import ERM
from flsuite.data.datasets.pacs import transform, augmentation, weak_augmentation


batch_size = 32
local_steps = 200
rounds = 80
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--alg", type=str, default="fed_avg")
parser.add_argument("-d", "--domain", type=str, default="Clipart")
parser.add_argument("-b", "--beta", type=float, default=200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-w", "--workers", type=int, default=8)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--augment", action="store_true")
parser.add_argument("--no-augment", dest="augment", action="store_false")
parser.add_argument("--weak-augment", action="store_true")
parser.set_defaults(weak_augment=False)
parser.set_defaults(augment=False)

def experiment(args):
    arg_transform = None
    if args.weak_augment:
        args.augment = False
        path_augment = "weak_augmentation"
        arg_transform = weak_augmentation
    elif args.augment:
        args.weak_augment = False
        path_augment = "augmentation"
        arg_transform = augmentation
    else:
        path_augment = "no_augmentation"
        arg_transform = transform
    print(args)

    path = f"./data/experiments/federated_learning/OfficeHome/{args.domain}/{path_augment}/{args.alg.lower()}/"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    if args.alg == "var_min":
        algorithm = algs.var_min
        kwargs = {"var_beta": 0.5}
    elif args.alg == "gen_afl":
        algorithm = algs.afl
        kwargs = {"lambda_min": -1}
    else:
        algorithm = getattr(algs, args.alg)
        kwargs = {}

    dataset = data.datasets.OfficeHome("./data/datasets")
    train_sets, val_sets = list(), list()
    for i in range(len(dataset)):
        if not dataset[i].root.endswith(args.domain):
            train_idx, val_idx = data.processing.train_test_split(
                np.arange(len(dataset[i])), 0.1, args.seed
            )
            filtered_train = dataset[i].filter_by_id(train_idx)
            filtered_train.set_transform(arg_transform)
            train_sets.append(filtered_train)
            val_sets.append(dataset[i].filter_by_id(val_idx))
        else:
            test_set = dataset[i]

    val_loaders = data.build_dataloaders(
        val_sets, batch_size, shuffle=False, num_workers=args.workers
    )
    test_loader = data.build_dataloaders(
        [test_set], batch_size, shuffle=False, num_workers=args.workers
    )[0]

    global_val = [*val_loaders, test_loader]
    local_val = [(loader, test_loader) for loader in val_loaders]

    # all train_sets are roughly the same size at this point
    train_sets_ = list()
    for i, x in enumerate(train_sets):
        np.random.seed(args.seed)
        sizes = np.random.normal(loc=len(x) // 5, scale=100, size=4).astype(int)
        idxs = data.processing.non_iid_split(
            np.arange(len(x)), sizes, seed=args.seed, beta=args.beta, labels=x.targets
        )
        train_sets_.append([x.filter_by_id(y) for y in idxs])
    train_sets = train_sets_

    flatten = lambda x: [y for z in x for y in z]
    local_val = flatten([[x] * len(y) for x, y in zip(local_val, train_sets)])
    train_sets = flatten(train_sets)
    print("num clients", len(train_sets))
    print("dataset sizes", [len(x) for x in train_sets])

    torch.manual_seed(args.seed)
    train_loaders = data.build_dataloaders(
        train_sets, batch_size, shuffle=True, num_workers=args.workers
    )
    models = flsuite.models.model_loader(
        "OfficeHome", len(train_sets), args.seed, groupnorm=True
    )
    models = [
        ERM.bind_to(x, optimizer=optim.Adam(x.parameters(), lr=5e-5)) for x in models
    ]
    global_model = algorithm(
        models,
        train_loaders,
        rounds,
        local_steps,
        local_validation=local_val,
        local_eval_steps=40,
        global_validation=global_val,
        device=device,
        save=path,
        verbose=1,
        **kwargs,
    )

    print(
        "Validation accuracy: %.3f"
        % np.mean(utils.eval.accuracies(global_model, val_loaders, device))
    )
    print("Test accuracy: %.3f" % utils.eval.accuracy(global_model, test_loader))


if __name__ == "__main__":
    experiment(parser.parse_args())
