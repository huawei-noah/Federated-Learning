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
import json
import re
import os

import numpy as np
import torch
import torch.optim as optim

import sys
path_ = os.getcwd()
sys.path.append(path_)

import flsuite
import flsuite.data as data
import flsuite.utils as utils
import flsuite.algs.trainers as trainers
from flsuite.algs import individual_train
from flsuite.data.datasets.pacs import augmentation, transform, weak_augmentation

batch_size = 32
local_steps = 200
rounds = 80
steps = local_steps * rounds

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--trainer", type=str, default="ERM")
parser.add_argument("-w", "--workers", type=int, default=4)
parser.add_argument("-d", "--domain", type=str, default="S")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--augment", action="store_true")
parser.add_argument("--no-augment", dest="augment", action="store_false")
parser.add_argument("--weak-augment", action="store_true")
parser.add_argument("--groupnorm", action="store_true")
parser.add_argument("--batchnorm", dest="groupnorm", action="store_false")
parser.add_argument("--resnet50", action="store_true")
parser.add_argument("--pretrain", action="store_true")
parser.add_argument("--no-pretrain", dest="pretrain", action="store_false")
parser.add_argument("--pin_memory", action="store_true")
parser.add_argument("--no-pin_memory", dest="pin_memory", action="store_false")
parser.set_defaults(pin_memory=True)
parser.set_defaults(weak_augment=False)
parser.set_defaults(groupnorm=True)
parser.set_defaults(resnet50=False)
parser.set_defaults(augment=False)
parser.set_defaults(pretrain=True)


def experiment(args):
    if args.weak_augment:
        args.augment = False
        path_augment = "weak_augmentation"
    elif args.augment:
        args.weak_augment = False
        path_augment = "augmentation"
    else:
        path_augment = "no_augmentation"

    print(args)
    path = (
        f"./data/experiments/domain_generalization/PACS/{path_augment}/{args.trainer}/"
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    trainer = getattr(trainers, args.trainer)

    arg_transform = None
    if args.augment:
        arg_transform = augmentation
    elif args.weak_augment:
        arg_transform = weak_augmentation
    else:
        arg_transform = transform

    dataset = data.datasets.PACS("./data/datasets")
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

    # all data loader
    val_loaders = data.build_dataloaders(
        val_sets,
        batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=args.workers,
    )
    test_loader = data.build_dataloaders(
        [test_set],
        batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=args.workers,
    )[0]
    eval_loaders = [*val_loaders, test_loader]

    torch.manual_seed(args.seed)
    train_loaders = data.build_dataloaders(
        train_sets,
        batch_size,
        seed=args.seed,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.workers,
    )
    global_loader = data.utils.DataLoaderWrapper(train_loaders)

    saver = utils.save.Saver(path, args.domain, overwrite=True)
    with open(path + f"{args.domain}/script_meta.json", "w+") as f:
        json.dump(vars(args), f)

    global_model = flsuite.models.ResNet(
        seed=args.seed,
        pretrain=args.pretrain,
        resnet50=args.resnet50,
        groupnorm=args.groupnorm,
    )
    global_model = trainer.bind_to(
        global_model, optimizer=optim.Adam(global_model.parameters(), lr=5e-5)
    )
    global_model = individual_train(
        global_model,
        global_loader,
        steps,
        validation_loader=eval_loaders,
        device=device,
        save=saver,
        eval_steps=50,
    )

    print(
        "Validation accuracy: %.3f"
        % np.mean(utils.eval.accuracies(global_model, val_loaders, device))
    )
    print("Test accuracy: %.3f" % utils.eval.accuracy(global_model, test_loader))


if __name__ == "__main__":
    experiment(parser.parse_args())
