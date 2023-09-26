'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import logging
import pickle
import gc
import os
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .individual_train import individual_train
from flsuite.utils.eval import accuracies, losses
from flsuite.utils import aggregate
from flsuite import utils


def var_min(
    models: List["nn.Module"],
    train_loaders: List["DataLoader"],
    rounds: int,
    local_steps: Union[int, list] = 1,
    var_beta: Union[float, int] = 10,
    anneal_rounds: int = 1,
    local_validation: List["DataLoader"] = None,
    local_eval_steps: int = 1,
    global_validation: "DataLoader" = None,
    device: Optional[torch.device] = None,
    verbose: int = -1,
    save: Optional[str] = None,
) -> "nn.Module":
    """
    Train models and perform variance minimization aggregation
    Args:
        models: List["nn.Module"]
            list of models
        train_loaders: List["DataLoader"]
            list of train dataloaders
        rounds: int
            number of global rounds
        local_steps: int
            number of steps to execute for each client
        local_validation: List["DataLoader"]
            list of validation dataloaders for local models
        local_eval_steps: int
            evaluate the validation set at each local steps
        global_validation: "DataLoader"
            validation dataloader for global model
        device: torch.device
            device to cast data and models
        verbose: int
            verbosity mode
        save: str
            path to save logs
    Returns:
        global model trained
    """
    # configure verbosity mode
    utils.io.config_logger(verbose)

    # if given an int, steps are uniform for all models
    if isinstance(local_steps, int):
        local_steps = [local_steps] * len(models)

    if local_validation is None:
        local_validation = [None] * len(models)
    elif isinstance(local_validation, DataLoader):
        local_validation = [local_validation] * len(models)

    # if device is not set, get the device from model
    if device is None:
        device = next(models[0].parameters()).device

    for model in models:
        model.to(device)

    if global_validation and not isinstance(global_validation, (tuple, list)):
        global_validation = [global_validation]

    if save:
        global_saver = utils.save.Saver(save, "global")
        saver = [
            utils.save.Saver(save, "client_%d" % i, overwrite=False)
            for i in range(len(models))
        ]
    else:
        global_saver = None
        saver = [None] * len(models)

    assert len(models) == len(local_validation)
    assert len(models) == len(train_loaders)
    assert len(models) == len(local_steps)

    # calculate weights for the aggregation step
    weights = np.array([len(x.dataset) for x in train_loaders])
    weights = list(weights / np.sum(weights))
    global_model = aggregate.aggregate(models, weights=weights)

    for t in range(rounds):
        logs = [{}] * len(models)
        old_models = deepcopy(models)
        for i, model in enumerate(models):
            models[i], logs[i] = individual_train(
                model,
                train_loaders[i],
                local_steps[i],
                local_validation[i],
                device=device,
                eval_steps=local_eval_steps,
                save=saver[i],
                last_log=True,
            )

        pseudo_grad = [
            aggregate.sub_models(old, new) for old, new in zip(old_models, models)
        ]
        loss_list = np.array(
            utils.eval.losses(models, train_loaders, models[0].criterion, device)
        )

        term_1 = aggregate.avg_models(pseudo_grad)
        term_2 = [aggregate.sub_models(x, term_1) for x in pseudo_grad]
        term_2 = aggregate.avg_models(term_2, (loss_list - loss_list.mean()).tolist())

        var_beta_val = var_beta if rounds > anneal_rounds else 0.5
        update = aggregate.avg_models([term_1, term_2], [1, 2 * var_beta_val])
        global_model = aggregate.sub_models(global_model, update)
        models = aggregate.assign_models(models, global_model)

        del pseudo_grad, update, term_1, term_2
        gc.collect()
        torch.cuda.empty_cache()

        utils.io.config_logger(verbose)

        log = dict()
        log["global_round"] = t
        log["train_acc"] = [x["train_acc"] for x in logs]
        log["train_loss"] = [x["train_loss"] for x in logs]

        logging.info("global round: %d" % log["global_round"])
        logging.info(
            "train acc.: %s" % str(utils.print.round_list(log["train_acc"], 3))
        )
        logging.info(
            "train losses: %s" % str(utils.print.round_list(log["train_loss"]))
        )

        if local_validation[0]:
            log["val_acc"] = [x["val_acc"] for x in logs]
            log["val_loss"] = [x["val_loss"] for x in logs]

            for i in range(len(local_validation[0])):
                valid_acc = utils.print.round_list([x[i] for x in log["val_acc"]], 3)
                valid_loss = utils.print.round_list([x[i] for x in log["val_loss"]])
                logging.info("val. acc. [%d]: %s" % (i, str(valid_acc)))
                logging.info("val. losses [%d]: %s" % (i, str(valid_loss)))

        if global_validation:
            log["global_val_loss"] = utils.eval.losses(
                global_model, global_validation, global_model.criterion, device
            )
            log["global_val_acc"] = utils.eval.accuracies(
                global_model, global_validation, device
            )
            logging.info(
                "global val. acc.: %s"
                % str(utils.print.round_list(log["global_val_acc"], 3))
            )
            logging.info(
                "global val. loss: %s"
                % str(utils.print.round_list(log["global_val_loss"]))
            )

        if global_saver:
            global_saver.write_csv(**log)
            global_saver.save_model(global_model)

    utils.io.config_logger(-1)
    return global_model
