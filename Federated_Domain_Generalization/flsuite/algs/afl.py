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
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from flsuite import utils
from flsuite.utils.project import project
from .individual_train import individual_train


def afl(
    models: List["nn.Module"],
    train_loaders: List["DataLoader"],
    rounds: int,
    local_steps: Union[int, list] = 1,
    lambda_min: float = 0.0,
    step_size_lambda: float = 0.1,
    local_validation: List["DataLoader"] = None,
    local_eval_steps: int = 1,
    global_validation: "DataLoader" = None,
    device: Optional[torch.device] = None,
    verbose: int = -1,
    save: Optional[str] = None,
    **kwargs
) -> "nn.Module":
    """
    Train models and perform Agnostic Federated Learning (AFL) aggregation
    Args:
        models: List["nn.Module"]
            list of models
        train_loaders: List["DataLoader"]
            list of train dataloaders
        rounds: int
            number of global rounds
        local_steps: int
            number of steps to execute for each client
        step_size_lambda: float
            step size for the weight projection
        lambda_min: float
            projection parameter
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
        nn.Module
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
        local_validation = [validation_loaders] * len(models)

    # if device is not set, get the device from model
    if device is None:
        device = next(models[0].parameters()).device

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
    n_clients = len(models)
    afl_lambda = kwargs.get("afl_lambda", None)
    if afl_lambda is None:
        afl_lambda = np.array([len(x.dataset) for x in train_loaders])
        afl_lambda = list(afl_lambda / np.sum(afl_lambda))

    for t in range(rounds):
        logs = [{}] * len(models)
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

        utils.io.config_logger(verbose)
        global_model = utils.aggregate.aggregate(models, weights=afl_lambda)

        log = dict()
        log["global_round"] = t
        log["afl_lambda"] = afl_lambda
        log["train_acc"] = [x["train_acc"] for x in logs]
        log["train_loss"] = [x["train_loss"] for x in logs]

        logging.info("global round: %d" % log["global_round"])
        logging.info(
            "train acc.: %s" % str(utils.print.round_list(log["train_acc"], 3))
        )
        logging.info(
            "train losses: %s" % str(utils.print.round_list(log["train_loss"]))
        )
        logging.info("lambdas: %s" % str(utils.print.round_list(afl_lambda)))

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

        loss_list = utils.eval.losses(
            models, train_loaders, models[0].criterion, device
        )
        afl_lambda = get_afl_lambda(afl_lambda, loss_list)

    utils.io.config_logger(-1)
    return global_model


def get_afl_lambda(afl_lambda, loss_list, step_size_lambda=0.1, lambda_min=0.0):
    n_clients = len(loss_list)
    afl_lambda = np.array(afl_lambda) + step_size_lambda * np.array(loss_list)
    afl_lambda = (afl_lambda - lambda_min * np.ones(n_clients)) / (
        1 - n_clients * lambda_min
    )
    afl_lambda = np.array(utils.project.project(afl_lambda))
    return list(
        afl_lambda * (1 - n_clients * lambda_min) + lambda_min * np.ones(n_clients)
    )
