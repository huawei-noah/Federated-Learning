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
from typing import Optional, Union

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from . import trainers
from flsuite import utils


def individual_train(
    model: nn.Module,
    train_loader: DataLoader,
    steps: int,
    validation_loader: Optional[DataLoader] = None,
    loss_func: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    trainer: Optional[trainers.Trainer] = None,
    device: Optional[torch.device] = None,
    verbose: int = -1,
    eval_steps: int = 1,
    save: Optional[Union[str, utils.save.Saver]] = None,
    **kwargs
) -> nn.Module:
    """
    Train model for a given number of steps
    Args:
        model: nn.Module
            torch model to train
        train_loader: DataLoader
            train data loader
        steps: int
            number of steps to train
        validation_loader: DataLoader
            validation data loader
        loss_func: nn.Module
            loss function
            If None, it will attempt to use the attribute `model.criterion`
        optimizer: torch.optim.Optimizer
            optimization method for training
            If None, it will attempt to use the attribute `model.optimizer`
        device: torch.device
            Device to run training routine
            If None, it will use the same device used for the model
        verbosity: int
            verbosity mode
        eval_steps: int
            evaluate the validation set at each steps
        save: str or Saver
    Returns:
        nn.Module
            trained model

    """
    saver = utils.save.Saver(save) if isinstance(save, str) else save
    device = next(model.parameters()).device if device is None else device
    utils.io.config_logger(verbose)
    model.to(device)

    if hasattr(model, "trainer"):
        if optimizer is not None:
            model.trainer.optimizer = optimizer
        if loss_func is not None:
            model.trainer.criterion = loss_func
        trainer = model.trainer

    else:
        trainer_kwargs = {"optimizer": optimizer, "loss_func": loss_func}
        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}
        trainer = trainers.ERM(model, **trainer_kwargs)

    if validation_loader and not isinstance(validation_loader, (tuple, list)):
        validation_loader = [validation_loader]

    if saver:
        saver.write_meta(
            model=str(model),
            loss=str(trainer.criterion),
            optimizer=str(trainer.optimizer),
            trainer=str(trainer),
        )

    step = 0
    log = dict()
    while step < steps:
        for batch in train_loader:
            log["step"] = step

            if not isinstance(batch[0], (list, tuple)):
                batch = [batch]

            trainer_logs = trainer.update(batch, device=device)
            log.update(trainer_logs)

            logging.info("step %d" % (log["step"]))
            for key, value in trainer_logs.items():
                logging.info("%s: %.4f" % (key.replace("_", " "), value))

            if validation_loader is not None and (
                step % eval_steps == 0 or step + 1 == steps
            ):
                log["val_loss"] = utils.eval.losses(
                    model, validation_loader, trainer.criterion, device
                )
                log["val_acc"] = utils.eval.accuracies(model, validation_loader, device)
                logging.info(
                    "val. acc.: %s" % str(utils.print.round_list(log["val_acc"], 4))
                )
                logging.info(
                    "val. losses: %s" % str(utils.print.round_list(log["val_loss"], 4))
                )

            if saver:
                saver.write_csv(**log)
                saver.save_model(model)

            step += 1
            if step >= steps:
                break

    utils.io.config_logger(-1)
    if kwargs.get("last_log", None):
        return model, log
    return model
