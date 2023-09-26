'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

# Code inspired from Facebook's DomainBed project
# https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py

from typing import Optional

import torch
import torch.nn as nn
from torch import autograd

from flsuite import utils


class Trainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self):
        pass

    def _get_device(self):
        return next(self.forward.__self__.parameters()).device

    def update(self, dataloader):
        """
        Update the model's parameters based on the training instances provided
        Args:
            dataloader: torch DataLoader
        Returns:
            train_loss: float
                training loss
            train_acc: float
                training accuracy
        """
        raise NotImplementedError

    @classmethod
    def bind_to(cls, model, **kwargs):
        default_args = model.__dict__
        default_args.update({k: v for k, v in kwargs.items() if v is not None})
        model.trainer = cls(model=model, **default_args)
        return model


class ERM(Trainer):
    def __init__(self, model: nn.Module, **kwargs):
        """
        Performs training according to the Empirical Risk Minimization (ERM) principle (Vapnik, 1991)

        Args:
            model: nn.Module
                model to be trained
            optimizer: torch.optim
                torch optmizer
                If not provided, optmizer is expected to be `model.optimizer`
            loss_func: nn.Module
                to1ch loss function
                If not provided, loss function is expected to be `model.criterion`
        """
        super().__init__()
        self.forward = model.forward
        self.optimizer = kwargs.get("optimizer", model.optimizer)
        self.criterion = kwargs.get("loss_func", model.criterion)

    def __repr__(self):
        return "ERM"

    def update(self, dataloader, device: Optional[torch.device] = None):
        """
        Update the model's parameters based on the training instances provided
        Args:
            dataloader: torch DataLoader
            device: torch.device
                device to cast data for training

        Returns:
            train_loss: float
                training loss
            train_acc: float
                training accuracy
        """
        loss, train_acc = 0.0, 0.0
        device = self._get_device() if device is None else device
        for i, (images, target) in enumerate(dataloader):
            images, target = images.to(device), target.to(device)
            outputs = self.forward(images)

            loss += self.criterion(outputs, target).to(device)
            train_acc += utils.eval.acc_fn(outputs, target)

        loss /= len(dataloader)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        log = dict()
        log["train_loss"] = loss.item()
        log["train_acc"] = train_acc / len(dataloader)
        return log


class IRM(ERM):
    def __init__(
        self,
        model: nn.Module,
        irm_lambda: int = 1e2,
        activate_after: Optional[int] = 500,
        scalar: bool = False,
        **kwargs
    ):
        """
        Performs training according to the Invariant Risk Minimization (IRM) principle (Arjovski, 2019)

        Args:
            model: nn.Module
                model to be trained
            irm_lambda: int, float
                regularization factor
            activate_after: int
                number of iterations to apply the regularization factor
                If left None, regularization will be applied throughout all training
            optimizer: torch.optim
                torch optmizer
                If not provided, optmizer is expected to be `model.optimizer`
            loss_func: nn.Module
                torch loss function
                If not provided, loss function is expected to be `model.criterion`
        """
        super(IRM, self).__init__(model, **kwargs)
        self.irm_lambda = irm_lambda
        self.activate_after = activate_after
        self.register_buffer("update_count", torch.tensor([0]))
        self._irm_penalty = self._irm_penalty_scalar

        if not scalar:
            is_linear = [isinstance(x, nn.Linear) for x in list(model.modules())[::-1]]
            if True in is_linear:
                idx = is_linear.index(True) + 1
                self.classifier = list(model.modules())[-idx]
                self._irm_penalty = self._irm_penalty_weights

    def __repr__(self):
        return "IRM"

    def clear_update_count(self):
        self.update_count = torch.tensor([0])

    def _irm_penalty_weights(self, logits, y):
        loss = self.criterion(logits, y)
        grad = autograd.grad(
            loss, [self.classifier.weight], create_graph=True, retain_graph=True
        )[0]
        return torch.sum(grad**2)

    def _irm_penalty_scalar(self, logits, y):
        scale = torch.tensor(1.0).to(logits.device).requires_grad_()
        loss = self.criterion(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    @property
    def penalty_weight(self):
        if self.activate_after is None:
            return self.irm_lambda
        return self.irm_lambda if self.update_count >= self.activate_after else 1.0

    def update(self, dataloader, device: Optional[torch.device] = None):
        """
        Update the model's parameters based on the training instances provided
        Args:
            dataloader: torch DataLoader
            device: torch.device
                device to cast data for training
        Returns:
            train_loss: float
                training loss
            train_acc: float
                training accuracy
        """
        device = self._get_device() if device is None else device
        base_loss, base_penalty = 0.0, 0.0
        train_acc = 0.0

        for i, (images, target) in enumerate(dataloader):

            images, target = images.to(device), target.to(device)
            outputs = self.forward(images)

            loss = self.criterion(outputs, target)
            penalty = self._irm_penalty(outputs, target)
            base_loss += loss
            base_penalty += penalty
            train_acc += utils.eval.acc_fn(outputs, target)

        if self.update_count.item() == self.activate_after:
            self.optimizer = self.optimizer.__class__(
                self.forward.__self__.parameters(), **self.optimizer.defaults
            )

        base_loss /= len(dataloader)
        penalty /= len(dataloader)

        # In IRM the gradient step is performed in the cummulative loss
        total_loss = (base_loss + self.penalty_weight * penalty) / self.penalty_weight

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.update_count += 1

        log = dict()
        log["train_loss"] = base_loss.item()
        log["train_acc"] = train_acc / len(dataloader)
        log["train_penalty"] = penalty.item()

        return log


class GroupDRO(ERM):
    def __init__(self, model, eta=1e-2, **kwargs):
        super().__init__(model, **kwargs)
        self.eta = eta
        self.q = None

    def update(self, dataloader, device: Optional[torch.device] = None):
        """
        Update the model's parameters based on the training instances provided
        Args:
            dataloader: torch DataLoader
            device: torch.device
                device to cast data for training
        Returns:
            train_loss: float
                training loss
            train_acc: float
                training accuracy
        """
        device = self._get_device() if device is None else device
        losses = torch.zeros(len(dataloader)).to(device)
        train_acc = 0.0

        if self.q is None:
            self.q = torch.ones(len(dataloader))
        elif isinstance(self.q, torch.Tensor):
            self.q = self.q.clone().detach()
        else:
            self.q = torch.tensor(self.q)

        self.q = self.q.to(device)

        for i, (images, target) in enumerate(dataloader):
            images, target = images.to(device), target.to(device)
            outputs = self.forward(images)
            losses[i] = self.criterion(outputs, target)
            self.q[i] *= (self.eta * losses[i].data).exp()
            train_acc += utils.eval.acc_fn(outputs, target)

        self.q /= self.q.sum()
        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # to avoid any issues when deepcopying model
        self.q = self.q.cpu().detach().numpy()

        log = dict()
        log["train_loss"] = losses.mean().item()
        log["train_acc"] = train_acc / len(dataloader)
        log["weighted_loss"] = loss.item()

        return log

    def __repr__(self):
        return "GroupDRO"


class MMRex(ERM):
    def __init__(self, model, lambda_min=-1, **kwargs):
        super().__init__(model, **kwargs)
        self.lambda_min = lambda_min

    def update(self, dataloader, device: Optional[torch.device] = None):
        """
        Update the model's parameters based on the training instances provided
        Args:
            dataloader: torch DataLoader
            device: torch.device
                device to cast data for training
        Returns:
            train_loss: float
                training loss
            train_acc: float
                training accuracy
        """
        device = self._get_device() if device is None else device
        losses = torch.zeros(len(dataloader)).to(device)
        train_acc = 0.0

        for i, (images, target) in enumerate(dataloader):
            images, target = images.to(device), target.to(device)
            outputs = self.forward(images)
            losses[i] = self.criterion(outputs, target)
            train_acc += utils.eval.acc_fn(outputs, target)

        loss = self.lambda_min * torch.sum(losses)
        loss += (1 - len(losses) * self.lambda_min) * torch.max(losses)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        log = dict()
        log["train_loss"] = losses.mean().item()
        log["train_acc"] = train_acc / len(dataloader)
        log["weighted_loss"] = loss.item()

        return log

    def __repr__(self):
        return "MMRex"


class VRex(ERM):
    def __init__(
        self,
        model: nn.Module,
        vrex_lambda: int = 1e1,
        activate_after: Optional[int] = 500,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.vrex_lambda = vrex_lambda
        self.activate_after = activate_after
        self.register_buffer("update_count", torch.tensor([0]))

    def __repr__(self):
        return "VRex"

    @property
    def penalty_weight(self):
        if self.activate_after is None:
            return self.vrex_lambda
        return self.vrex_lambda if self.update_count >= self.activate_after else 1.0

    def clear_update_count(self):
        self.update_count = torch.tensor([0])

    def update(self, dataloader, device: Optional[torch.device] = None):
        """
        Update the model's parameters based on the training instances provided
        Args:
            dataloader: torch DataLoader
            device: torch.device
                device to cast data for training
        Returns:
            train_loss: float
                training loss
            train_acc: float
                training accuracy
        """
        device = self._get_device() if device is None else device
        losses = torch.zeros(len(dataloader)).to(device)
        train_acc = 0.0

        for i, (images, target) in enumerate(dataloader):
            images, target = images.to(device), target.to(device)
            outputs = self.forward(images)
            losses[i] = self.criterion(outputs, target)
            train_acc += utils.eval.acc_fn(outputs, target)

        if self.update_count.item() == self.activate_after:
            self.optimizer = self.optimizer.__class__(
                self.forward.__self__.parameters(), **self.optimizer.defaults
            )

        loss = losses.mean()
        penalty = ((losses - loss) ** 2).mean()
        loss += self.penalty_weight * penalty

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1

        log = dict()
        log["train_loss"] = losses.mean().item()
        log["train_acc"] = train_acc / len(dataloader)
        log["train_penalty"] = penalty.item()

        return log
