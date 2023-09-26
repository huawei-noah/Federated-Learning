from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from flsuite.data.utils import CustomDataset


def accuracy(
    model: "nn.Module",
    data: Union[DataLoader, TensorDataset, list],
    device: Optional[torch.device] = None,
) -> float:
    """
    Calculate model accuracy
    Args:
        model: torch model
        data: DataLoader, TensorDataset, list, tuple
            data to be evaluated
        device: torch.device
            device to cast data
    Returns:
        acc
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model_device = next(model.parameters()).device
        if device != model_device:
            model.to(device)

    if isinstance(data, DataLoader):
        pass
    elif isinstance(data, TensorDataset):
        data = ((data.tensors),)
    elif isinstance(data, CustomDataset):
        data = ((data.X, data.y),)
    elif isinstance(data, (list, tuple)) and not isinstance(data[0], (list, tuple)):
        data = (data,)

    samples, acc = 0, 0.0
    with torch.no_grad():
        for images, target in data:
            images, target = images.to(device), target.to(device)
            outputs = model(images)
            acc += acc_fn(outputs, target) * len(target)
            samples += len(target)
    return acc / samples


def acc_fn(outputs: torch.tensor, targets: torch.tensor):
    correct = (outputs.argmax(dim=1) == targets).sum().item()
    return correct / targets.numel()


def loss(
    model: "nn.Module",
    data: Union[DataLoader, TensorDataset, list],
    loss_fn: nn.Module,
    device: Optional[torch.device] = None,
) -> float:
    """
    Calculate loss over given data and model
    Args:
        model: torch model
        data: DataLoader, TensorDataset, list, tuple
            data to be evaluated
        loss_fn: nn.Module
            loss function
        device: torch.device
            device to cast data
    Returns:
        loss
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model_device = next(model.parameters()).device
        if device != model_device:
            model.to(device)

    if isinstance(data, DataLoader):
        pass
    elif isinstance(data, TensorDataset):
        data = ((data.tensors),)
    elif isinstance(data, CustomDataset):
        data = ((data.X, data.y),)
    elif isinstance(data, (list, tuple)) and not isinstance(data[0], (list, tuple)):
        data = (data,)

    samples, cum_loss = 0, 0.0
    with torch.no_grad():
        for images, target in data:
            images, target = images.to(device), target.to(device)
            outputs = model(images)
            cum_loss += loss_fn(outputs, target).item() * len(target)
            samples += len(target)
    return cum_loss / samples


def accuracies(models, loaders, device):
    if not isinstance(models, (list, tuple)):
        models = [models] * len(loaders)
    return [accuracy(models[i], loaders[i], device) for i in range(len(loaders))]


def losses(models, loaders, loss_fn, device):
    if not isinstance(models, (list, tuple)):
        models = [models] * len(loaders)
    return [loss(models[i], loaders[i], loss_fn, device) for i in range(len(models))]
