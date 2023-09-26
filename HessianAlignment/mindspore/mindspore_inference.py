'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import numpy as np
from torchvision import datasets
import argparse
import random
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore import Tensor, set_seed


parser = argparse.ArgumentParser(description='Colored MNIST')
# select your algorithm
parser.add_argument('--label_flipping_prob', type=float, default=0.25)
parser.add_argument('--hidden_dim', type=int, default=390)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.00110794568)
parser.add_argument('--lr', type=float, default=0.0004898536566546834)
parser.add_argument('--penalty_anneal_iters', type=int, default=190)
parser.add_argument('--penalty_weight', type=float, default=91257.18613115903)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument("--method", default='curve', type=str, help="curve or hutch")
# experimental setup
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--seed', type=int, default=0, help='Seed for everything')
parser.add_argument("--gpu_idx", default='0', type=str, help="gpu_idx.")
flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

random.seed(flags.seed)
np.random.seed(flags.seed)
set_seed(flags.seed)

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()
    
    
# Build environments
def make_environment(images, labels, e, grayscale=False):
    
    def torch_bernoulli(p, size):
        return (F.rand(size) < p).float()
    
    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1
    
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.1, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = F.stack([images, images], axis=1)
    if not grayscale:
        images[Tensor([x for x in range(len(images))]), (1 - colors).long(), :, :] *= 0
    return {'images': (images.float() / 255.), 'labels': labels[:, None]}    


class MLP(nn.Cell):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Dense(2*14*14, 390)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Dense(390, 390)
        self.relu2 = nn.ReLU()
        self.classifier = nn.Dense(390, 1)

    def construct(self, X):
        out = F.reshape(X, (X.shape[0], 2*14*14))
        features = self.relu2(self.lin2(self.relu1(self.lin1(out))))
        logits = self.classifier(features)
        return features, logits


# Load checkpoint to Mindspore model
mlp = MLP()
if flags.methods == 'curve':
    mlp.lin1.weight.set_data(Tensor(np.load("checkpoints/lin1_w.npy")))
    mlp.lin1.bias.set_data(Tensor(np.load("checkpoints/lin1_b.npy")))
    mlp.lin2.weight.set_data(Tensor(np.load("checkpoints/lin2_w.npy")))
    mlp.lin2.bias.set_data(Tensor(np.load("checkpoints/lin2_b.npy")))
    mlp.classifier.weight.set_data(Tensor(np.load("checkpoints/clsfr_w.npy")))
    mlp.classifier.bias.set_data(Tensor(np.load("checkpoints/clsfr_b.npy")))
elif flags.methods == 'hutch':
    mlp.lin1.weight.set_data(Tensor(np.load("checkpoints/lin1_w_h.npy")))
    mlp.lin1.bias.set_data(Tensor(np.load("checkpoints/lin1_b_h.npy")))
    mlp.lin2.weight.set_data(Tensor(np.load("checkpoints/lin2_w_h.npy")))
    mlp.lin2.bias.set_data(Tensor(np.load("checkpoints/lin2_b_h.npy")))
    mlp.classifier.weight.set_data(Tensor(np.load("checkpoints/clsfr_w_h.npy")))
    mlp.classifier.bias.set_data(Tensor(np.load("checkpoints/clsfr_b_h.npy")))

final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples
    envs = [np.load(f"checkpoints/env-{i}.npy", allow_pickle=True).item() for i in range(4)]
    for env in envs:
        env['images'] = Tensor(env['images'].cpu().numpy())
        env['labels'] = Tensor(env['labels'].cpu().numpy())

    for edx, env in enumerate(envs):
        features, logits = mlp(env['images'])
        env['acc'] = mean_accuracy(logits, env['labels'])

    train_acc = F.stack([envs[0]['acc'], envs[1]['acc']]).mean()
    test_acc = envs[2]['acc']

    final_train_accs.append(train_acc.numpy())
    final_test_accs.append(test_acc.numpy())

print('Final train acc (mean/std across restarts so far):')
print(np.mean(final_train_accs), np.std(final_train_accs))
print('Final test acc (mean/std across restarts so far):')
print(np.mean(final_test_accs), np.std(final_test_accs))
