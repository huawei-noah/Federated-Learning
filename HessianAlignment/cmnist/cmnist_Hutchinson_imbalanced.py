# This script was first copied from https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/main.py under the license
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''


import random
import argparse
import numpy as np
from collections import OrderedDict

import torch
from torchvision import datasets
from torch import nn, optim, autograd

from backpack import backpack, extend
from backpack.extensions import BatchGrad

parser = argparse.ArgumentParser(description='Colored MNIST')

# select your algorithm
parser.add_argument(
    '--algorithm',
    type=str,
    default="sad",
    choices=[
        ## Four main methods, for Table 2 in Section 6.A
        'erm',  # Empirical Risk Minimization
        'irm',  # Invariant Risk Minimization (https://arxiv.org/abs/1907.02893)
        'rex',  # Out-of-Distribution Generalization via Risk Extrapolation (https://icml.cc/virtual/2021/oral/9186)
        'fishr',  # Our proposed Fishr
        ## two Fishr variants, for Table 6 in Appendix C.2.4
        'fishr_offdiagonal',  # Fishr but on the full covariance rather than only the diagonal
        'fishr_notcentered',  # Fishr but without centering the gradient variances
        'sad', #sharpness aware domain generalization
    ]
)
# select whether you want to apply label flipping or not
# Set to 0 in Table 5 in Appendix C.2.3 and in the right half of Table 6 in Appendix C.2.4
parser.add_argument('--label_flipping_prob', type=float, default=0.25)

# Following hyperparameters are directly taken from from https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/reproduce_paper_results.sh
# They should not be modified except in case of a new proper hyperparameter search with an external validation dataset.
# Overall, we compare all approaches using the hyperparameters optimized for IRM.
parser.add_argument('--hidden_dim', type=int, default=390)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.00110794568)
parser.add_argument('--lr', type=float, default=0.0004898536566546834)
parser.add_argument('--penalty_anneal_iters', type=int, default=190)
parser.add_argument('--penalty_weight', type=float, default=91257.18613115903)
parser.add_argument('--steps', type=int, default=501)
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
torch.manual_seed(flags.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
os.environ['NVIDIA_VISIBLE_DEVICES'] = flags.gpu_idx
os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu_idx


final_train_accs = []
final_test_accs = []
final_graytest_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments


    def make_environment(images, labels, e, grayscale=False):

        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()

        def torch_xor(a, b):
            return (a - b).abs()  # Assumes both inputs are either 0 or 1

        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(flags.label_flipping_prob, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        if not grayscale:
            images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
        return {'images': (images.float() / 255.).cuda(), 'labels': labels[:, None].cuda()}

    def trh(T,S):
        return round(T*S)


    idx_0 = mnist_train[1] < 5
    idx_1 = mnist_train[1] >= 5
    P = 0.97


    print(trh(P,idx_0.shape[0]))
    print(trh(1-P,idx_1.shape[0]))

    Env0_X = mnist_train[0][idx_0][0:trh(P,idx_0.shape[0])]
    Env0_X = torch.cat([Env0_X,mnist_train[0][idx_1][0:trh(1-P,idx_1.shape[0])]],dim=0)

    Env0_Y = mnist_train[1][idx_0][0:trh(P,idx_0.shape[0])]
    Env0_Y =  torch.cat([Env0_Y,mnist_train[1][idx_1][0:trh(1-P,idx_1.shape[0])]],dim=0)



    Env1_X = mnist_train[0][idx_0][trh(P,idx_0.shape[0]):]
    Env1_X =  torch.cat([Env1_X,mnist_train[0][idx_1][trh(1-P,idx_1.shape[0]):]],dim=0)

    Env1_Y = mnist_train[1][idx_0][trh(P,idx_0.shape[0]):]
    Env1_Y = torch.cat([Env1_Y,mnist_train[1][idx_1][trh(1-P,idx_1.shape[0]):]],dim = 0)


    print(Env0_Y[Env0_Y>=5].shape[0]/Env0_Y.shape[0])
    print(Env0_Y[Env0_Y<5].shape[0]/Env0_Y.shape[0])


    # envs = [
    #     make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
    #     make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
    #     make_environment(mnist_val[0], mnist_val[1], 0.9),
    #     make_environment(mnist_val[0], mnist_val[1], 0.9, grayscale=True)
    # ]
    envs = [
    make_environment(Env0_X, Env0_Y, 0.2),
    make_environment(Env1_X,Env1_Y, 0.1),
    make_environment(mnist_val[0], mnist_val[1], 0.9),
    make_environment(mnist_val[0], mnist_val[1], 0.9, grayscale=True)
    ]

    # Define and instantiate the model


    class MLP(nn.Module):

        def __init__(self):
            super(MLP, self).__init__()
            if flags.grayscale_model:
                lin1 = nn.Linear(14 * 14, flags.hidden_dim)
            else:
                lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)

            self.classifier = extend(nn.Linear(flags.hidden_dim, 1))
            for lin in [lin1, lin2, self.classifier]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True))

        def forward(self, input):
            if flags.grayscale_model:
                out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
            else:
                out = input.view(input.shape[0], 2 * 14 * 14)
            features = self._main(out)
            logits = self.classifier(features)
            return features, logits

    mlp = MLP().cuda()

    # Define loss function helpers
    bce_extended = extend(nn.BCEWithLogitsLoss())

    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def compute_irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def compute_sad_penalty(logits, y):
        gradient_norm=[]
        numels=[]
        loss = bce_extended(logits , y)
        grads = autograd.grad(loss,  mlp.classifier.parameters(), create_graph=True,retain_graph=True)
        flatten_grads = []
        for grad in grads:
            flatten_grads.append(torch.flatten(grad))
            gradient_norm.append(torch.norm(grad, p=2))
            numels.append(torch.numel(grad))
        #gradient_loss = torch.norm(torch.stack(gradient_norm), p=2)
        flatten_grads = torch.cat(flatten_grads)
        gradient_loss =torch.norm(flatten_grads,p=2)
        return  gradient_loss , grads, flatten_grads

    def compute_grads_variance(features, labels, classifier):
        logits = classifier(features)
        loss = bce_extended(logits, labels)
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(classifier.parameters()), retain_graph=True, create_graph=True
            )

        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in classifier.named_parameters()
            ]
        )
        dict_grads_variance = {}
        for name, _grads in dict_grads.items():
            grads = _grads * labels.size(0)  # multiply by batch size
            env_mean = grads.mean(dim=0, keepdim=True)
            if flags.algorithm != "fishr_notcentered":
                grads = grads - env_mean
            if flags.algorithm == "fishr_offdiagonal":
                dict_grads_variance[name] = torch.einsum("na,nb->ab", grads,
                                                    grads) / (grads.size(0) * grads.size(1))
            else:
                dict_grads_variance[name] = (grads).pow(2).mean(dim=0)

        return dict_grads_variance

    def l2_between_grads_variance(cov_1, cov_2):
        assert len(cov_1) == len(cov_2)
        cov_1_values = [cov_1[key] for key in sorted(cov_1.keys())]
        cov_2_values = [cov_2[key] for key in sorted(cov_2.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in cov_1_values])) -
            torch.cat(tuple([t.view(-1) for t in cov_2_values]))
        ).pow(2).sum()

    # Train loop

    def _flatten_grad(grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def calc_hessian_diag(model, loss_grad, repeat=1000):
        diag = []
        gg = torch.cat([g.flatten() for g in loss_grad])
        for _ in range(repeat):
            z = 2*torch.randint_like(gg, high=2)-1
            loss = torch.dot(gg,z)
            Hz = autograd.grad(loss, model.classifier.parameters(), retain_graph=True, create_graph=True)
            Hz = torch.cat([torch.flatten(g) for g in Hz])
            diag.append(z*Hz)
        return sum(diag)/len(diag)

    def eval_hessian(loss_grad, model):
        cnt = 0
        for g in loss_grad:
            g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
            cnt = 1
        l = g_vector.size(0)
        hessian = torch.zeros(l, l)
        for idx in range(l):
            grad2rd = autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2
        return hessian.cpu().data.numpy()


    def pretty_print(*values):
        col_width = 13
        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)

        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))

    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print(
        'step', 'train nll', 'train acc', 'fishr penalty', 'rex penalty', 'irm penalty','sad penalty', 'test acc',
        "gray test acc"
    )
    grad_of_env = {}
    flatten_grad_of_env = {}

    grad_of_env_C0 = {}
    flatten_grad_of_env_C0 = {}

    grad_of_env_C1 = {}
    flatten_grad_of_env_C1 = {}
    hessain_dists = []
    test_accuracies = []
    grey_test_accuracies = []
    sad_loss = []
    for step in range(flags.steps):
        for edx, env in enumerate(envs):
            features, logits = mlp(env['images'])
            #print('features',features.size())
            #env['labels'] = torch.tensor(env['labels'], dtype=torch.int64)
            idx_0 = env['labels']==0
            idx_1 = env['labels']==1
            
            labels_0 = env['labels'][idx_0]
            labels_1 = env['labels'][idx_1]
            #print('labels_0',labels_0.size())

            features_0 = features[idx_0.squeeze()]
            features_1 = features[idx_1.squeeze()]

            #print('features_0',features_0.size())

            #print('logits',logits.size())
            logits_0 = logits[idx_0.squeeze(),0]
            logits_1 = logits[idx_1.squeeze(),0]

            #print('logits_0',logits_0.size())
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['irm'] = compute_irm_penalty(logits, env['labels'])
            env['sad'] , grad_of_env[edx], flatten_grad_of_env[edx] = compute_sad_penalty(logits, env['labels'])

            env['nll_C0'] = mean_nll(logits_0, labels_0)
            env['nll_C1'] = mean_nll(logits_1, logits_1)
            env['sad_C0'] , grad_of_env_C0[edx], flatten_grad_of_env_C0[edx] = compute_sad_penalty(logits_0, labels_0)
            env['sad_C1'] , grad_of_env_C1[edx], flatten_grad_of_env_C1[edx] = compute_sad_penalty(logits_1, labels_1)

            if edx in [0, 1]:
                # True when the dataset is in training
                optimizer.zero_grad()
                env["grads_variance"] = compute_grads_variance(features, env['labels'], mlp.classifier)

        train_nll_C0 = torch.stack([envs[0]['nll_C0'], envs[1]['nll_C0']]).mean() 
        train_nll_C1 = torch.stack([envs[0]['nll_C1'], envs[1]['nll_C1']]).mean() 

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean() 
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)
        
        #new_train_nll =  torch.stack([torch.mul(envs[0]['sad'],envs[0]['nll']), torch.mul(envs[1]['sad'],envs[1]['nll'])]).mean()
        
        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm

        irm_penalty = torch.stack([envs[0]['irm'], envs[1]['irm']]).mean() #- torch.mul(envs[0]['irm'], envs[1]['irm'])
        
        
        
    
        sad_penalty = 0
        # for g in range(len(grad_of_env[0])):
        #     #sad_penalty=sad_penalty+(torch.flatten(grad_of_env[0][g])-torch.flatten(grad_of_env[1][g])).pow(2).sum().sqrt()
        #     sad_penalty=sad_penalty+torch.dot(torch.flatten(grad_of_env[0][g]),torch.flatten(grad_of_env[1][g]))
        # sad_penalty=-1*sad_penalty    



        sad_penalty_list = []
        # grads_0 = autograd.grad(envs[0]['sad'], mlp.classifier.parameters(), create_graph=True,retain_graph=True)
        # grads_1 = autograd.grad(envs[1]['sad'], mlp.classifier.parameters(), create_graph=True,retain_graph=True)
        
        # grads_0_C_0 = autograd.grad(envs[0]['sad_C0'], mlp.classifier.parameters(), create_graph=True,retain_graph=True)
        # grads_1_C_0 = autograd.grad(envs[1]['sad_C0'], mlp.classifier.parameters(), create_graph=True,retain_graph=True)

        # grads_0_C_1 = autograd.grad(envs[0]['sad_C1'], mlp.classifier.parameters(), create_graph=True,retain_graph=True)
        # grads_1_C_1 = autograd.grad(envs[1]['sad_C1'], mlp.classifier.parameters(), create_graph=True,retain_graph=True)
        

        # flatten_grad_of_norm_0 = []
        # flatten_grad_of_norm_1 = []
        # flatten_grad_of_norm_0_C_0 = []
        # flatten_grad_of_norm_0_C_1 = []
        # flatten_grad_of_norm_1_C_0 = []
        # flatten_grad_of_norm_1_C_1 = []
        # for g in range(len(grads_0)):
        #     flatten_grad_of_norm_0.append(torch.flatten(grads_0[g]))
        #     flatten_grad_of_norm_1.append(torch.flatten(grads_1[g]))
            
        #     flatten_grad_of_norm_0_C_0.append(torch.flatten(grads_0_C_0[g]))
        #     flatten_grad_of_norm_0_C_1.append(torch.flatten(grads_0_C_1[g]))
        #     flatten_grad_of_norm_1_C_0.append(torch.flatten(grads_1_C_0[g]))
        #     flatten_grad_of_norm_1_C_1.append(torch.flatten(grads_1_C_1[g]))

        # flatten_grad_of_norm_0 = torch.cat(flatten_grad_of_norm_0)
        # flatten_grad_of_norm_1 = torch.cat(flatten_grad_of_norm_1)

        # flatten_grad_of_norm_0_C_0 = torch.cat(flatten_grad_of_norm_0_C_0)
        # flatten_grad_of_norm_0_C_1 = torch.cat(flatten_grad_of_norm_0_C_1)
        # flatten_grad_of_norm_1_C_0 = torch.cat(flatten_grad_of_norm_1_C_0)
        # flatten_grad_of_norm_1_C_1 = torch.cat(flatten_grad_of_norm_1_C_1)
        
        # Hg_0 = torch.mul(envs[0]['sad'],flatten_grad_of_norm_0)
        # Hg_1 = torch.mul(envs[1]['sad'],flatten_grad_of_norm_1)

        mean_grads = autograd.grad(train_nll, mlp.classifier.parameters(), create_graph=True,retain_graph=True)
       
        flatten_mean_grads = _flatten_grad(mean_grads)

        mean_Hg = calc_hessian_diag(mlp,mean_grads,repeat=200)

        Hg_0 = calc_hessian_diag(mlp,grad_of_env[0],repeat=200)
        Hg_1 = calc_hessian_diag(mlp,grad_of_env[1],repeat=200) 

        
        Hessian_1 = eval_hessian(grad_of_env[1],  mlp.classifier)
        Hessian_0 = eval_hessian(grad_of_env[0],  mlp.classifier)
        
        hessain_dists.append(np.linalg.norm(Hessian_0-Hessian_1))
        
    
        #sad_penalty = (Hg_0-Hg_1).pow(2).sum() + (flatten_grad_of_env[0] - flatten_grad_of_env[1]).pow(2).sum()
        sad_penalty = (flatten_grad_of_env[0] - flatten_mean_grads).pow(2).sum() + (flatten_grad_of_env[1] - flatten_mean_grads).pow(2).sum() +\
                         (Hg_0-mean_Hg).pow(2).sum() + (Hg_1-mean_Hg).pow(2).sum()
        #sad_penalty = sad_penalty/2
        #sad_penalty = (Hg_0_C_0-Hg_1_C_0).pow(2).sum() + (Hg_0_C_1-Hg_1_C_1).pow(2).sum()
        sad_loss.append(sad_penalty.detach().cpu().numpy())          

                    




        #temp = nn.functional.softmax(torch.stack([envs[0]['sad'] , envs[1]['sad']]),dim=0)
        # for g in range(len(grads_0)):

        #     Hg_0 = torch.mul(torch.flatten(grad_of_env[0][g]).pow(2).sum().sqrt(),torch.flatten(grads_0[g]))
        #     Hg_1 = torch.mul(torch.flatten(grad_of_env[1][g]).pow(2).sum().sqrt(),torch.flatten(grads_1[g]))
        #     sad_penalty_list.append((Hg_0-Hg_1).pow(2).sum() + \
        #         (torch.flatten(grad_of_env[0][g])-torch.flatten(grad_of_env[1][g])).pow(2).sum())


        #     sad_penalty_list.append((torch.flatten(grad_of_env[0][g]).pow(2).sum().sqrt()-torch.flatten(grad_of_env[1][g]).pow(2).sum().sqrt()).pow(2))


    

            #sad_penalty = sad_penalty + (torch.flatten(grad_of_env[0][g]).pow(2).sum().sqrt() - torch.flatten(grad_of_env[1][g]).pow(2).sum().sqrt()).pow(2)

            
        #sad_penalty = -1 * sad_penalty
        #sad_penalty = torch.norm(torch.stack(sad_penalty_list), p=2) 
            
            # sad_penalty=sad_penalty+torch.dot(torch.mul(torch.flatten(grad_of_env[0][g]).pow(2).sum().sqrt(),torch.flatten(grads_0[g])),torch.mul(torch.flatten(grad_of_env[1][g]).pow(2).sum().sqrt(),torch.flatten(grads_1[g]))) + \
            #     torch.dot(torch.flatten(grad_of_env[0][g]),torch.flatten(grad_of_env[1][g]))

            #sad_penalty=(torch.flatten(grad_of_env[0][g])-torch.flatten(grad_of_env[1][g])).pow(2).sum()
            
        #     sad_penalty=sad_penalty+torch.dot(torch.flatten(grad_of_env[0][g]),torch.flatten(grad_of_env[1][g]))
        #sad_penalty=-1*sad_penalty  
        
        #temp = nn.functional.softmax(torch.stack([envs[0]['nll'], envs[1]['nll']]))
        #sad_penalty = ((envs[0]['sad'] - envs[1]['sad'])**2).mean() #+ (envs[0]['nll'].mean() - envs[1]['nll'].mean())**2
        #sad_penalty = ((torch.stack([envs[0]['sad'] , envs[1]['sad']])-(torch.stack([envs[0]['sad'] ,envs[1]['sad']]).mean()))**2).mean()
        
        #temp = nn.functional.softmax(torch.stack([envs[0]['sad'] , envs[1]['sad']]),dim=0)
        #sad_penalty = (envs[0]['sad'] - envs[1]['sad'])**2
        #print(envs[0]['sad'],envs[1]['sad'])


        rex_penalty = (envs[0]['nll'].mean() - envs[1]['nll'].mean())**2
        #rex_penalty = ((torch.stack([envs[0]['nll'] , envs[1]['nll']])-train_nll)**2).mean()

        # Compute the variance averaged over the two training domains
        dict_grads_variance_averaged = OrderedDict(
            [
                (
                    name,
                    torch.stack([envs[0]["grads_variance"][name], envs[1]["grads_variance"][name]],
                                dim=0).mean(dim=0)
                ) for name in envs[0]["grads_variance"]
            ]
        )
        fishr_penalty = (
            l2_between_grads_variance(envs[0]["grads_variance"], dict_grads_variance_averaged) +
            l2_between_grads_variance(envs[1]["grads_variance"], dict_grads_variance_averaged)
        )

        # apply the selected regularization
        if flags.algorithm == "erm":
            pass
        else:
            if flags.algorithm.startswith("fishr"):
                train_penalty = fishr_penalty
            elif flags.algorithm == "rex":
                train_penalty = rex_penalty
            elif flags.algorithm == "irm":
                train_penalty = irm_penalty
            elif flags.algorithm == "sad":
                train_penalty = sad_penalty
            else:
                raise ValueError(flags.algorithm)
            penalty_weight = (flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0)
            #penalty_weight=100000000000000
            loss += penalty_weight * train_penalty #- torch.mul(envs[0]['sad'],envs[1]['sad']) )
            if penalty_weight > 1.0:
                #Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_acc = envs[2]['acc']
        grayscale_test_acc = envs[3]['acc']
        test_accuracies.append(test_acc.detach().cpu().numpy())
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                fishr_penalty.detach().cpu().numpy(),
                rex_penalty.detach().cpu().numpy(),
                irm_penalty.detach().cpu().numpy(),
                sad_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy(),
                grayscale_test_acc.detach().cpu().numpy(),
            )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    final_graytest_accs.append(grayscale_test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    print('Final gray test acc (mean/std across restarts so far):')
    print(np.mean(final_graytest_accs), np.std(final_graytest_accs))


np.save('hessain_dists_Hutch_var_imbalanced.npy', hessain_dists)

sad_loss = np.array(sad_loss)
np.save('sad_loss_Hutch_var_imbalanced.npy', sad_loss)



test_accuracies = np.array(test_accuracies)
np.save('test_acc_Hutch_var_imbalanced.npy', test_accuracies)