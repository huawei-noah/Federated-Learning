'''
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the BSD 3.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3.0 License for more details.
'''

import argparse
from datetime import time
import os
import sys
import copy
import logging
import json
import time
from pathlib import Path

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch

from models.cnn import CNN_CIFAR
from models.resnet import resnet18
from algs.algorithms import FedAvg, FedAwS, FedProx, SCAFFOLD, FedBN, FedLC, FedYogi, FedDecorr, FedRS
from utils.aggregate import zero_model
from utils.data import get_labels, get_data_loaders
from utils.io import log_global_epoch
from utils.random import fix_randomness


def main():
    """Parse experiment's arugumants"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed', type=int, default=0)

    # Datasets, augmentation, logs, and device setup
    parser.add_argument('--num_clients', type=int, default=10, help='number of participating clients')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--logdir', type=str, default='./logs/CIFAR10', help='log directory')
    parser.add_argument('--datadir', type=str, default='./data/CIFAR10', help='data directory')
    parser.add_argument('--data_split', type=str, default='10_clients_class_per_client_1_Full', help='name of data distribution for logging')
    parser.add_argument('--no_aug', action='store_true', help='disables data augmentation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--log_interval', type=int, default=100, help='Sets logging intervals')
    parser.add_argument('--device', type=int, default=0, help='selects the gpu device')
    parser.add_argument('--num_workers', type=int, default=0)

    # Model configuration
    parser.add_argument('--model', type=str, default='CNN_CIFAR', choices=['CNN_CIFAR', 'ResNet18'], help='type of the model')
    parser.add_argument('--norm_methods_per_layer', type=str, default='[c1,c1,c1,c1]',
                            help='''for CNN model: uses astring list to detrimne the function per layer
                                    (s:mean shift, n: l2 norm, v:l2 norm with variance scale, c: multipy by the numbers after c
                                    g: group normalization, b: batch normalization, x:learnable parameter)
                                    for ResNet: vanilla, fn, group_norm, batch_norm, and group_norm_after for after activation''')
    parser.add_argument('--before_activation', action='store_true', default=False, help='executes the normalization after activation')
    parser.add_argument('--affine', action='store_true', default=False, help='enables learnable parameters for normalization methods')
    parser.add_argument('--return_feature', action='store_true', default=False, help='returns the features embedding')
    parser.add_argument('--with_bias', action='store_true', default=False, help='enables bias for weights of the networks')
    parser.add_argument('--num_groups', type=int, default=1, help='number of groups for group normalization')

    # Algorithms and benchmarks configurations
    parser.add_argument('--algorithm', type=str, default='FedAvg',
                        choices=['FedAvg', 'FedAwS', 'FedProx', 'SCAFFOLD', 'FedBN', 'FedLC', 'FedYogi', 'FedDecorr', 'FedRs'],
                        help='algorithms')
    parser.add_argument('--mu', type=float, default=1.0, help='for FedProx')
    parser.add_argument('--t', type=float, default=1.0, help='for FedLC')
    parser.add_argument('--alpha', type=float, default=1.0, help='for FedRs')

    # Experiment setup
    parser.add_argument('--num_local_steps', type=int, default=10, help='number of local steps')
    parser.add_argument('--num_rounds', type=int, default=10000, help='total number of rounds')
    
    # Loss, optimizers and learning rate
    parser.add_argument('--loss_func', type=str, default='ce', choices=['ce', 'pos'], help='loss function')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=0.01, help='minimum learning rate')
    parser.add_argument('--decay_type', type=str, default='const',
                        choices=['const', 'linear', 'cosine', 'reciprocal'], help='decay of learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='sgd momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')

 
    # Additional parameters
    parser.add_argument('--normalize_w', action='store_true', help='normalizing w')
    parser.add_argument('--clip_w', type=float, default=None, help='clipping w')
    parser.add_argument('--tau', type=float, default=1.0, help='the temperature of the loss')
    args = parser.parse_args()

    '''Fixing the seed and device'''
    fix_randomness(args.seed)
    device = torch.device(f'cuda:{args.device}')

    '''Setting dataset folders and logging '''
    datadir = os.path.join(args.datadir, args.data_split)
    logdir = os.path.join(args.logdir, args.data_split,
                          args.model, args.algorithm, f'{args.lr}',
                          args.norm_methods_per_layer,
                          f'num_group_{args.num_groups}',
                          f'affine_{args.affine}',
                          f'before_activation_{args.before_activation}',
                          args.name)
    # log files
    logfile = os.path.join(logdir, 'logs.txt')
    logfile_csv = os.path.join(logdir, 'logs.csv')
    model_file = os.path.join(logdir, 'model.pth')
    Path(logfile).parent.mkdir(exist_ok=True, parents=True)
    print(f'| logdir: {logdir}')
    logging.basicConfig(level=logging.DEBUG, filename=logfile,
                        filemode="a", format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # tensorboard logging
    train_writer = SummaryWriter(log_dir=os.path.join(logdir, 'train'))
    test_writer = SummaryWriter(log_dir=os.path.join(logdir, 'test'))

    # saving args
    logging.info(vars(args))
    test_writer.add_text('args', json.dumps(vars(args)))
    argsfile = os.path.join(logdir, 'args.json')
    with open(argsfile, 'w', encoding='UTF-8') as f:
        json.dump(vars(args), f)

    """Augmentation"""
    if args.no_aug:
        transform_train = None
    else:
        if 'tinyimagenet' in args.datadir.lower():
            crop = transforms.RandomCrop(60, padding=4)
        elif 'pacs' in args.datadir.lower():
            transform_train = None
            transform_train_batch = torch.nn.Sequential(
                transforms.RandomResizedCrop(224),
                )
        else:
            crop = transforms.RandomCrop(32, padding=4)
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                crop,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

    """Data Loaders and label distribution"""
    train_loaders, test_loaders, global_test_loaders, \
    weights_train, client_class, label_dist = get_data_loaders(
        os.path.join(datadir, 'in.pickle'),
        os.path.join(datadir, 'out.pickle'),
        args.batch_size, args.num_workers, args.num_clients,
        transform_train=transform_train, transform_test=None,
        num_classes=args.num_classes)

    """Model setup"""
    if args.model == 'CNN_CIFAR':
        model = CNN_CIFAR(norm_method=args.norm_methods_per_layer, device=device, num_classes=args.num_classes,
                          num_groups=args.num_groups, tau=args.tau, before_activation=args.before_activation,
                          affine_group_norm=args.affine, return_feature=args.return_feature,
                          bias=args.with_bias)
    elif args.model == 'ResNet18':
        model = resnet18(num_classes=args.num_classes, norm_method=args.norm_methods_per_layer,
                         affine=args.affine, group_number=args.num_groups, return_feature=args.return_feature)
    else:
        raise NotImplementedError
    
    # Creating models and optimizer for each client
    models = [copy.deepcopy(model) for _ in range(args.num_clients)]
    del model
    optimizers = [torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
                                    for model in models]

    """Loss function """
    if args.loss_func == 'ce':
        loss_func = torch.nn.CrossEntropyLoss()
    elif args.loss_func == 'pos':  # for FedAwS
        loss_func = pos_loss
    else:
        raise NotImplementedError

    """Algorithms setup"""
    if args.algorithm == 'FedAvg':
        alg = FedAvg(models, optimizers,
                     num_clients=args.num_clients,
                     num_local_steps=args.num_local_steps,
                     loss_func=loss_func,
                     )

    elif args.algorithm == 'FedAwS':  
        labels = get_labels(os.path.join(
            datadir, 'in.pickle'), args.num_clients)
        alg = FedAwS(models, optimizers,
                     num_clients=args.num_clients,
                     num_local_steps=args.num_local_steps,
                     loss_func=loss_func,
                     sp=args.sp, sp_lr=args.sp_lr)

    elif args.algorithm == 'FedProx':
        alg = FedProx(models, optimizers,
                      num_clients=args.num_clients,
                      num_local_steps=args.num_local_steps,
                      loss_func=loss_func, mu=args.mu)

    elif args.algorithm == 'SCAFFOLD':
        model = copy.deepcopy(models[0])
        controls = [zero_model(model).to(device)
                    for _ in range(args.num_clients)]
        server_control = zero_model(model).to(device)
        alg = SCAFFOLD(models, controls, server_control, optimizers,
                       num_clients=args.num_clients,
                       num_local_steps=args.num_local_steps,
                       loss_func=loss_func, lr=args.lr)

    elif args.algorithm == 'FedBN':
        alg = FedBN(models, optimizers,
                    num_clients=args.num_clients,
                    num_local_steps=args.num_local_steps,
                    loss_func=loss_func)

    elif args.algorithm == 'FedLC':
        label_margin={}
        for client in range(args.num_clients):
            label_margin[client] = args.t * torch.pow(label_dist[client], -1 / 4)
            print('label margins for client', client, torch.round(label_margin[client], decimals=2).tolist())
        alg = FedLC(models, optimizers,
                    num_clients=args.num_clients,
                    num_local_steps=args.num_local_steps,
                    loss_func=loss_func,
                    label_margin=label_margin)
    
    elif args.algorithm == 'FedRs':
        class_access = {}
        for client in range(args.num_clients):
            class_access[client] = torch.tensor([1.0 if label_dist[client][i] > 0 else args.alpha 
                                                for i in range(args.num_classes)])
            print('class_access for client', client, class_access[client])
        alg = FedRS(models, optimizers,
                    num_clients=args.num_clients,
                    num_local_steps=args.num_local_steps,
                    loss_func=loss_func,
                    class_access=class_access)

    elif args.algorithm == 'FedYogi':
            alg = FedYogi(models, optimizers,
                        num_clients=args.num_clients,
                        num_local_steps=args.num_local_steps,
                        loss_func=loss_func, device=device)

    elif args.algorithm == 'FedDecorr':
            alg = FedDecorr(models, optimizers,
                        num_clients=args.num_clients,
                        num_local_steps=args.num_local_steps,
                        loss_func=loss_func, device=device)
    else:
        raise NotImplementedError

    """Check points"""
    start_round = 0
    if os.path.exists(model_file):
        start_round = torch.load(model_file)['epoch']
        logging.info(
            f'| checkpoint exists: restarting from %s with %s rounds completed', model_file, start_round)
        for model in models:
            model.load_state_dict(torch.load(model_file, map_location='cpu')['state_dict'])
    else:
        logging.info('| starting training from scratch')
        with open(logfile_csv, 'w', encoding='UTF-8') as f:
            f.write('round,train_losses,train_accs,train_acc_mean,\
                test_losses,test_accs,test_acc_mean,w_norms\n')

    if start_round == args.num_rounds:
        logging.info('| all rounds elapsed. Exiting...')
        sys.exit()

    f = open(logfile_csv, 'a', encoding='UTF-8')
    

    """Main iterations"""
    timer = time.perf_counter() # Timer
    for t in range(start_round + 1, args.num_rounds + 1):
        """Local training"""
        alg.local_updates(train_loaders, device,
                            normalize_w=args.normalize_w,
                            clip_w=args.clip_w)

        """Aggregation"""
        alg.aggregate(weights=weights_train)

        """Log global test results and save checkpoints"""
        log_global_epoch(alg, args, train_loaders, test_loaders, loss_func,
                             device, train_writer, test_writer, f, t)

        if t % args.log_interval == 0:
            # Time 
            logging.info(
                f'| time for {args.log_interval} rounds: {round(time.perf_counter() - timer, 2)} s')
            timer = time.perf_counter()
            # Save model
            torch.save({'epoch': t, 'state_dict': alg.models[0].state_dict()}, model_file)
    f.close()

if __name__ == "__main__":
    main()
