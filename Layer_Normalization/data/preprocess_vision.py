'''
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the BSD 3.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3.0 License for more details.
'''

# generate data split for each client for following training

import argparse
import pickle
import os
import math
import sys

from PIL import Image
import numpy as np
from tqdm import trange, tqdm
import torchvision
from torchvision.datasets import EMNIST, MNIST, CIFAR10, CIFAR100
from torchvision.transforms import ToTensor


def iid(dataset, num_clients):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_clients:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_clients)
    dict_clients, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    return dict_clients


def generate_part(num_classes=10, num_clients=10, class_per_client=1):
    # the following code has a probability to fail, just repeat if it fails
    time_per_class = num_clients * class_per_client // num_classes
    if time_per_class != int(time_per_class):
        raise NotImplementedError # There will be residues from the dataset
    available = [time_per_class] * num_classes
    available_class = [i for i in range(num_classes)]
    client = {}
    for i in range(num_clients):
        client[i] = []
    for i in range(num_clients):
        # chose from available classes
        chosens = list(np.random.choice(available_class, class_per_client, replace=False))
        print(f'client {i}: ', 'choose: ', chosens)
        for chosen in chosens:
            client[i] += [chosen]
            available[chosen] -= 1
            if available[chosen] == 0:   # not available anymore
                available_class.remove(chosen)
    print('successful!')
    return client


def class_per_client_train(dataset, num_classes=10, num_clients=10, class_per_client=1):
    labeled = {}
    for i in range(num_classes):
        labeled[i] = []
        

    for images, label in dataset:
        try:
            labeled[label].append((images, label))
        except KeyError:
            print('key does not exists')

    client = generate_part(num_classes, num_clients, class_per_client)
    client_data = {}
    # the split of chunks for each class
    split = num_clients * class_per_client // num_classes
    # the remaining number of chunks for each class
    remaining_labels = np.ones(num_classes, dtype=int) * split 
    for i in range(num_clients):
        client_data[i] = []
    for i in range(num_clients):
        for chosen_label in client[i]:
            chunk = remaining_labels[chosen_label]
            total_samples = len(labeled[chosen_label])
            step = total_samples // split
            client_data[i] += labeled[chosen_label][(chunk - 1) * step : chunk * step]
            remaining_labels[chosen_label] -= 1
    return client_data


# function for using seperate test dataset using the traindata
def class_per_client_test(dataset, train_data, num_classes=10, num_clients=10, class_per_client=1):
    labeled = {}
    for i in range(num_classes):
        labeled[i] = []

    for images, label in dataset:
        labeled[label].append((images, label))
        
    client_train_labels = {}
    for i in range(num_clients):
        client_train_labels[i] = []
    for i in range(num_clients):
        for _, label in train_data[i]:
            if label not in client_train_labels[i]:
                client_train_labels[i].append(label)

    # the split of chunks for each class
    split = num_clients * class_per_client // num_classes
    # the remaining number of chunks for each class
    remaining_labels = np.ones(num_classes, dtype=int) * split 

    client_data_test = {}
    for i in range(num_clients):
        client_data_test[i] = []
    for i in range(num_clients):
        for chosen_label in client_train_labels[i]:
            chunk = remaining_labels[chosen_label]
            total_samples = len(labeled[chosen_label])
            step = total_samples // split
            client_data_test[i] += labeled[chosen_label][(chunk - 1) * step : chunk * step]
            remaining_labels[chosen_label] -= 1
    return client_data_test


# function for train-test split
def train_test_split(sample_set, frac=0.8):   # split a sample set into train-test
    total = len(sample_set)
    train_num = int(total * frac)
    if train_num < 1:
        train_set = None
    else:
        train_set = sample_set[:train_num]
    test_set = sample_set[train_num:]
    return train_set, test_set


def quantity_from_dirichlet(beta=3, num_clients=10, num_samples=60000):
    unrounded = np.random.dirichlet([beta] * num_clients) * num_samples
    client_to_ceil = np.random.choice(num_clients)
    rounded = np.ones(num_clients, dtype=int)
    for i in range(num_clients):
        if i is not client_to_ceil:
            rounded[i] = int(unrounded[i])
    rounded[client_to_ceil] = num_samples - (sum(rounded) - 1)
    return rounded


def quantity_split(dataset, beta=0.5, num_clients=12):
    in_data, out_data = [], []
    quantities = quantity_from_dirichlet(beta=beta, num_clients=num_clients, num_samples=len(dataset))
    total_ = 0
    for i in range(num_clients):
        client_data = []
        print(f'client {i}')
        for j in trange(quantities[i]):
            num = total_ + j
            client_data.append((dataset[num][0], dataset[num][1]))

        train_set, test_set = train_test_split(client_data, frac=args.frac)
        if quantities[i] == 0:
            train_set, test_set = [], []
        if not train_set:
            train_set = []
        if not test_set:
            test_set = []
        in_data.append(train_set)
        out_data.append(test_set)
        total_ += quantities[i]
    return in_data, out_data


# funcation for using seperate test dataset using the traindata
def quantity_split_seperate_testset(dataset, dataset_test, ratio, beta=0.5, num_clients=12):
    in_data, out_data = [], []
    quantities = quantity_from_dirichlet(beta=beta, num_clients=num_clients, num_samples=len(dataset))

    # heuristic method to make every item in the list dividable by ratio
    ratio_rev = 1/ratio
    assert int(ratio_rev) == ratio_rev
    assert sum(quantities) % ratio_rev == 0 # Check that that the total is dividable by ratio

    sum_quant = sum(quantities)
    sur_plus = quantities[0] % ratio_rev
    quantities[0] -= sur_plus
    for idx, item in enumerate(quantities[1:]):
        if item % ratio_rev == 0:
            continue
        elif sur_plus >= (ratio_rev - item % ratio_rev):
            quantities[idx + 1] += ratio_rev - item % ratio_rev
            sur_plus -= ratio_rev - item % ratio_rev
        else:
            quantities[idx + 1] -=  item % ratio_rev
            sur_plus += item % ratio_rev
    assert sum_quant == sum(quantities)

    total_ = 0
    total_test = 0
    for i in range(num_clients):
        client_data = []
        print(f'client {i}')
        for j in trange(quantities[i]):
            num = total_ + j
            client_data.append((dataset[num][0], dataset[num][1]))

        train_set = client_data
        if quantities[i] == 0:
            train_set = []
        if not train_set:
            train_set = []

        client_data_test = []
        test_quant = math.floor(quantities[i] * ratio)
        for j in trange(test_quant):
            num = total_test + j
            client_data_test.append((dataset_test[num][0], dataset_test[num][1]))

        test_set = client_data_test
        if test_quant == 0:
            test_set = []
        if not train_set:
            test_set = []

        in_data.append(train_set)
        out_data.append(test_set)
        total_ += quantities[i]
        total_test += test_quant

    return in_data, out_data


def shuffle(list_):
    num = len(list_)
    a = np.array(range(num))
    np.random.shuffle(a)
    return [list_[i] for i in a]


def main():
    
    root = '.'

    parser = argparse.ArgumentParser(description='data split')
    parser.add_argument('--data_dir', type=str, default=root)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--frac', type=float, default=0.5)
    parser.add_argument('--use_only_training_dataset', default=False, action='store_true')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--iid_type', type=str, default='iid', help='[iid, quantity shift, label\
                    shift, sort_and_part, class_part]')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--class_per_client', type=int, default=1)
    parser.add_argument('--dirichlet', type=float, default=0.1, help='parameter for Dirichlet allocation')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.iid_type == 'label_shift':
        if args.use_only_training_dataset:
            args.output_dir = os.path.join(root, args.dataset, args.output_dir \
                                    + f'{args.num_clients}_clients_dirichlet_{args.dirichlet}')
        else:
            args.output_dir = os.path.join(root, args.dataset, args.output_dir \
                                    + f'{args.num_clients}_clients_dirichlet_{args.dirichlet}_full')
    elif args.iid_type == 'sort_and_part':
        args.output_dir = os.path.join(root, args.dataset, f'{args.num_clients}_clients_sort_and_part')
    elif args.iid_type == 'class_part':
        if args.use_only_training_dataset:
            args.output_dir = os.path.join(root, args.dataset, \
                                       f'{args.num_clients}_clients_class_per_client_{args.class_per_client}')
        else:
            args.output_dir = os.path.join(root, args.dataset, \
                                       f'{args.num_clients}_clients_class_per_client_{args.class_per_client}_Full')
    os.makedirs(args.output_dir, exist_ok=True)
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # initialize the right dataset
    
    root = os.path.join(args.data_dir, args.dataset)
    
    if args.dataset == 'MNIST':
        dataset = MNIST(root=root, download=True, transform=ToTensor())

    elif args.dataset == 'CIFAR10':
        print('downloading CIFAR-10')
        dataset = CIFAR10(root=root, download=True, transform=ToTensor())
        if args.use_only_training_dataset == False:
            print('Using all the 50000 images as training data')
            dataset_testset = CIFAR10(root=root, download=True, train=False, transform=ToTensor())

    elif args.dataset == 'CIFAR100':
        dataset = CIFAR100(root=root, download=True,transform=ToTensor())
        if args.use_only_training_dataset == False:
            print('Using all the 50000 images as training data')
            dataset_testset = CIFAR100(root=root, download=True, train=False, transform=ToTensor())
    
    elif args.dataset == 'PACS':
        print(os.curdir)
        with open(os.path.join(args.data_dir,'train.pkl') , 'rb') as outp:
            dataset = pickle.load(outp)
        if args.use_only_training_dataset == False:
            with open(os.path.join(args.data_dir,'test.pkl'), 'rb') as outp:
                dataset_testset = pickle.load(outp)
    elif args.dataset == 'EMNIST':
        dataset = EMNIST(root=root,split='mnist',download=True,transform=ToTensor())
    else:
        raise NotImplementedError



    np.random.seed(args.seed)
    print('len: ', len(dataset))

    # in_data and out_data are lists, containing the training and testing samples for each client respectively
    # each element, called client_data, is the data for each client
    # each client_data is a list
    # each element in client_data is a tuple
    # each tuple is (image, label)
    # image is a torch tensor, and label is an integer

    total = len(dataset)
    sample_per_client = total // args.num_clients

    in_data, out_data = [], []
    if args.iid_type == 'iid':
        dict_ = iid(dataset, args.num_clients)
        for i in range(args.num_clients):
            client_data = []
            for j in tqdm(dict_[i]):
                client_data.append((dataset[j][0], dataset[j][1]))

            train_set, test_set = train_test_split(client_data, frac=args.frac)
            in_data.append(train_set)
            out_data.append(test_set)

    elif args.iid_type == 'quantity_shift':   # quantity shift
        quantities = quantity_from_dirichlet(beta=args.dirichlet, num_clients=args.num_clients, \
                                             num_samples=total)
        print('quantity list: ', quantities)
        total_ = 0
        for i in range(args.num_clients):
            client_data = []
            print(f'client {i}')
            for j in trange(quantities[i]):
                num = total_ + j
                client_data.append((dataset[num][0], dataset[num][1]))

            client_data = shuffle_list(client_data)
            train_set, test_set = train_test_split(client_data, frac=args.frac)
            in_data.append(train_set)
            out_data.append(test_set)
            total_ += quantities[i]

    elif args.iid_type == 'label_shift':   # label shift
        num_classes = args.num_classes
        labeled = {}
        for i in range(num_classes):
            labeled[i] = []
            
        for images, label in dataset:
            labeled[label].append((images, label))
        
        in_datas, out_datas = {}, {}    
        for i in range(args.num_clients):
            in_datas[i], out_datas[i] = [], []

        if args.use_only_training_dataset == False:
            labeled_test = {}
            for i in range(num_classes):
                labeled_test[i] = []

            for images, label in dataset_testset:
                labeled_test[label].append((images, label))
            
        for label in range(num_classes):
            print('class: ', label)
            if args.use_only_training_dataset == False:
                ratio = len(labeled_test[label])/len(labeled[label])
                in_data, out_data = quantity_split_seperate_testset(labeled[label], labeled_test[label], ratio, beta=args.dirichlet, \
                                                num_clients=args.num_clients) 
            else:
                in_data, out_data = quantity_split(labeled[label], beta=args.dirichlet, \
                                                num_clients=args.num_clients)    
            for i in range(args.num_clients):
                in_datas[i] += in_data[i]
                out_datas[i] += out_data[i]
        
        in_data, out_data = in_datas, out_datas
        
    elif args.iid_type == 'sort_and_part':
        num_classes = args.num_classes
        labeled = {}
        for i in range(num_classes):
            labeled[i] = []
            
        for images, label in dataset:
            labeled[label].append((images, label))
        
        sorted_list = []
        for i in range(num_classes):
            sorted_list += labeled[i]
        
        in_data, out_data = {}, {}
        step = len(sorted_list) // args.num_clients
        for i in range(args.num_clients):
            data = sorted_list[i * step : (i + 1) * step]
            data = shuffle(data)
            in_data[i], out_data[i] = train_test_split(data, frac=args.frac)  # drop last

    elif args.iid_type == 'class_part':
        in_data, out_data = {}, {}
        client_data = class_per_client_train(dataset, args.num_classes, args.num_clients, args.class_per_client)
        if args.use_only_training_dataset == False:
            client_data_test = class_per_client_test(dataset_testset, client_data, args.num_classes, args.num_clients, args.class_per_client, )
        for i in range(args.num_clients):
            data = client_data[i]
            data = shuffle(data)
            if args.use_only_training_dataset == False:
                in_data[i] = data
                out_data[i] = shuffle(client_data_test[i])
            else:
                in_data[i], out_data[i] = train_test_split(data, frac=args.frac)  # drop last 
    else:
        raise NotImplementedError

    # print stats
    for i in range(args.num_clients):
        print('client: ', i, 'in data: ', len(in_data[i]), 'out data: ', len(out_data[i]))
    
    # dump in_data and out_data 
    output_in = os.path.join(args.output_dir, 'in.pickle')
    output_out = os.path.join(args.output_dir, 'out.pickle')
    with open(output_in, 'wb') as output:
        pickle.dump(in_data, output)
    with open(output_out, 'wb') as output:
        pickle.dump(out_data, output)
    
    print('data split successful!')



if __name__ == '__main__':
    main()