'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

# generate data split for each client for following training

import torchvision
from torchvision.datasets import EMNIST, MNIST, CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
import argparse
import pickle
import os
from tqdm import trange, tqdm
import numpy as np
from PIL import Image


def build_dataset(classes, root='CINIC10/raw_data', suffix=None):
    class_keys = list(classes.keys())
    dataset = []
    for key in class_keys:
        if suffix is not None:
            folder = os.path.join(root, key, suffix)
        else:
            folder = os.path.join(root, key)
        for image_file in os.listdir(folder):
            img = Image.open(os.path.join(folder, image_file))
            img_tensor = ToTensor()(img)
            if img_tensor.shape[0] is not 3:   # skip black/white images
                continue
            dataset.append((img_tensor, classes[key]))
    return dataset

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


# function for train-test split
def train_test_split(sample_set, frac=0.5):   # split a sample set into train-test
    total = len(sample_set)
    if total == 0:
        return None, None
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
    rounded = np.ones(num_clients,dtype=int)
    for i in range(num_clients):
        sum_ = 0.
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

        #client_data = shuffle_list(client_data)
        train_set, test_set = train_test_split(client_data, frac=args.frac)
        if quantities[i] == 0:
            train_set, test_set = [], []
        if train_set == None:
            train_set = []
        if test_set == None:
            test_set = []
        in_data.append(train_set)
        out_data.append(test_set)
        total_ += quantities[i]
    return in_data, out_data


def shuffle_list(list_):
    num = len(list_)
    a = np.array(range(num))
    np.random.shuffle(a)
    return [list_[i] for i in a]


if __name__ == '__main__':
    
    root = '.'

    # options for data split

    parser = argparse.ArgumentParser(description='data split')
    parser.add_argument('--data_dir', type=str, default=root)
    parser.add_argument('--dataset', type=str, default='CIFAR100')
    parser.add_argument('--frac', type=float, default=0.5)
    parser.add_argument('--dirichlet', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='split_CIFAR100')
    parser.add_argument('--iid', type=int, default=2, help='0 means iid')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=100)
    args = parser.parse_args()
    args.output_dir = os.path.join(root, args.dataset, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(args)

    # initialize the right dataset

    if args.dataset == 'MNIST':
        dataset = MNIST(root=root, download=True,transform=ToTensor())
    elif args.dataset == 'CIFAR10':
        dataset = CIFAR10(root=root, download=True,transform=ToTensor())
    elif args.dataset == 'CIFAR100':
        dataset = CIFAR100(root=root, download=True,transform=ToTensor())
    elif args.dataset == 'TinyImageNet':
        root = 'tiny-imagenet-200/train'
        class_names = os.listdir(root)
        classes = {}
        for i, class_ in enumerate(class_names):
            classes[class_] = i
        dataset = build_dataset(classes, root=root, suffix='images')
    else:
        raise NotImplementedError

    np.random.seed(0)
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
    if args.iid == 0:
        dict_ = iid(dataset, args.num_clients)
        for i in range(args.num_clients):
            client_data = []
            #print(f'client {i}', 'dict_[i]', dict_[i])
            for j in tqdm(dict_[i]):
                client_data.append((dataset[j][0], dataset[j][1]))

            train_set, test_set = train_test_split(client_data, frac=args.frac)
            in_data.append(train_set)
            out_data.append(test_set)

    elif args.iid == 1:   # quantity shift
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

    elif args.iid == 2:   # label shift
        num_classes = args.num_classes
        labeled = {}
        for i in range(num_classes):
            labeled[i] = []
            
        for images, label in dataset:
            labeled[label].append((images, label))
        
        in_datas, out_datas = {}, {}    
        for i in range(args.num_clients):
            in_datas[i], out_datas[i] = [], []
            
        for label in range(num_classes):
            in_data, out_data = quantity_split(labeled[label], beta=args.dirichlet, num_clients=args.num_clients)
            for i in range(args.num_clients):
                in_datas[i] += in_data[i]
                out_datas[i] += out_data[i]
        
        in_data, out_data = in_datas, out_datas
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
