'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

import os
import torch
from torch.utils.data import Dataset
import json, pickle
import numpy as np
import random
import sys
sys.path.append('..')
from utils.io import read_json
import argparse

class ShakespeareDataset(Dataset):
# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
    def __init__(self, client_id: int, client_str: str, data: list, targets: list):
        """get `Dataset` for shakespeare dataset

        Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): sentence list data
            targets (list): next-character target list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.ALL_LETTERS, self.VOCAB_SIZE = self._build_vocab()
        self.data = data
        self.targets = targets
        self._process_data_target()

    def _build_vocab(self):
        """ according all letters to build vocab

        Vocabulary re-used from the Federated Learning for Text Generation tutorial.
        https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation

        Returns:
            all letters vocabulary list and length of vocab list
        """
        ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        VOCAB_SIZE = len(ALL_LETTERS)
        return ALL_LETTERS, VOCAB_SIZE

    def _process_data_target(self):
        """process client's data and target
        """
        self.data = torch.tensor(
            [self.__sentence_to_indices(sentence) for sentence in self.data])
        self.targets = [self.__letter_to_index(letter) for letter in self.targets]

    def __sentence_to_indices(self, sentence: str):
        """Returns list of integer for character indices in ALL_LETTERS

        Args:
            sentence (str): input sentence

        Returns: a integer list of character indices
        """
        indices = []
        for c in sentence:
            indices.append(self.ALL_LETTERS.find(c))
        return indices

    def __letter_to_index(self, letter: str):
        """Returns index in ALL_LETTERS of given letter

        Args:
            letter (char/str[0]): input letter

        Returns: int index of input letter
        """
        index = self.ALL_LETTERS.find(letter)
        return index

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data split')
    parser.add_argument('--frac', type=float, default=0.8)
    parser.add_argument('--num_clients', type=int, default=20)
    args = parser.parse_args()
    print(args)
    output_dir = f'./Shakespeare/split_{args.frac}'
    os.makedirs(output_dir, exist_ok=True)
    
    #json_file = './Shakespeare/shakespeare/data/rem_user_data/all_data_niid_0_keep_10000.json'  # run in the data folder after download_shakespeare.sh is run
    json_file = './Shakespeare/shakespeare/data/all_data/all_data.json'  # run in the data folder after download_shakespeare.sh is run
    results = read_json(json_file)[0]
    users = results['users']
    num_samples = results['num_samples']
    user_sample = {}
    for _ in range(len(users)):
        user = users[_]
        num_sample = num_samples[_]
        user_sample[user] = num_sample
    np.random.seed(0)
    #choice = np.random.choice(len(users), args.num_clients, replace=False)
    choice = np.rint(np.random.exponential(200.0, size=args.num_clients))
    user_sample = sorted(user_sample.items(), key=lambda item: item[1], reverse=True)
    users = [user_sample[int(choice[i])] for i in range(args.num_clients)]
    datasets = {}
    in_data, out_data = {}, {}
    for i in range(args.num_clients):
        datasets[i] = list(ShakespeareDataset(i, users[i][0], results['user_data'][users[i][0]]['x'], \
                                results['user_data'][users[i][0]]['y']))
        random.shuffle(datasets[i])
        in_data[i] = datasets[i][:int(args.frac * users[i][1])]  # train-test split
        out_data[i] = datasets[i][int(args.frac * users[i][1]):]
    
    
    # print stats
    for i in range(args.num_clients):
        print('client: ', i, 'in data: ', len(in_data[i]), 'out data: ', len(out_data[i]))
    
    # dump in_data and out_data 
    output_in = os.path.join(output_dir, 'in.pickle')
    output_out = os.path.join(output_dir, 'out.pickle')
    torch.save(in_data, output_in)
    torch.save(out_data, output_out)
    print('data split successful!')
