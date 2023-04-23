import argparse
import logging
import os
import pickle
import tqdm
import random
import numpy as np
import torch.utils.data as data
import torch
import h5py
import utils
import argparse


client_ids_train = None
client_ids_test = None
# DEFAULT_TRAIN_CLIENTS_NUM = 342477
# DEFAULT_TEST_CLIENTS_NUM = 204088
DEFAULT_TRAIN_CLIENTS_NUM = 10000
DEFAULT_TEST_CLIENTS_NUM = 10000
DEFAULT_BATCH_SIZE = 1
DEFAULT_TRAIN_FILE = 'stackoverflow_train.h5'
DEFAULT_TEST_FILE = 'stackoverflow_test.h5'
DEFAULT_CACHE_FILE = 'stackoverflow_nwp.pkl'




def tokenizer(sentence, data_dir, max_seq_len=20):

    truncated_sentences = sentence.split(' ')[:max_seq_len]

    def word_to_id(word, num_oov_buckets=1):
        word_dict = get_word_dict(data_dir)
        if word in word_dict:
            return word_dict[word]
        else:
            return hash(word) % num_oov_buckets + len(word_dict)

    def to_ids(sentence, num_oov_buckets=1):
        '''
        map list of sentence to list of [idx..] and pad to max_seq_len + 1
        Args:
            num_oov_buckets : The number of out of vocabulary buckets.
            max_seq_len: Integer determining shape of padded batches.
        '''
        tokens = [word_to_id(token) for token in sentence]
        if len(tokens) < max_seq_len:
            tokens = tokens + [word_to_id(_eos)]
        tokens = [word_to_id(_bos)] + tokens
        if len(tokens) < max_seq_len + 1:
            tokens += [word_to_id(_pad)] * (max_seq_len + 1 - len(tokens))
        return tokens

    return to_ids(truncated_sentences)



def get_word_dict(data_dir):
    global word_dict
    if word_dict == None:
        frequent_words = get_most_frequent_words(data_dir)
        words = [_pad] + frequent_words + [_bos] + [_eos]
        word_dict = collections.OrderedDict()
        for i, w in enumerate(words):
            word_dict[w] = i
    return word_dict



class StackOverflowDataset(data.Dataset):
    """StackOverflow dataset"""

    __train_client_id_list = None
    __test_client_id_list = None

    def __init__(self, h5_path, client_idx, datast, preprocess):
        """
        Args:
            h5_path (string) : path to the h5 file
            client_idx (idx) : index of train file
            datast (string) : "train" or "test" denoting on train set or test set
            preprocess (callable, optional) : Optional preprocessing
        """
        
        self._EXAMPLE = 'examples'
        self._TOKENS = 'tokens'
        self.h5_path = h5_path
        self.datast = datast
        self.client_id = self.get_client_id_list()[client_idx]
        self.preprocess = preprocess

    def get_client_id_list(self):
        if self.datast == "train":
            if StackOverflowDataset.__train_client_id_list is None:       
                with h5py.File(self.h5_path, 'r') as h5_file:
                    StackOverflowDataset.__train_client_id_list = list(h5_file[self._EXAMPLE].keys())
            return StackOverflowDataset.__train_client_id_list
        elif self.datast == "test":
            if StackOverflowDataset.__test_client_id_list is None:       
                with h5py.File(self.h5_path, 'r') as h5_file:
                    StackOverflowDataset.__test_client_id_list = list(h5_file[self._EXAMPLE].keys())
            return StackOverflowDataset.__test_client_id_list 
        else:
            raise Exception ("Please specify either train or test set!") 


    def __len__(self):
        with h5py.File(self.h5_path, 'r') as h5_file:
            return len(h5_file[self._EXAMPLE][self.client_id][self._TOKENS][()])

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as h5_file:
            sample = h5_file[self._EXAMPLE][self.client_id][self._TOKENS][()][idx].decode('utf8')
            sample = self.preprocess(sample)
        return np.asarray(sample[:-1]), np.asarray(sample[1:])



def get_dataloader(dataset, data_dir, train_bs, test_bs, client_idx=None):

    def _tokenizer(x):
        return utils.tokenizer(x, data_dir)

    if client_idx is None:

        train_dl = data.DataLoader(data.ConcatDataset(
            StackOverflowDataset(
                os.path.join(data_dir, DEFAULT_TRAIN_FILE), client_idx, _tokenizer) 
                for client_idx in range(DEFAULT_TRAIN_CLIENTS_NUM)),
                                   batch_size=train_bs,
                                   shuffle=True)

        test_dl = data.DataLoader(data.ConcatDataset(
            StackOverflowDataset(
                os.path.join(data_dir, DEFAULT_TEST_FILE), client_idx, "test", _tokenizer)
                for client_idx in range(DEFAULT_TEST_CLIENTS_NUM)),
                                  batch_size=test_bs,
                                  shuffle=True)
        return train_dl, test_dl

    else:
        train_ds = StackOverflowDataset(
            os.path.join(data_dir, DEFAULT_TRAIN_FILE), client_idx, "train", _tokenizer) 
        train_dl = data.DataLoader(dataset=train_ds,
                                   batch_size=train_bs,
                                   shuffle=True,
                                   drop_last=False)

        if client_idx >= DEFAULT_TEST_CLIENTS_NUM:
            test_dl = None
        else:
            test_ds = StackOverflowDataset(
                os.path.join(data_dir, DEFAULT_TEST_FILE), client_idx, "test", _tokenizer) 
            test_dl = data.DataLoader(dataset=test_ds,
                                      batch_size=test_bs,
                                      shuffle=True,
                                      drop_last=False)

        return train_dl, test_dl



def load_partition_data_federated_stackoverflow_nwp(dataset, data_dir, batch_size = DEFAULT_BATCH_SIZE, frac=0.5):
    logging.info("load_partition_data_federated_stackoverflow_nwp START")

    cache_path = os.path.join(data_dir, DEFAULT_CACHE_FILE)
    if os.path.exists(cache_path):
        #load cache
        with open(cache_path, 'rb') as cache_file:
            cache_data = pickle.load(cache_file)
            train_data_num = cache_data['train_data_num']
            test_data_num = cache_data['test_data_num']
            train_data_global = cache_data['train_data_global']
            test_data_global = cache_data['test_data_global']
            data_local_num_dict = cache_data['data_local_num_dict']
            train_data_local_dict = cache_data['train_data_local_dict']
            test_data_local_dict = cache_data['test_data_local_dict']
            VOCAB_LEN = cache_data['VOCAB_LEN']

    else:
        # get local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()
        
        for client_idx in tqdm.tqdm(range(DEFAULT_TRAIN_CLIENTS_NUM)):

            train_data_local, test_data_local = get_dataloader(
                dataset, data_dir, batch_size, batch_size, client_idx)
            local_data_num = len(train_data_local.dataset)
            data_local_num_dict[client_idx] = local_data_num
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local

        train_data_global = data.DataLoader(
                    data.ConcatDataset(
                        list(dl.dataset for dl in list(train_data_local_dict.values()))
                    ),
                    batch_size=batch_size, shuffle=True)
        train_data_num = len(train_data_global.dataset)
        
        test_data_global = data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None)
                ),
                batch_size=batch_size, shuffle=True)
        test_data_num = len(test_data_global.dataset)

        VOCAB_LEN = len(utils.get_word_dict(data_dir)) + 1

        #save cache
        nums = [i for i in data_local_num_dict.values()]
        train_loader = [train_data_local_dict[i] for i in range(DEFAULT_TRAIN_CLIENTS_NUM) if nums[i]>args.min_num_samples and nums[i]<args.max_num_samples]
        samples = [nums[i] for i in range(DEFAULT_TRAIN_CLIENTS_NUM) if nums[i]>args.min_num_samples and nums[i]<args.max_num_samples]

        
        num_sampled_users = len(samples)
        print('{} users were found with min {} and max {} samples'.format(num_sampled_users, args.min_num_samples, args.max_num_samples))
        in_datas, out_datas = {}, {}
        for i in range(num_sampled_users):
            in_datas[i], out_datas[i] = [], []

        for client_index in range(num_sampled_users):
            print('fetching data of client {}'.format(client_index))
            num_samples = samples[client_index]
            for j in range(num_samples):
                sample, label = next(iter(train_loader[client_index]))

                if j <= np.ceil(frac*num_samples):
                    in_datas[client_index].append((sample[0], label[0]))
                elif j > np.ceil(frac*num_samples):
                    out_datas[client_index].append((sample[0], label[0]))

        os.makedirs(args.output_dir, exist_ok=True)
        output_in = os.path.join(args.output_dir, 'in.pickle')
        output_out = os.path.join(args.output_dir, 'out.pickle')


        with open(output_in, 'wb') as output:
            pickle.dump(in_datas, output)
        with open(output_out, 'wb') as output:
            pickle.dump(out_datas, output)
        print('done')            




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='data split')
    parser.add_argument('--num_clients', type=float, default=20)
    parser.add_argument('--frac', type=float, default=0.5)
    parser.add_argument('--min_num_samples', type=int, default=10000)
    parser.add_argument('--max_num_samples', type=int, default=20000)
    parser.add_argument('--output_dir', type=str, default='./split_StackOverflow')
    args = parser.parse_args()

    dataset_name = "stackoverflow_nwp"
    # make sure that the data is downloaded and saved in the "saved_data_dir" directory:
    saved_data_dir = "."

    # clients satisfying following number of samples will be sampled from "DEFAULT_TRAIN_CLIENTS_NUM" number of clients.
    # with min=10000 and max=12500, this results in 20 clients. Finally the sampled users' train data is splitted to train 
    # and test data with ratio of "frac".

    
    # Set the random seed. The np.random seed determines the dataset partition.
    random.seed(0)
    np.random.seed(0)

    # load and save data
    load_partition_data_federated_stackoverflow_nwp(dataset_name, saved_data_dir, frac=args.frac)
