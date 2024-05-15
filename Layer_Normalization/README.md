# Understanding the Role of Layer Normalization in Label-Skewed Federated Learning

This repository contains the implementation details of the TMLR paper, "Understanding the Role of Layer Normalization in Label-Skewed Federated Learning". It contains implementation of different federated learning algorithgms and various normalization methods in deep networks. 


## Requirments
Install the packages in the `requirements-conda.txt` and `requirements-pip.txt` files:

* torch 1.13.1
* torchvision 0.14.1
* tqdm
* cuda 10.1
* h5py
* matplotlib
* numpy

An example installation command:
``pip install torch==1.4.0 torchvision==0.5.0``;
``conda install cudatoolkit=10.1``


## Data
Currently, we support 4 datasets:
* CIFAR10
* CIFAR100
* PACS
* TinyImageNet

For CIFAR10 and CIFAR100, download train and test datasets manually or they will be automatically downloaded from torchvision datasets, and partitioned automatically (see `data/preprocess_vision.py`). In `preprocess_vision.py`, you need to set the arguments based on the dataset you want to use. The default values of "$\beta$" (used for Dirichlet split of the whole data among clients). For example, if you want to create an 1 class split with 10 clients on CIFAR10, run the following inside the `data/` folder:
```sh
python data/preprocess_vision.py --dataset=CIFAR10 --output_dir='data' --iid_type=class_part\
     --class_per_client=1 --num_clients=10 --num_classes=10 
```


## Algorithms

* FedAvg
* FedRS
* FedLC
* FedDecorr
* FedYogi
* FedProx
* SCAFFOLD

To run the FedLN model with FedYogi algorithm for CIFAR10 and ResNet18 run the following command in the main folder:
```sh
python run_fl.py \
     --name='exp1' \
     --device=0 \
     --lr=0.01 \
     --num_local_steps=10 \
     --algorithm='FedYogi' \
     --model='ResNet18' \
     --num_rounds=10000 \
     --log_interval=100 \
     --logdir='./logs/CIFAR10/' \
     --datadir='./CIFAR10/' \
     --norm_methods_per_layer=group_norm_after \
     --affine
```


## Citation

Comments are welcome! Please use the following bib if you use our code in your research:

```
@article{Zhang2024Understanding,
  title={Understanding the Role of Layer Normalization in Label-Skewed Federated Learning},
  author={Zhang, Guojun and Beitollahi, Mahdi and Bie, Alex and Chen, Xi},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```

## License
We follow BSD-3.0. Please see the [License](./LICENSE.md) file for more information.
Disclaimer: This is not an officially supported HUAWEI product.
