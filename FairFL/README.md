# Proportional Fairness in Federated Learning

This repo is for comparing different federated learning algorithms using standard benchmarking datasets. It contains methods for obtaining the datasets for each client, preprocessing, training and evaluating. Currently, we support the following FL algorithms:

* Proportional Fairness in Federated Learning ([PropFair](https://openreview.net/forum?id=ryUHgEdWCQ))
* Federated Averaging ([FedAvg](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)) 
* Agnostic Federated Learning ([AFL](http://proceedings.mlr.press/v97/mohri19a/mohri19a.pdf))
* q-Fair Federated Learning ([q-FFL](https://openreview.net/pdf?id=ByexElSYDr))
* Tilted Empirical Risk Miminization ([TERM](https://openreview.net/forum?id=K5YasWXZT3O))
* FedMGDA+ ([FedMGDA+](https://ieeexplore.ieee.org/document/9762229/))
* GIFAIR-FL ([GIFAIR-FL](https://pubsonline.informs.org/doi/full/10.1287/ijds.2022.0022))



## Requirments
Install the following packages:

* torch 1.4.0
* torchvision 0.5.0
* tqdm
* cuda 10.1
* h5py
* matplotlib
* numpy

An example installation command:
``conda install torch==1.4.0 torchvision==0.5.0``;
``conda install cudatoolkit=10.1``


## Data
Currently, we support 6 datasets:
* CIFAR10
* CIFAR100
* CINIC-10
* TinyImageNet
* Shakespeare
* StackOverflow

For CIFAR10 and CIFAR100, download train and test datasets manually or they will be automatically downloaded from torchvision datasets, and partitioned automatically (see /data/preprocess_vision.py). In `preprocess_vision.py`, you need to set the arguments based on the dataset you want to use. The default values of "beta" (used for Dirichlet split of the whole data among clients) and "frac" (used for train/test splitting of each client's data) are 0.8 and 0.2, respectively. For example, if you want to create an iid split with 4 clients on MNIST, run the following inside the `data/` folder:
```sh
python preprocess_vision.py --dataset=MNIST --output_dir='iid-4' --iid=0 --num_clients=4
```
and the created split would be saved in `data/MNIST/iid-4`. For CINIC-10/TinyImageNet, first download the data by running ``download_cinic10.sh``/``download_tinyimagenet.sh`` inside the ``data`` folder, and then run ``preprocess_vision.py`` in the same way as before. 

For Shakespeare, you need to download it by running `/data/download_shakespeare.sh` in the folder `/data` to download the data and preprocess into json file (for further details of preprocess.sh file, read this [repo](https://github.com/SMILELab-FL/FedLab-benchmarks/tree/master/fedlab_benchmarks/leaf). If you run this file more than once, remember to delete folders `/data/Shakespeare/shakespeare/data/` and 
`/data/Shakespeare/shakespeare/meta/` before each time.) Then you need to run `/data/preprocess_shakespeare.py` in the `/data` folder to save Shakespeare into pickle files.

For StackOverflow, you need to download `stackoverflow_train.h5` manually on [google drive](https://drive.google.com/drive/folders/1-zQivrESzi8GMPMql57mWf0qJ5FCp1cK) and save it to `data/StackOverflow`. Then you need to run the file `/data/StackOverflow/generate_StackOverflow_nwp.py` to sample some users from the dataset and save their data in two pickle files to be used later.


## Experiments
* The following command shows an example command for running FedAvg algorithm on CIFAR10 with a fixed random seed. The command should be run inside the `algs` folder. Always make sure the preprocessed data directory, where 
`in.pickle` and `out.pickle` files are stored, are consistent with your `data_dir` input:

```sh
python run.py --algorithm=FedAvg --device=0 --data_dir=split_0.8 --num_clients=10 --learning_rate=0.005 \
      --dataset=CIFAR10 --num_epochs=200 --batch_size=64 --seed=0
```

## Output and Plot
* Outputs of the experiment (test accuracies and models) will be stored in pickle and checkpoint files. You can access the files and plot them afterwards by writing your own plotting script.
* Remember to modify the output file names in the code according to your demand so that if you run multiple process at once, the output files won't be overwritten.


## Safety notice
Loading unwarranted pickle files may result in vulnerability of deserialization.
