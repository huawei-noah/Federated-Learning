# Mitigating Data Heterogeneity in Federated Learning with Data Augmentation

This repository contains the implementation details of the work "Mitigating Data Heterogeneity in Federated Learning with Data Augmentation".
It contains different algorithms in both Federated Learning and Domain Generalization, as well as auxilary methods for the experimental setup.
We currently support the following algorithms:

### Federated Learning
* Federated Averaging (FedAvg, [McMahan et al., 2017](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)) 
* Agnostic Federated Learning (AFL, [Mohri et al., 2019](http://proceedings.mlr.press/v97/mohri19a/mohri19a.pdf))
* FedProx (FedProx, [Li et al., 2020](https://arxiv.org/abs/1812.06127))
* Generalized AFL (Gen-AFL)
* Variance Minimization (VM)
* Federated IRM (Fed-IRM)

### Domain Generalization
* Empirical Risk Minimization (ERM, [Vapnik et al., 1991](https://proceedings.neurips.cc/paper/1991/file/ff4d5fbbafdf976cfdc032e3bde78de5-Paper.pdf))
* Invariant Risk Minimization (IRM, [Arjovski et al., 2019](https://arxiv.org/abs/1907.02893))
* Distributionally Robust Neural Networks for Group Shifts (GroupDRO, [Sagawa et al., 2019](https://arxiv.org/abs/1911.08731))
* Minimax Risk Extrapolation (MM-REx, [Krueger et al., 2021](https://arxiv.org/abs/2003.00688))
* Variance Risk Extrapolation (V-REx, [Krueger et al., 2021](https://arxiv.org/abs/2003.00688)) 

## Dependencies
This repository was developed under Python 3.6.9, with the following packages
* torch 1.8.0
* torchvision 0.9.0
* scipy
* matplotlib
* numpy
* tqdm

Install the dependencies by running:
``pip install .``

## Datasets
We currently support the following datasets:
* RotatedMNIST ([Ghifary et al., 2015](https://arxiv.org/abs/1508.07680))
* PACS ([Li et al., 2017](https://arxiv.org/abs/1710.03077))
* OfficeHome ([Venkateswara et al., 2017](https://arxiv.org/abs/1706.07522))

For RotatedMNIST, the dataset is automatically downloaded once the corresponding functions are called.
As for PACS and OfficeHome, datasets must be manually downloaded and placed inside the `data/datasets/` folder.
Additionally, subfolders must be in the following structure:
* PACS (`P, A, C, and S`)
* OfficeHome (`Art, Clipart, Product, and RealWorld`)

## Experiments
Once the dependencies have been installed and the datasets are in place, refer to the `scripts/` folder to execute the experiments.
For instance, the following command illustrates how to run FedAvg on Rotated MNIST 
* The following command illustrates how to run FedAvg on Rotated MNIST with 15° rotation:
```sh
python scripts/fl_rmnist.py --alg fed_avg --steps 200 --rounds 80 --data da_15 --seed 0
```

By default the experiments are stored in `data/experiments/`. Remember to modify the folder name when running similar experiments multiple times, so that the output files won't be overwritten.

## Citation

Comments are welcome! Please use the following bib if you use our code in your research:

```
@article{back2022mitigating,
  title={Mitigating data heterogeneity in federated learning with data augmentation},
  author={Back de Luca, Artur and Zhang, Guojun and Chen, Xi and Yu, Yaoliang},
  journal={arXiv preprint arXiv:2206.09979},
  year={2022}
}
```

Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.

## License
We follows Apache License Version 2.0. Please see the [License](./LICENSE) file for more information.
Disclaimer: This is not an officially supported HUAWEI product.