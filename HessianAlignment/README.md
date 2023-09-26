# Understanding Hessian Alignment for Domain Generalization

This repository contains the implementation details of the ICCV 2023 paper, "Understanding Hessian Alignment for Domain Generalization".
It contains the HGP and Hutchinson algorithms for Domain Generalization, on the domainbed benchmark.  
Our implementation is an extension of the [DomainBed](https://github.com/facebookresearch/DomainBed) testbed. You can find the implementations of our algorithms at `domainbed/algorithms.py`.


## Dependencies
This repository was developed under Python 3.6.9, with the following packages

* numpy
* wilds
* imageio
* gdown
* torchvision
* torch
* tqdm
* backpack
* parameterized
* Pillow

Install the dependencies by running:
``pip install -r requirements.txt``

## Datasets
We currently support the following datasets in the domainbed testbed:

The [currently available datasets](domainbed/datasets.py) are:

* ColoredMNIST ([Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893))
* VLCS  ([Fang et al., 2013](https://openaccess.thecvf.com/content_iccv_2013/papers/Fang_Unbiased_Metric_Learning_2013_ICCV_paper.pdf))
* PACS ([Li et al., 2017](https://arxiv.org/abs/1710.03077))
* Office-Home ([Venkateswara et al., 2017](https://arxiv.org/abs/1706.07522))
* DomainNet ([Peng et al., 2019](http://ai.bu.edu/M3SDA/))

## ColoredMNIST Experiments

In the `cmnist` folder, you can find the different implementations of `ColoredMNIST` experiments.  
Each file corresponds to one of the algorithms `HGP` or `Hutchinson` for both imbalanced and balanced cases.  
For example you can run the balanced ColoredMNIST for `HGP` using the following command: `python cmnist/cmnist_HGP.py`  

## Experiments

Download the datasets:

```sh
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data
```

Run a sweep:

```sh
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher\
       --algorithms HGP Hutchinson\
       --datasets PACS VLCS\
       --n_hparams 20\
       --n_trials 3
```
Here, `MyLauncher` is your cluster's command launcher, as implemented in `command_launchers.py`.  
After all jobs have either succeeded or failed, you can delete the data from failed jobs with ``python -m domainbed.scripts.sweep delete_incomplete`` and then re-launch them by running ``python -m domainbed.scripts.sweep launch`` again. Specify the same command-line arguments in all calls to `sweep` as you did the first time; this is how the sweep script knows which jobs were launched originally.

To view the results of your sweep:

````sh
python -m domainbed.scripts.collect_results --input_dir=/my/sweep/output/path
````

## Citation

Comments are welcome! Please use the following bib if you use our code in your research:

```
@article{hemati2023understanding,
  title={Understanding Hessian Alignment for Domain Generalization},
  author={Hemati, Sobhan and Zhang, Guojun and Estiri, Amir and Chen, Xi},
  journal={ICCV},
  year={2023}
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
