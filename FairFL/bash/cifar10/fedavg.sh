#!/bin/bash

cd ../../algs/

python run.py --device=$1 \
    --data_dir=split_0.8 \
    --dataset=CIFAR10 \
    --algorithm=FedAvg \
    --num_clients=10 \
    --num_epochs=200 \
    --learning_rate=$2 \
    --save_epoch=10 \
    --seed=$3

cd ../bash/cifar10/
