#!/bin/bash

cd ../../algs/

python run.py --device=$1 \
    --data_dir=split_0.5 \
    --dataset=CIFAR10 \
    --algorithm=qFedAvg \
    --num_clients=10 \
    --num_epochs=100 \
    --save_epoch=10 \
    --q=$2 \
    --learning_rate=$3 \
    --seed=$4

cd ../bash/cifar10/
