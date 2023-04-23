#!/bin/bash

cd ../../algs/

python run.py --device=$1 \
    --data_dir=split_0.5 \
    --dataset=CIFAR10 \
    --algorithm=TERM \
    --alpha=$2 \
    --num_clients=10 \
    --num_epochs=100 \
    --learning_rate=$3 \
    --seed=$4

cd ../bash/cifar10/
