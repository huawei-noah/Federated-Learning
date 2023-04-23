#!/bin/bash

cd ../../algs/

python run.py --device=$1 \
    --data_dir=label-10 \
    --dataset=CIFAR10 \
    --algorithm=qFFL \
    --num_clients=10 \
    --num_epochs=300 \
    --q=0.1 \
    --learning_rate=0.002 \
    --seed=$2

cd ../bash/cifar10/
