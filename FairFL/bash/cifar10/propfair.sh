#!/bin/bash

cd ../../algs/

python run.py --device=$1 \
    --data_dir=split_0.8 \
    --dataset=CIFAR10 \
    --algorithm=PropFair \
    --num_clients=10 \
    --num_epochs=$2 \
    --base=$3 \
    --learning_rate=$4 \
    --epsilon=0.2 \
    --seed=$5

cd ../bash/cifar10/
