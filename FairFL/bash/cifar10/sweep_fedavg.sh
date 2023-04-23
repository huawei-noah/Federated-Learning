#!/bin/bash

device=$1
. fedavg.sh ${device} 32 4
. fedavg.sh ${device} 32 5
. fedavg.sh ${device} 32 6


. fedavg.sh ${device} 128 7
. fedavg.sh ${device} 128 8
. fedavg.sh ${device} 128 9


. fedavg.sh ${device} 256 10
. fedavg.sh ${device} 256 11
. fedavg.sh ${device} 256 12
