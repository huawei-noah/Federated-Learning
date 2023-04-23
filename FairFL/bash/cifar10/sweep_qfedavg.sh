#!/bin/bash

device=$1
. qfedavg.sh ${device} 32 4
. qfedavg.sh ${device} 32 5
. qfedavg.sh ${device} 32 6


. qfedavg.sh ${device} 128 7
. qfedavg.sh ${device} 128 8
. qfedavg.sh ${device} 128 9


. qfedavg.sh ${device} 256 10
. qfedavg.sh ${device} 256 11
. qfedavg.sh ${device} 256 12
