#!/bin/bash


device=$1
base=$2

. propfair.sh ${device} ${base} 0.005 0
. propfair.sh ${device} ${base} 0.01 1
. propfair.sh ${device} ${base} 0.02 2
. propfair.sh ${device} ${base} 0.05 3


