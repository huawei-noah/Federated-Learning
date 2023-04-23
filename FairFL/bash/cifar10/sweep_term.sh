#!/bin/bash


device=6
alpha=0.01

. term.sh ${device} ${alpha} 0.001 0
. term.sh ${device} ${alpha} 0.002 1
. term.sh ${device} ${alpha} 0.005 2
. term.sh ${device} ${alpha} 0.01 3
. term.sh ${device} ${alpha} 0.02 4
. term.sh ${device} ${alpha} 0.05 5
. term.sh ${device} ${alpha} 0.1 6


