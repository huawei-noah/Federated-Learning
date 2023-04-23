#!/bin/bash

device=2
. afl.sh ${device}  0.001 0
. afl.sh ${device} 0.002 1
. afl.sh ${device} 0.005 2
. afl.sh ${device} 0.01 3
. afl.sh ${device} 0.02 4
. afl.sh ${device} 0.05 5
. afl.sh ${device} 0.1 6


