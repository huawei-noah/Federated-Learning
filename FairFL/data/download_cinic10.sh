#!/bin/bash

mkdir -p CINIC10/raw_data

wget -nc 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'

tar -xvf CINIC-10.tar.gz -C CINIC10/raw_data

classes=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck")

for i in ${classes[@]}; do
      mv CINIC10/raw_data/test/$i/* CINIC10/raw_data/train/$i
      mv CINIC10/raw_data/valid/$i/* CINIC10/raw_data/train/$i
done

rm -r CINIC10/raw_data/test CINIC10/raw_data/valid
mv CINIC10/raw_data/train/* CINIC10/raw_data/
