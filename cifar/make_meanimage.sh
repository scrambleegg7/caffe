#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

CAFEPATH=/Users/donchan/caffe/caffe
EXAMPLE=examples/cifar10
DATA=/$HOME/Documents/Statistical_Mechanics/caffe/cifar
DBTYPE=hdf5

echo "Computing image mean..."

$CAFEPATH/build/tools/compute_image_mean -backend=$DBTYPE \
  $DATA/train.h5 ./cnn_mean.binaryproto

echo "Done."
