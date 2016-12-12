#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/imagenet
DATA=data/ilsvrc12
TOOLS=/Users/donchan/caffe/caffe/build/tools

$TOOLS/compute_image_mean ./mylmdb \
  ./prescription_mean.binaryproto

echo "Done."
