#!/usr/bin/env sh

CAFFE=/Users/donchan/caffe/caffe
TOOLS=$CAFFE/build/tools

$TOOLS/caffe train \
    --solver=cifar100_full_solver.prototxt
