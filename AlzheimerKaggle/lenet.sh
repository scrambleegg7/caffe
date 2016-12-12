#!/usr/bin/env sh

TOOLS=/Users/donchan/caffe/caffe/build/tools

$TOOLS/caffe train \
  --solver=./lenet_alz_solver.prototxt
