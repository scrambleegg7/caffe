#!/usr/bin/env sh

TOOLS=/home/hideaki/cafferoot/caffe/build/tools

$TOOLS/caffe train \
  --solver=./lenet_alz_solver.prototxt
