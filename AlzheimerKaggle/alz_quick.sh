#!/usr/bin/env sh

TOOLS=/Users/donchan/caffe/caffe/build/tools

$TOOLS/caffe train \
  --solver=./alz_quick_solver.prototxt
