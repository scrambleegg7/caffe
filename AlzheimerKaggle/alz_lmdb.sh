#!/usr/bin/env sh

TOOLS=/home/hideaki/cafferoot/caffe/build/tools

$TOOLS/caffe train \
  --solver=./alz_lmdb_quick_solver.prototxt
