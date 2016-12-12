#!/usr/bin/env sh

CAFFE=/Users/donchan/caffe/caffe
TOOLS=$CAFFE/build/tools

$TOOLS/caffe train \
    --solver=cnn_solver_h5_lr1.prototxt \
    --snapshot=cnn_snapshot_iter_50000.solverstate
