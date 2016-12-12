#!/bin/sh

CAFFEPATH=/Users/donchan/caffe/caffe/python

$CAFFEPATH/draw_net.py alz_lmdb_quick_train_test.prototxt alz_lmdb.png 
$CAFFEPATH/draw_net.py alz_quick_train_test.prototxt alz_train.png 
$CAFFEPATH/draw_net.py lenet_alz_train_test.prototxt lenet_train.png 
