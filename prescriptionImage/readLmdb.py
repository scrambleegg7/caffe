# -*- coding: utf-8 -*-

import os
import glob
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import patches

from scipy import ndimage
from time import time

from sklearn import decomposition
from sklearn.cluster import KMeans

from skimage.feature import match_template
import lmdb
import caffe

def readLmdbFile(lmdb_dir):

    #lmdb_dir="/Users/donchan/Documents/Statistical_Mechanics/caffe/mylmdb"
    lmdb_env = lmdb.open(lmdb_dir, readonly=True)

    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()    

    labels_ = []
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        labels_.append(datum.label)
    
    print "lables:",len(set(labels_))
    
    idx = 0
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        #label = datum.label
    
        data = caffe.io.datum_to_array(datum)
        #print "Datum data shape", np.fromstring(datum.data,dtype=np.uint8).shape
        print "key val %s : Label %s" % (key, datum.label)
        #rows = datum.height;
        #cols = datum.width;
        #print "rows cols", rows,cols
    
        im = data.astype(np.uint8)
        print "image shape", im.shape
        
        #image stored on lmdb --> channel , col , row 
        im = np.transpose(im, (1,2,0)) # original (dim, col, row)
    
        fig = plt.figure()
        plt.imshow(im)
        
        if idx > 2:
            break
        idx += 1

def resizeLbdbFile(lmdb_dir,sizeratio=0.5):

    newlmdb_dir="/Users/donchan/Documents/Statistical_Mechanics/caffe/prescriptionImage/testlmdb"
    
    lmdb_env = lmdb.open(lmdb_dir, readonly=True)

    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()    
    
    idx = 0
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label

        
def main():
    
    #lmdb_dir="/Users/donchan/Documents/Statistical_Mechanics/caffe/prescriptionImage/mylmdb"
    #lmdb_dir="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/train_lmdb"
    lmdb_dir="/Users/donchan/caffe/caffe/examples/cifar10/cifar10_train_lmdb"

        
    #mydir = "L:\\Prescription_Photo\\20151102"
    #mydir = "/Volumes/myShare/Prescription_Photo/20151102"
    #im = readImage(mydir)
    #createLmdbFile(im)
    readLmdbFile(lmdb_dir)
    
if __name__ == "__main__":
    main()
