# -*- coding: utf-8 -*-

import copy
import os
from subprocess import call

import numpy as np
import sklearn
import sklearn.linear_model

import cPickle

import matplotlib.pyplot as plt

import lmdb
import caffe


def createLmdbFile(images,labels,test=False):
    
    #lmdb_dir = "C:\\VBox\\Prescription"
    if test:
        lmdb_dir="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/test_lmdb"
    else:
        lmdb_dir="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/train_lmdb"
        
    #X = np.array(images)
    X = images
    print X.shape
    map_size = X.nbytes * 10
    lmdb_env = lmdb.open(lmdb_dir,map_size=map_size)

    N = len(images)
    with lmdb_env.begin(write=True) as txn:
    # txn is a Transaction object
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(labels[i])
            str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            
            
            
            im = np.array(Image.open(in_)) # or load whatever ndarray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())



def process(nfile,test=False):
    
    ## training data 
    nfiletrain = nfile + "train"
    data_dict = unpickle(nfiletrain)
    
    train_data = data_dict['data']
    train_fine_labels = data_dict['fine_labels']
    train_coarse_labels = data_dict['coarse_labels']


    print "training data shape", train_data.shape
    print "training fine label", type(train_fine_labels),len(train_fine_labels)
    print "training coarse shape", type(train_coarse_labels),len(train_coarse_labels)

    
    ## meta
    nfilemeta = nfile + "meta"
    bm = unpickle(nfilemeta)
    clabel_names = bm['coarse_label_names']
    flabel_names = bm['fine_label_names']
    print "meta clabel ", type(clabel_names),len(clabel_names)
    print "meta flabel ", type(flabel_names),len(flabel_names)

    if test:
    # example of data display
        n = 10
        im = train_data[n]
        print "image data shape", im.shape
        im0 = im.reshape(3,32,32)
        im00 = np.transpose(im0, (1,2,0))
    
        plt.imshow(im00)
        plt.show()
    
        print "fine label" , train_fine_labels[n]
        print "super label", train_coarse_labels[n]
        print "fine class name", flabel_names[train_fine_labels[n]]        
        print "super class name", clabel_names[train_coarse_labels[n]]


    X = train_data.reshape(50000,3,32,32)
    y = train_fine_labels
    createLmdbFile(X,y)    
    
    
    nfiletrain = nfile + "test"
    test_dict = unpickle(nfiletrain)
    
    test_data = test_dict['data']
    test_fine_labels = test_dict['fine_labels']
    test_coarse_labels = test_dict['coarse_labels']

    X = test_data.reshape(10000,3,32,32)
    y = test_fine_labels
    createLmdbFile(X,y,True)    
    

    
def unpickle(nfile):
    print "unpacking file", nfile
    fo = open(nfile, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
        
        
def main():

    myTrainingFile = "/Users/donchan/caffe/caffe/data/cifar-100-python/"
    process(myTrainingFile)
    
    
if __name__ == "__main__":
    main()