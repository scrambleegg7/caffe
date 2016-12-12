# -*- coding: utf-8 -*-

import os

import numpy as np
import sklearn
import sklearn.linear_model

import cPickle

import matplotlib.pyplot as plt

import lmdb
import caffe

import h5py

#from caffe.proto import caffe_pb2
from caffe.io import array_to_blobproto
from skimage import io

from sklearn.cross_validation import train_test_split



def createh5MultiLabelFile(X,y_f,y_c,test=False):
    
    #lmdb_dir = "C:\\VBox\\Prescription"
    if test:
        h5_filedir="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/test.h5"
        file_name="test.txt"
    else:
        h5_filedir="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/train.h5"
        file_name="train.txt"

    cifar_caffe_directory = "/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar"
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    # Train
    with h5py.File(h5_filedir, 'w') as f:
        f.create_dataset('data', data=X, **comp_kwargs)
        f.create_dataset('label_coarse', data=y_c.astype(np.int_), **comp_kwargs)
        f.create_dataset('label_fine', data=y_f.astype(np.int_), **comp_kwargs)
    with open(os.path.join(cifar_caffe_directory, file_name), 'w') as f:
        f.write(h5_filedir + '\n')

    print('Conversion successfully done to "{}".\n'.format(cifar_caffe_directory))



def createLmdbFile(images,labels,test=False):
    
    #lmdb_dir = "C:\\VBox\\Prescription"
    if test:
        lmdb_dir="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/test_lmdb"
    else:
        lmdb_dir="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/train_lmdb"
        
    #X = np.array(images)
    X = images
    
    print "test -  X" if test else "train - X"
    print " ******* X shape --> ", X.shape
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
            
            
            
            #im = np.array(Image.open(in_)) # or load whatever ndarray you need
        #im = im[:,:,::-1]
        #im = im.transpose((2,0,1))
        #im_dat = caffe.io.array_to_datum(im)
        #in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())


def makeMeanBinary(X):
    
    
    c, h, w = X.shape[1:]
    mean = np.zeros( ( 1,c,h,w )  )
    loop_ =X.shape[0]
    
    print "channel height  width", c,h,w
    print "loop count :", loop_
    
    for i in range(loop_):
        mean[0][0] += X[i,0,:,:]
        mean[0][1] += X[i,1,:,:]
        mean[0][2] += X[i,2,:,:]
        
    mean[0] /= loop_
    
    print "mean shape", mean.shape
    print "mean[0] shape", mean[0].shape
    #print "mean image", mean
    
    #print np.asarray(mean[0],dtype=np.uint8)
    print "*** subtract mean image from X training image"
    #for i in range(loop_):
    #    X[i] -= np.asarray(mean[0],dtype=np.uint8)
    
    myroot = "/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/"
    meanbinary_proto = os.path.join(myroot, "cnn_mean.binaryproto")
    meanbinary_npy = os.path.join(myroot,"cnn_mean.npy")
    meanbinary_png = os.path.join(myroot,"cnn_mean.png")

    blob = array_to_blobproto(mean)
    print "set [num x c x h x w] parameters for reshaping with blobproto_to_array"
    # set for reshaping with blobproto_to_array  
    blob.num,blob.channels,blob.height,blob.width = mean.shape

    with open(meanbinary_proto, 'wb') as f:
        f.write(blob.SerializeToString())
    np.save(meanbinary_npy, mean[0])

    meanImg = np.transpose(mean[0].astype(np.uint8), (1, 2, 0))
    io.imsave(meanbinary_png, meanImg)

    mean_array = np.asarray(
        blob.data,
        dtype=np.float32).reshape(
        (blob.height,
         blob.width,blob.channels ))


    return mean_array

def process(nfile,test=False):
    
    ## training data 
    nfiletrain = nfile + "train"
    data_dict = unpickle(nfiletrain)
    
    train_data = data_dict['data']
    train_fine_labels = data_dict['fine_labels']
    train_coarse_labels = data_dict['coarse_labels']


    print "- training data shape", train_data.shape
    print "- training fine label", type(train_fine_labels),len(train_fine_labels)
    print "- training coarse shape", type(train_coarse_labels),len(train_coarse_labels)

    
    ## meta for categorize with String Character, other than numerical data
    nfilemeta = nfile + "meta"
    bm = unpickle(nfilemeta)
    clabel_names = bm['coarse_label_names']
    flabel_names = bm['fine_label_names']
    print "- meta clabel ", type(clabel_names),len(clabel_names)
    print "- meta flabel ", type(flabel_names),len(flabel_names)

    if test:
    # example of data display
        n = 10
        im = train_data[n]
        print "image data shape", im.shape
        im0 = im.reshape(3,32,32)
        
        print "******* should be converted to BGR from RGB for caffe"
        im0 = im0[::-1,:,:]
        im00 = np.transpose(im0, (1,2,0))
    
        plt.imshow(im00)
        plt.show()
    
        print "fine label" , train_fine_labels[n]
        print "super label", train_coarse_labels[n]
        print "fine class name", flabel_names[train_fine_labels[n]]        
        print "super class name", clabel_names[train_coarse_labels[n]]


    X = train_data.reshape(50000,3,32,32)
    

    #print "---- select just 10000 with random select -----"
    #r = np.random.randint(50000,size=10000)    
    #X = X[r]
    mean_array = makeMeanBinary(X)
    print "------ convert RGB to BGR for training data....."
    X = X[:, ::-1, :, :] # RGB to BGR  

    y_f = np.array(train_fine_labels)
    y_c = np.array(train_coarse_labels)
    

    if not test:
        createh5MultiLabelFile(X,y_f,y_c,False)    
        createLmdbFile(X,y_f) 
    
    
    nfiletrain = nfile + "test"
    test_dict = unpickle(nfiletrain)
    
    test_data = test_dict['data']
    test_fine_labels = test_dict['fine_labels']
    test_coarse_labels = test_dict['coarse_labels']

    X = test_data.reshape(10000,3,32,32)
    
    print "------ convert RGB to BGR for test data....."    
    X = X[:, ::-1, :, :] # RGB to BGR      

    y_f = np.array(test_fine_labels)
    y_c = np.array(test_coarse_labels)
    if not test:    
        createh5MultiLabelFile(X,y_f,y_c,True)    
        createLmdbFile(X,y_f,True)    

    
def unpickle(nfile):
    print "unpacking file", nfile
    fo = open(nfile, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
        
        
def main():

    myTrainingFile = "/Users/donchan/caffe/caffe/data/cifar-100-python/"

    test=False
    process(myTrainingFile,test)
    
    
if __name__ == "__main__":
    main()