# -*- coding: utf-8 -*-

import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import sys
import os
import caffe
import h5py
import cv2
import lmdb

from caffe.proto import caffe_pb2


def myclassifier(envlist):
    
    caffe_root = envlist['caffe_root']
    model = envlist['caffe_model']
    deploy = envlist['deploy']
    mean = envlist['mean']
    sfile = envlist['imagefile']
    
    reference_model = os.path.join(caffe_root,deploy)
    reference_pretrained = os.path.join(caffe_root,model)
    imagenet_mean = os.path.join(caffe_root,mean)
    
    print(imagenet_mean)
    mean_ = np.load(imagenet_mean)
    print mean_.shape

    net = caffe.Classifier(
          reference_model,
          reference_pretrained,
          mean=mean_,
          #channel_swap=(2, 1, 0),
          raw_scale=255)
    
    input_image = caffe.io.load_image(sfile)

    output = net.predict([input_image])
    predictions = output[0]
    predicted_class_index = predictions.argmax()
    
    print predicted_class_index
    print np.argpartition(predictions, -3)[-3:]
    


def setNet(envlist,test=False):
    
    caffe_root = envlist['caffe_root']
    model = envlist['caffe_model']
    deploy = envlist['deploy']
    #sfile = envlist['imagefile'] # after loading image with caffe io load_image
    
    reference_model = os.path.join(caffe_root,deploy)
    reference_pretrained = os.path.join(caffe_root,model)

    if test:
        print reference_model
        print reference_pretrained
        #plt.imshow(sfile)
        return
        
    #mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
    #f = open(mean_file, 'rb')
    #mean_blobproto_new.ParseFromString(f.read())
    #mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
    #f.close()
    
    caffe.set_mode_cpu()
    net = caffe.Net(reference_model,      # defines the structure of the model
                reference_pretrained,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)    

    return net    
    

def process(net,envlist,test=False):
    
    caffe_root = envlist['caffe_root']
    meannpy = envlist['meannpy']
    
    #sfile = envlist['imagefile'] # after loading image with caffe io load_image
    uint8_image = envlist['uint8_image']    
    #uint8_image = skimage.img_as_ubyte(sfile).astype(np.uint8)
        
    #mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
    #f = open(mean_file, 'rb')
    #mean_blobproto_new.ParseFromString(f.read())
    #mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
    #f.close()

    mean_npy = os.path.join(caffe_root,meannpy)
    # load the mean npy file
    mu_ = np.load(mean_npy)
    #if (mu == mean_image[0]).all():
    #    print "exact same mu and mean_image"
    
     
    uint8_image_ = np.transpose(uint8_image,(2,0,1)) 
    out = net.forward_all(data=np.asarray([uint8_image_]) - mu_)
    plabel = int(out['loss'][0].argmax(axis=0))
    if test:    
        mu = mu_.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
        print 'mean-subtracted values:', zip('BGR', mu)
        print "uint8 shape", uint8_image.shape
        print "mu_ shape", mu_.shape
        print "predicted label (uint8 - mean)", plabel

        # create transformer for the input called 'data'
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

        transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
        transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

        #image = caffe.io.load_image(caffe_root + 'exa')
        sfile = skimage.img_as_float(uint8_image).astype(np.float32)
        transformed_image = transformer.preprocess('data', sfile)
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image

        ### perform classification
        output = net.forward()
        output_prob_coarse = output['loss'][0]  # the output probability vector for the first image in the batch
        print 'predicted class (transformer):', output_prob_coarse.argmax()

    return plabel

def cifarh5Test(h5_filedir,test=False):
    
    
    envlist = setEnvCifar100()
    net = setNet(envlist)

    conf_mat = np.zeros((100,100))
    with h5py.File(h5_filedir, 'r') as hf:
        images = hf.get('data')
        label_fines = hf.get('label_fine')
        label_coarses = hf.get('label_coarse')

        label_num = len(label_fines)
        print "total label length", label_num
        
        correct = 0
        x = range(label_num)
        np.random.shuffle(x)
        for i,idx in enumerate(x):        
            img = images[idx]
            im= np.transpose(img, (1,2,0))
            
            label_fine = label_fines[idx]
            if test:
                print "-- fine label %d : coarse %d  " % (label_fine,label_coarses[idx])
                fig = plt.figure()
                plt.imshow(im)
        
            envlist['uint8_image'] = im
            predict=process(net,envlist,test)

            iscorrect = label_fine == predict
            correct += (1 if iscorrect else 0)
        
            conf_mat[label_fine,predict] += 1

            if not test and i % (1000 == 0):
                print i
            if test and i > 20:
                break
            if (i % 100) == 0:
                print "loop:%d = accurancy : %.5f" % (i,np.float32(correct / ( i + 1.)))
                
    
    print "accurancy : %.5f" % np.float32(correct / ( i + 1.))
    print "confusion matrix"
    print conf_mat

    
    
def cifarLmdbTest(lmdb_dir,test=False):
    
    envlist = setEnv()    
    net = setNet(envlist)

    category = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','track']
    
    
    lmdb_env = lmdb.open(lmdb_dir, readonly=True)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()    

    labels_ = []
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        labels_.append(datum.label)
    
    print "lmdb - lables length:",len(set(labels_))
    print "lmdb - total length:", len(labels_)
    
    x = np.arange(len(labels_))
    np.random.shuffle(x)
    print "after shuffle", x
    
    conf_mat = np.zeros((10,10))

    correct = 0    
    for i,idx in enumerate(x):    
    
        mykey = '%0*d' % (5,idx)
        value = lmdb_txn.get(mykey)
        datum.ParseFromString(value)
        #label = datum.label
        data = caffe.io.datum_to_array(datum)
        label_ = datum.label
        im = data.astype(np.uint8)
        #print "lmdb image shape", im.shape
        
        #image stored on lmdb --> channel , col , row 
        im = np.transpose(im, (1,2,0)) # original (dim, col, row)
        if test:
            print "key val %s : Label %s, %s" % (idx, label_ ,category[label_])
            plt.figure()            
            plt.imshow(im)
        
        envlist['uint8_image'] = im
        
        predict=process(net,envlist)
    
        iscorrect = label_ == predict
        correct += (1 if iscorrect else 0)
        
        conf_mat[label_,predict] += 1
        
        if test and i > 5:
            break
        #fig = plt.figure()
        #plt.imshow(im)
        #sfile="/Users/donchan//caffe/caffe/examples/cifar10/test_.jpg"
        if test and (i % 10000) == 0:
            print "accurancy : %.4f", np.float32(correct / ( i + 1.))
            print conf_mat
    
    print "accurancy : %.5f" % np.float32(correct / ( i + 1.))
    print conf_mat
    
    #plt.imsave(sfile,im)
    #im0 = caffe.io.load_image(sfile)

    #return (im0,datum.label)


def setEnv():

    envlist = {}
    envlist['caffe_root'] = '/Users/donchan/caffe/caffe/examples/cifar10/quick_'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    envlist['caffe_model'] = 'cifar10_quick_iter_5000.caffemodel.h5'
    envlist['deploy'] = 'deploy.prototxt'
    envlist['meannpy'] = 'mean.npy'
    envlist['meanfile'] = '/Users/donchan/caffe/caffe/examples/cifar10/mean.binaryproto'
    #envlist['imagefile'] = lmdb_img # after loading image with caffe io load_image

    return envlist
    
def setEnvCifar100():

    envlist = {}
    envlist['caffe_root'] = '/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    envlist['caffe_model'] = 'cnn_snapshot_iter_150000.caffemodel'
    envlist['deploy'] = 'cnn_deploy.prototxt'
    envlist['meannpy'] = 'mean.npy'
    envlist['meanfile'] =  '/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/mean.binaryproto'
    #envlist['imagefile'] = lmdb_img # after loading image with caffe io load_image

    return envlist

def cifar10Test():
    
    lmdb_dir = "/Users/donchan/caffe/caffe/examples/cifar10/cifar10_train_lmdb"
    cifarLmdbTest(lmdb_dir)

    #category = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','track']
    #print "correct label : %s" % category[label]

    #process(net,envlist)

def cifar100Test():
    
    h5_filedir="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/train.h5"
    cifarh5Test(h5_filedir)

def main():

    # test for quick trained     
    #cifar10Test()
    
    
    cifar100Test()
    
    
    


if __name__ == "__main__":
    main()