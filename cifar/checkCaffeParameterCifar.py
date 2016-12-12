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

def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    #image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image


def process():
    
    
    pass

def setEnv():

    envlist = {}
    envlist['caffe_root'] = '/Users/donchan/caffe/caffe/examples/cifar10/quick_'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    envlist['caffe_model'] = 'cifar10_quick_iter_5000.caffemodel.h5'
    envlist['deploy'] = 'deploy.prototxt'
    envlist['meannpy'] = 'mean.npy'
    envlist['solver'] = 'cifar10_quick_solver.prototxt'
    envlist['meanfile'] = '/Users/donchan/caffe/caffe/examples/cifar10/mean.binaryproto'
    #envlist['imagefile'] = lmdb_img # after loading image with caffe io load_image

    return envlist
    
    
def setEnvCifar100():

    envlist = {}
    envlist['caffe_root'] = '/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    envlist['caffe_model'] = 'cnn_snapshot_iter_150000.caffemodel'
    envlist['deploy'] = 'cnn_deploy.prototxt'
    envlist['meannpy'] = 'mean.npy'
    envlist['solver'] = 'cnn_solver_h5.prototxt'
    envlist['meanfile'] =  '/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/mean.binaryproto'
    #envlist['imagefile'] = lmdb_img # after loading image with caffe io load_image

    return envlist

def main():


    env = setEnvCifar100()
    solver_proto = os.path.join(env['caffe_root'],env['solver'])
    print solver_proto
    #caffe.set_device(0)
    #caffe.set_mode_cpu()

    ### load the solver and create train and test nets
    solver = None  # ignore 
    solver = caffe.SGDSolver(solver_proto)


    # each output is (batch size, feature dim, spatial dim)
    for k, v in solver.net.blobs.items():
        print k, v.data.shape

    # just print the weight sizes (we'll omit the biases)
    for k, v in solver.net.params.items():
        print k, v[0].data.shape
        
    solver.net.forward()  # train net
    print solver.test_nets[0].forward()  # test net (there can be more than one)
    
    im = solver.net.blobs['data'].data[0]
    
    im = deprocess_net_image(im)
    #im = im.transpose(1, 2, 0)
    
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #print "cv2 convert BGR to RGB", im[:3,:3,:]   
    #im = skimage.img_as_float(im).astype(np.float32)

    #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    print "data shape", solver.net.blobs['data'].data[0].shape 
    fig = plt.figure()    
    plt.imshow(im)
    
    print 'train labels:', solver.net.blobs['label_fine'].data[:1]

    # test net
    imt = solver.test_nets[0].blobs['data'].data[0].transpose(1, 2, 0)
    fig = plt.figure()    
    plt.imshow(imt)
    
    print solver.net.params['conv1'][0].diff[:,0].shape
    
if __name__ == "__main__":
    main()