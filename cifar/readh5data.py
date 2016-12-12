# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 
import h5py

import numpy as np
from caffe.proto import caffe_pb2

import caffe

def readAndShow(h5_filename,mean):
    
    
    SIZE = 32
    with h5py.File(h5_filename, 'r') as hf:
        images = hf.get('data')
        label_fines = hf.get('label_sex')
        #label_coarses = hf.get('label_coarse')
    
    
        for i, img in enumerate(images):
        
            print "image shape of h5", img.shape
            # img = img[::-1,:,:]
            img = np.transpose(img, (1,2,0))
            
            #im0 += mean
            #image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
            #image = np.round(image)
            #image = np.require(image, dtype=np.uint8)

            #img+=mean

            
            #print "fine %d : coarse %d  " % (label_fines[i],label_coarses[i])        
            #im0 = img.reshape(SIZE, SIZE, 3)
            fg = plt.figure()
            plt.imshow(img)
            plt.show()

            fg = plt.figure()
            #img-=mean
            #print mean[:2,:2,:]
            plt.imshow(img)
            plt.show()
    
            if i == 0:
                break
    
def readMean(mean_npy):
    
    mean = np.load(mean_npy)
    mean = np.transpose(mean,(1,2,0))
    mean = mean.astype(dtype=np.uint8)
    print mean.shape
    #print mean
    #print "by 3 x 3 from meannpy", mean[:2,:2,:]
    
    meanfile="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/cnn_mean.binaryproto"
    
    mean_blob = caffe_pb2.BlobProto()
    with open(meanfile) as f:
        mean_blob.ParseFromString(f.read())

    mean_array_new = caffe.io.blobproto_to_array( mean_blob )
    print "mean_array_new shape", mean_array_new.shape
    mean_new = np.transpose(mean_array_new[0],(1,2,0))
    mean_new = mean_new.astype(np.uint8)
    #print "by 3 x 3 mean binaryprot", mean_new[:2,:2,:]

    return mean_new
    
def main():
    
    #h5_filedir="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/test.h5"
    h5_filedir="/Users/donchan/caffe/caffe/examples/lfw/train.h5"
    
    mean_npy="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/cnn_mean.npy"
    
    mean = readMean(mean_npy)
    readAndShow(h5_filedir,mean)
    
    
if __name__ == "__main__":
    main()
