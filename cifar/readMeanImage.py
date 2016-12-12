# -*- coding: utf-8 -*-

import caffe
import numpy as np
import sys

import matplotlib.pyplot as plt



def main():

    mydir = "/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/"
    mydir = "/Users/donchan/caffe/caffe/examples/cifar10"

    mfile = mydir + "/" + "mean.binaryproto"
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( mfile , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    
    im0 = arr[0]
    print im0.shape
    print "average image from cifar 100"
    im0 = np.transpose(im0,(1,2,0))
    plt.imshow(im0)
    plt.show()
    
    #to_load_datum('imagenet_mean.binaryproto')

    
    #out = arr[0]
    #np.save( sys.argv[2] , out )
    
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( mfile , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    outfile = mydir + "/" + "mean.npy"
    np.save( outfile , out )

if __name__ == "__main__":
    main()