# -*- coding: utf-8 -*-

import numpy as np
import struct

from sklearn.datasets import load_digits
import os

udir = "/Users/donchan/Documents/jdwittenauer_DeepLearning/data"
mnistDir = "/Users/donchan/caffe/caffe/data/mnist"

def load_digitsData():
    
    data = load_digits()
        
    return data    

def digitsXY():
    
    data = load_digitsData()
    
    X = data.images
    y = data.target
    
    return X,y
    
    
def load_ex2data1():
    
    mfile = os.path.join(udir,"ex2data1.txt")
    data = np.genfromtxt(mfile, delimiter=",")
    X = data[:, (0, 1)]
    y = data[:, 2]
    
    return X,y


def loadFullDigitsData():
    
    dfile = os.path.join(mnistDir,'train-images-idx3-ubyte')
    infile = open(dfile,'rb')
    
    header = infile.read( 4 * 4 )
    mn, num, nrow, ncol = struct.unpack('>4i', header) 

    pixel = np.zeros(( num, nrow, ncol))
    npixel = nrow * ncol
    
    for i in range(num):
        
        buf = struct.unpack( '>%dB' % npixel, infile.read( npixel ) )
        pixel[i, :, :] = np.asarray( buf ).reshape( ( nrow, ncol ) )

    infile.close()

    return pixel

def loadFullDigitsLabel():

    dfile = os.path.join(mnistDir,'train-labels-idx1-ubyte')
    infile = open(dfile,'r')
    
    

    ### header (two 4B integers, magic number(2049) & number of items)
    #
    header = infile.read( 8 )
    mn, num = struct.unpack( '>2i', header )  # MSB first (bigendian)
    assert mn == 2049
    #print mn, num

    ### labels (unsigned byte)
    #
    label = np.array( struct.unpack( '>%dB' % num, infile.read() ), dtype = int )

    infile.close()

    return label



def main():
    
    X = loadFullDigitsData()
    print X.shape

    y = loadFullDigitsLabel()
    print y.shape
    
    
    
    
if __name__ == "__main__":
    main()
    