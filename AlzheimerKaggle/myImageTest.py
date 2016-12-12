# -*- coding: utf-8 -*-

import caffe

import matplotlib.pyplot as plt

import numpy as np
import os

from caffeBase.envParam import envParamAlz


def main():
    
    #filename = "/Users/donchan/Pictures/IMAG1296.png"
    
    myenv = envParamAlz()

    cropdir = myenv.envlist['datadir'] + "/cropimages"
    
    filename = os.path.join(cropdir,"testimage.jpg")

    #img = caffe.io.load_image(filename, color=False)
    img = plt.imread(filename)
    
    print "-- original image shape", img.shape
    print np.max(img),np.min(img)
    
    
    #grayimg = img[:,:,0]
    #h,w,c = img.shape
     
    #print "-- gray image shape:", grayimg.shape
    
    #grayre = np.reshape(grayimg,(h,w,1))
    #print grayre.shape
    
    #grayimg_ = img[:,:,0]    
    #grayimg_ = np.reshape(grayre,(h,w) )    

    plt.imshow(img,cmap='Greys',  interpolation='nearest')
    
    #plt.imshow(img,cmap=plt.cm.Greys_r)
    plt.show()


    mydir = "/Users/donchan/caffe/caffe/examples/lfw/lfw_funneled/Zahir_Shah"
    filename = os.path.join(mydir,"Zahir_Shah_0001.jpg")
    img = plt.imread(filename)
    
    print "-- original image shape", img.shape
    print np.max(img),np.min(img)
    plt.imshow(img,cmap='Greys',  interpolation='nearest')
    
    #plt.imshow(img,cmap=plt.cm.Greys_r)
    plt.show()


    
    
if __name__ == "__main__":
    main()