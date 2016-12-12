# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import skimage

from caffeBase.imageProcessClass import imageProcessClass
from AlzheimerClass import AlzheimerClass

from caffeBase.envParam import envParamAlz

from AlzImageCrop import OpenCVResizeCrop 


def proc1():
    
 
    myenv = envParamAlz()
    
    read = True
    test = True

    AlzCls = AlzheimerClass(myenv,read,test)
    
    images = AlzCls.getImages()
    myimage = images[0]
    
    """    
    plt.subplot(2,1,1)
    plt.imshow(myimage,cmap='Greys',  interpolation='nearest')
    
    plt.subplot(2,1,2)
    plt.hist(myimage)
    plt.show()
    """
    
    
    h,w = myimage.shape
    #myimage = np.reshape(myimage,(h,w,1))
    print "-- max and min -- ",  np.max(myimage),np.min(myimage)
    print type(myimage)
    print "-- shape of image0: ", myimage.shape
    
    #myimage /= 256    

    myimage *= 255.0/myimage.max()     
    
    cropdir = myenv.envlist['datadir'] + "/cropimages"
    
    output_file = os.path.join(cropdir,"testimage.jpg")
    
    #OpenResizeCls = OpenCVResizeCrop()
    #OpenResizeCls.resize_and_crop_image(myimage,output_file,h)
    
    cv2.imwrite(output_file, myimage)

    #
    #print "-- making Alzheimer H5 ........."    
    #print " lmdb done........."

    return AlzCls
    
def main():
    AlzCls = proc1()
    
    
if __name__ == "__main__":
    main()