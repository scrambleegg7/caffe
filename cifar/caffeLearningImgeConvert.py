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

def testImageLoad():
    
    
    
    h5_filedir="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/train.h5"
    x = np.arange(50000)
    #np.random.shuffle(x)
    print "after shuffle", x
    
    SIZE = 32
    with h5py.File(h5_filedir, 'r') as hf:
        images = hf.get('data')
        label_fines = hf.get('label_fine')
        label_coarses = hf.get('label_coarse')

        i = x[0]
        img = images[i]
        print "cifar image reshape to 32 x 32 x 3"         
        im0 = np.transpose(img, (1,2,0))
        print "fine %d : coarse %d  " % (label_fines[i],label_coarses[i])
        print "cifar image", im0[:3,:3,:]    
        
        
        #img[:,:,0] = numpy.ones([5,5])*64/255.0
        #img[:,:,1] = numpy.ones([5,5])*128/255.0
        #img[:,:,2] = numpy.ones([5,5])*192/255.0

        fig = plt.figure()        
        plt.imshow(im0)
        
        
        sfile="/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/test.jpg"
        plt.imsave(sfile,im0)
        #cv2.imwrite(sfile, im0)

    

    caffe_image = caffe.io.load_image(sfile)
    opencv_img = cv2.imread(sfile)
    print "shape with caffe load_image", caffe_image.shape
#    print "caffe load image",caffe_image[:3,:3,:]    
#    print "cv2 imread", opencv_img[:3,:3,:]   
    
    
    # rescaling 0 1 to 0 255
    mn = np.min(caffe_image)
    mx = np.max(caffe_image)
    output = np.uint8((caffe_image - mn)*255/(mx - mn))    
#    print "after normalize image of caffe", output[:3,:3,:]    
    
    
    #opencv_img = cv2.imread('image.jpg')
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
#    print "cv2 convert BGR to RGB", opencv_img[:3,:3,:]   
    opencv_img = skimage.img_as_float(opencv_img).astype(np.float32)
    print "skimage as float -> float32", opencv_img[:3,:3,:]
    fig = plt.figure()    
    plt.imshow(opencv_img)
    
    back_to_original1 = skimage.img_as_ubyte(opencv_img).astype(np.uint8)
    back_to_original2 = skimage.img_as_ubyte(caffe_image).astype(np.uint8)
    
    if (back_to_original1 == back_to_original2).all():
        print "exact same"
    #input_image *= 255
    #print input_image[:3,:3,:]   
    #plt.imshow(input_image)    
    return im0    
    


def main():
    img = testImageLoad()
    #process(img)
    


if __name__ == "__main__":
    main()