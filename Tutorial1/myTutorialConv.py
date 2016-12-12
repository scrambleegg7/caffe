# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
from skimage.io import imread
from skimage import color

from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def process():
    

    convfile = "/Users/donchan/caffe/caffe-public/examples/tutorial1/conv.prototxt"
    imagefile = "/Users/donchan/caffe/caffe-public/examples/tutorial1/cat.jpg"
    cat_model = "/Users/donchan/caffe/caffe-public/examples/tutorial1/cat.caffemodel"
    
    net = caffe.Net(convfile,caffe.TEST)

    print "conv structure information"    
    for k,v in net.blobs.items():
        print k,v.data.shape
    
    for k,v in net.params.items():
        print k, "length of conv parameters:",len(v)
    print type(net.params['conv'][0])
    
    print net.params['conv'][0].data.shape
    print net.params['conv'][1].data.shape
    print "mean of weight filter" , np.mean(net.params['conv'][0].data[0])
    print "std of weight filter", np.std(net.params['conv'][0].data[0]), "close to defined value 0.01"
    print "constant value 0 on bias filter",net.params['conv'][1].data

    print "blobs conv data shape:",net.blobs['conv'].data.shape

    print "read color image from CAT"    
    im = imread(imagefile)
    im0 = caffe.io.load_image(imagefile)
    
    print "color image has 3 channes, therefore h x w x c, c = 3"
    print "shape of imread", im.shape
    print "shape of caffe load image",im0.shape  

    print "read gray image from CAT"   
    # another way to read color image with gray scale
    imgray = color.rgb2gray(imread(imagefile))
    
    
    im = np.array(Image.open(imagefile).convert('L'))
    print "gray shape of image:", im.shape
    print "image type : ",type(im)
    plt.imshow(im,cmap = plt.get_cmap('gray'))
    
    im_input = im[np.newaxis, np.newaxis, :, :]
    print "new shape of Image ......", im_input.shape
    net.blobs['data'].reshape(*im_input.shape)
    print "set image into blobs data of conv ........."
    net.blobs['data'].data[...] = im_input
    
    net.forward()
    
    print "after forward calc, conv data shape", net.blobs["conv"].data.shape
    
    for i in range(net.blobs["conv"].data.shape[1]):
        fig = plt.figure()
        plt.title("image filter of conv : %d" % i)
        plt.imshow(net.blobs["conv"].data[0,i],cmap = plt.get_cmap('gray'))

    net.backward()
    print len(net.blobs)    
    print net.blobs['conv'].diff
    print "weights gradient",net.params['conv'][0].diff
    print "bias gradiant",net.params['conv'][1].diff
    
        
    
    net.save(cat_model)

    
def main():
    process()



if __name__ == "__main__":
    main()