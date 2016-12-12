# -*- coding: utf-8 -*-

import numpy as np
import os

from caffe.proto import caffe_pb2

import caffe

import matplotlib.pyplot as plt

import cv2

import skimage


class imageProcessClass(object):
    
    def __init__(self,test=False):
        self.image = None
        #pass
        self.test = test
        
        self.splitImageXY = None
        
        self.saveFileNames = []
        
    def getSavedFileNames(self):
        
        return self.saveFileNames
        
    def rgb2gray(self,rgb):
        
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    def convertToCHW(self,myimage,rgb=False):
    
        image = myimage.copy()

        if rgb:        
            image = image[::-1]               # BGR -> RGB
        image = np.transpose(image,(1,2,0))
    
        #print "by 2 x 2", meanarray[:2,:2,:]
    
        #meanarray = meanarray.astype(np.uint8)
        #image = image + meanarray

        # clamp values in [0, 255]
        image[image < 0], image[image > 255] = 0, 255

        # round and cast from float32 to uint8
        image = np.round(image)
        image = np.require(image, dtype=np.uint8)

        return image

    def convertToCHW_Mean(self,myimage,meanarray,rgb=False):
    
        image = myimage.copy()

        if rgb:        
            image = image[::-1]               # BGR -> RGB
        image = np.transpose(image,(1,2,0))
    
        #print "by 2 x 2", meanarray[:2,:2,:]
    
        meanarray = meanarray.astype(np.uint8)
        image = image + meanarray

        # clamp values in [0, 255]
        image[image < 0], image[image > 255] = 0, 255

        # round and cast from float32 to uint8
        image = np.round(image)
        image = np.require(image, dtype=np.uint8)

        return image


    def meanArray(self,env):
    
    # read mean binary image 
        meanfile = os.path.join(env.envlist['rootdir'],env.envlist['meanfile'])
        
        mean_blob = caffe_pb2.BlobProto()
        with open(meanfile) as f:
            mean_blob.ParseFromString(f.read())
        
        print "---- mean blob data shape:" , np.asarray(mean_blob.data,dtype=np.float32).shape 
        print "---- mean blob h w c num", mean_blob.height,mean_blob.width,mean_blob.channels,mean_blob.num
        #print mean_blob.data
        # convert array into h x w x channel
        mean_array_new = caffe.io.blobproto_to_array( mean_blob )
        print "--- mean_array_new shape with blobproto_to_array", mean_array_new[0].shape
        return mean_array_new[0]
    
    def meanbinary(self,env):
    
    # read mean binary image 
        meanfile = os.path.join(env.envlist['rootdir'],env.envlist['meanfile'])
        
        mean_blob = caffe_pb2.BlobProto()
        with open(meanfile) as f:
            mean_blob.ParseFromString(f.read())
        
        print "---- mean blob data shape:" , np.asarray(mean_blob.data,dtype=np.float32).shape 
        print "---- mean blob h w c num", mean_blob.height,mean_blob.width,mean_blob.channels,mean_blob.num
        #print mean_blob.data
        # convert array into h x w x channel
        mean_array_new = caffe.io.blobproto_to_array( mean_blob )
        print "--- mean_array_new shape with blobproto_to_array", mean_array_new.shape

        mean_array = np.transpose( mean_array_new[0],(1,2,0) ) 
        #mean_array = np.asarray(
        #    mean_blob.data,
        #    dtype=np.float32).reshape(
        #    (mean_blob.height,
        #     mean_blob.width,mean_blob.channels ))

        return mean_array
    
    def convertBGR(self,image):
        
        return image[:,:,::-1]
        
    def readCaffeImage(self,imagefile):
        
        caffe_image = caffe.io.load_image(imagefile)
        imagefile = skimage.img_as_ubyte(caffe_image).astype(np.uint8)

        return caffe.io.load_image(imagefile)
        
    def readPlotImage(self,imagefile):
        
        return plt.imread(imagefile)
        
    def splitterWithSqure(self,imagefile,rectsize_=256):
        
#        
#        rectangle size is 256 in default
#        
        #img = self.readCaffeImage(imagefile)
        img = self.readPlotImage(imagefile)
        
        
        # caffe reader is same as matplotlib imread        
        height_, width_, ch = img.shape

        # center of image 256 256 rectange    
        starty_ = height_ / 2 - (rectsize_ / 2)
        startx_ = width_ / 2 - (rectsize_ / 2)

        endx_ = startx_ + rectsize_
        endy_ = starty_ + rectsize_

        #startxy_ = (startx_, starty_)
        #endxy_ = (endx_, endy_)

        b = int(startx_ / rectsize_)
        c = int(starty_ / rectsize_)
    
        if self.test:
            print "** number of squres splitted with rectsize -- %d x %d" % (b * 2+ 1, c * 2 +1)
    
        leftx = startx_ - b * rectsize_
        topy = starty_ - c * rectsize_
    
        if self.test:        
            print "leftx topy", leftx,topy
    
        rightx = endx_ + b * rectsize_
        bottomy = endy_ + c * rectsize_
    

        if self.test:        
            print "rightx bottomy", rightx,bottomy

        i = 0
        splitImages = []
        for hx in range( 2 * b + 1 ):
        
            sx = leftx + hx * rectsize_
            ex = sx + rectsize_        
        
            for wy in range( 2 * c + 1 ):
            
                sy = topy + wy * rectsize_
                ey = sy + rectsize_
            
                
                split = [ int(sx),int(sy),int(ex),int(ey) ]
                
                #cv2.rectangle(img,(int(sx),int(sy)),(int(ex),int(ey)),(0,0,255),2)
            
                #dst = img[sy:ey,sx:ex]

                #dst = self.convertBGR(dst)
                #dst = dst[:,:,::-1]            
                splitImages.append(split)
        
        self.splitImageXY = splitImages
        
        return splitImages


    def getSplitImages(self,imagefile,bw=False):
        
        splits = self.splitterWithSqure(imagefile)
        img = self.readPlotImage(imagefile)
        
        if bw:
            img = self.rgb2gray(img)
            
        # /aaaa/bbbb/ccc/image.jpg --> /aaa/bbbb/ccc/image + jpb
        # /aaa/bbbb/ccc/image ---> image 
        #nameOfFilename = imagefile.split(".")
        #name  = nameOfFilename[0].split("/")[-1]

        images = []
        for rect in splits:
            
            sx = rect[0]
            sy = rect[1]
            ex = rect[2]
            ey = rect[3]
        
            dst = img[sy:ey,sx:ex]

            images.append(dst)
            
        return images
        
    def getNameOfImageFile(self,imagefile):

        # /aaaa/bbbb/ccc/image.jpg --> /aaa/bbbb/ccc/image + jpb
        # /aaa/bbbb/ccc/image ---> image 

        nameOfFilename = imagefile.split(".")
        name  = nameOfFilename[0].split("/")[-1]
        return name        
        
    def saveSplitImages(self,imagefile,saveDir):
        
        splits = self.splitterWithSqure(imagefile)
        img = self.readPlotImage(imagefile)

        # /aaaa/bbbb/ccc/image.jpg --> /aaa/bbbb/ccc/image + jpb
        # /aaa/bbbb/ccc/image ---> image 
        nameOfFilename = imagefile.split(".")
        name  = nameOfFilename[0].split("/")[-1]

        i = 0        
        for rect in splits:
            
            sx = rect[0]
            sy = rect[1]
            ex = rect[2]
            ey = rect[3]
        
            dst = img[sy:ey,sx:ex]

            dst =self.convertBGR(dst)        
            
            filename = name + "_" + str(i) + ".jpg"
            self.saveFileNames.append(filename)            
               
            upath = os.path.join(saveDir,filename)
            
            cv2.imwrite(upath,dst)
            
            
            i += 1            
            
    def drawSplitImagesOnImage(self,imagefile):
        
        splits = self.splitterWithSqure(imagefile)
        
        im = self.readPlotImage(imagefile)
        for rect in splits:
            
            sx = rect[0]
            sy = rect[1]
            ex = rect[2]
            ey = rect[3]
            
            cv2.rectangle(im,(int(sx),int(sy)),(int(ex),int(ey)),(0,0,255),2)


        plt.imshow(im)
        plt.show()
                        
    def drawSplitImagesOnBWImage(self,imagefile):
                
        splits = self.splitterWithSqure(imagefile)
        
        im = self.readPlotImage(imagefile)
        im = self.rgb2gray(im)
        for rect in splits:
            
            sx = rect[0]
            sy = rect[1]
            ex = rect[2]
            ey = rect[3]
            
            cv2.rectangle(im,(int(sx),int(sy)),(int(ex),int(ey)),(0,0,255),2)
            
        plt.imshow(im , cmap = plt.get_cmap('gray'))
        plt.show()

        

        

        
        
        

