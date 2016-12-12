# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

"""
Created on Wed Jul  6 09:24:20 2016

@author: donchan
"""

import cv2

import matplotlib.pyplot as plt

import os

import numpy as np

from caffeBase.myDirClass import myDirClass

from caffeBase.imageProcessClass import imageProcessClass


from kmeansClassify import kmeansClassify


def process2():
    
    myDir = myDirClass()
    
    photoDir = "/Volumes/myShare/Prescription_Photo"
    saveDir = "/Volumes/home/PC_Backup/caffe/presciption"

    myDir.getDirs(photoDir)
    
    myDir.getFiles( os.path.join( photoDir, myDir.getLatestDir() ) , "JPG"   )


    latestDir = myDir.getLatestDir()
    print "-- latest dir", latestDir

    imageProcessCls = imageProcessClass()
    
    saveDir = os.path.join(saveDir,latestDir)
    myDir.makeDirDelIfExist(saveDir)
    rootDir = saveDir

    print "top of file list : ", myDir.getFileList()[0]    

    i=0
    images = []    
    for imagefile in myDir.getFileList():

        print "processing .... ", imagefile

        #splitImages = imageProcessCls.splitterWithSqure(imagefile)

        blackWhite = False
        s_images = imageProcessCls.getSplitImages(imagefile,blackWhite)   
        
        print "-- shape ", s_images[0].shape
        

        for im in s_images:    
            images.append(im.ravel())
            
            
    # convert numpy array from list
    X = np.array(images)
    
    print "-- total size of X from images --", X.shape            
    Kmeans = kmeansClassify()
    x_test,test_label = Kmeans.Kmeans(X,10)
        
        #
        #myimage = x_test[0].reshape(256,256)
        #plt.imshow(myimage)
        #plt.show()
        #
        
    print "x test and labels",len(x_test),len(test_label)        
    for label_ in set(test_label):
        saveDir = os.path.join(rootDir,str(label_))
        myDir.makeDirDelIfExist(saveDir)
        
    i = 0
    for im , label_ in zip(x_test,test_label):
            
        im =  im.reshape(256,256,3)
        saveDir = os.path.join(rootDir,str(label_))
        
        filename = os.path.join(saveDir,"test_" + str(i) + ".jpg") 
        cv2.imwrite(filename,im)
        i += 1
        
        
    
def main():
    process2()




if __name__ == "__main__":
    main()
