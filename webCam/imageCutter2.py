# -*- coding: utf-8 -*-

"""
Created on Wed Jul  6 09:24:20 2016

@author: donchan
"""

import cv2

import matplotlib.pyplot as plt

import os

import numpy as np

import csv

from caffeBase.myDirClass import myDirClass

from caffeBase.imageProcessClass import imageProcessClass


def process2():
    
    myDir = myDirClass()
    
    photoDir = "/Volumes/myShare/Prescription_Photo"
    saveDir = "/Volumes/home/PC_Backup/caffe/presciption/splited"

    myDir.getDirs(photoDir)
    
    
    directories = myDir.getDirList()

    images = []
    i = 0
    for d in directories:
        print "-- reading : %s " % d
    
    
        myDir.getFiles( os.path.join( photoDir, d ) , "JPG"   )


        print "top 10 file list ", myDir.getFileList()[:10]    
        latestDir = d

        imageProcessCls = imageProcessClass()
    
        saveDir2 = os.path.join(saveDir,latestDir)
        myDir.makeDirDelIfExist(saveDir2)
        

        for imagefile in myDir.getFileList():

            #print "processing .... ", imagefile
            imageProcessCls.saveSplitImages(imagefile,saveDir2)
            saveimages = imageProcessCls.getSavedFileNames()
            
            #dd = d.split("/")[-1]
            for im in saveimages:
                image00 = os.path.join(d,im)
                print "-- processing ....",image00
                images.append([image00,i])
        i += 1
        
        if i > 10:
            break


    outfile = os.path.join(saveDir, "imagelist.dat")
    f = open(outfile,'w')
    writer = csv.writer(f,delimiter='\t')
    writer.writerows(images)
        
    
def main():
    process2()




if __name__ == "__main__":
    main()
