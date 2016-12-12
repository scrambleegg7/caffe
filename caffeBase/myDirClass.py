# -*- coding: utf-8 -*-

import os
import glob
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

#import caffe

import cv2

import errno

import shutil


class myDirClass(object):
    
    def __init__(self,test=False):
        
        self.test = test
        
        self.dirs = None
        
        self.files = None
        
    def getDir(self):
        return self.dirs
    
    def getLatestDir(self):
        return self.dirs[-1]
    
    def getFileList(self):
        return self.files
        
    def getLatestFile(self):
        return self.files[-1]
        
    def makeDir(self,path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    def makeDirDelIfExist(self,path):

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path,True)
            
            self.makeDir(path)
            
        
    def getDirs(self,targetDir):
        
        #dirs = glob.glob(targetDir)
        mydirs = os.listdir(targetDir)
        dirs = [d for d in mydirs if os.path.isdir(os.path.join(targetDir,d)) ]
        
        if self.test:
            
            print "num of target dirs:",len(dirs)
            print "print top10 target dirs"
            print [d for d in dirs[:10]]
            
        self.dirs = dirs
    
    def getDirList(self):
        
        # return directory list        
        return self.dirs
        
    def getFiles(self,targetDir,ext=None):
        
        if ext == None:
            ext = "*"
        targetImages = os.path.join(targetDir,"*.%s" % ext)
        
        files = glob.glob(targetImages)
        
        files = [ i for i in files if not os.path.isdir( os.path.join(targetDir, i) ) ]
        
        self.files = files
        
    def saveImagesFile(self):
        
        pass   

    def saveSplitterImageFile(self):
        
        pass
        
    def saveImageFile(self,targetDir,filename,im):

        try:        
            if os.path.isdir(targetDir):
                uDirPath = os.path.join(targetDir,filename)
                cv2.imwrite(uDirPath,im)
            else:
                raise Exception("-- Error: Not Directory --!")
        except Exception as e:
            print e.args
            