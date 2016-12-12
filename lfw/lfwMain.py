# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import os

import caffe

#from lfw import lfw
from lfwCaffeClass import lfwCaffeClass

from caffeBase.envParam import envParamLFW

from caffeBase.imageProcessClass import imageProcessClass

def process():
    
    env = envParamLFW()
    lfwCaffeCl = lfwCaffeClass(env) 
    
    lfwCaffeCl.predictFromImageDB("label")
    
def classify():
    
    env = envParamLFW()
    lfwCaffeCl = lfwCaffeClass(env) 
    

    imagefile = os.path.join(env.envlist['rootdir'],'nishino_kana.jpeg')
    image = caffe.io.load_image(imagefile)
    print "test image 1 shape", image.shape    
    lfwCaffeCl.classify(image)
    
    imagefile = os.path.join(env.envlist['datadir'],'nishino_kana/nishino_kana-greenhat.png')
    image = caffe.io.load_image(imagefile)
    print "test image 2 shape", image.shape    
    lfwCaffeCl.classify(image)
    
    lfwCaffeCl.netblobAndParamChecker()
    

def styleNet():

    env = envParamLFW()
    lfwCaffeCl = lfwCaffeClass(env) 

    lfwCaffeCl.style_net(train=False, subset='train')    
    
    
def main():
    #process()
    
    classify()
    
    #styleNet()
if __name__ == "__main__":
    main()