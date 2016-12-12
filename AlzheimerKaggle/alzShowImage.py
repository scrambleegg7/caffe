# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import numpy as np
import os

from caffeBase.envParam import envParamAlz
from AlzheimerClass import AlzheimerClass

import matplotlib.pyplot as plt



def plot_gallary(images):
    
    plt.figure(  figsize = (4,3))

    plt.suptitle("Alzheimer desease MRI:",size=14)
    
    
    for i, comp in enumerate(images):
        plt.subplot(4,3,i+1)
        #print comp.shape
        plt.imshow(comp, cmap='Greys',  interpolation='nearest')
        print "-- max / min of image : ", np.max(comp),np.min(comp)
        
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(.01, .05, .99, .93, .04, 0.)
    plt.show()
    


def readH5Data():
    
    
    myenv = envParamAlz()   

    AlzCls = AlzheimerClass(myenv)


    images , labels = AlzCls.readH5Data()
    
    # top 12 from images
    
    images_ = images[:12]
    labels_ = labels[:12]
    
    print images_.shape
    l,c,h,w = images_.shape

    myimages = np.zeros((l,h,w))
    
    for idx,im in enumerate(images_):
        print " ** shape per image " , im.shape,labels_[idx] 
        myimages[idx] = np.reshape(im,(h,w))
        
    
    plot_gallary(myimages)


def readTest3chImage():

    myenv = envParamAlz()   

    item = myenv.envlist['datadir'] + '/mytest3ch.jpg'  
    im = plt.imread(item)
    print im.shape
    #plt.imshow(im,cmap='Greys',  interpolation='nearest')
    plt.imshow(im)
    plt.show()
    plt.hist(im[:,:,0])
    plt.show()
    
    im0 = im[:,:,0]
    print " (testimage) max min avg.:", np.max(im0),np.min(im0),np.mean(im0)
    im = im[:,:,1]
    print " (testimage) max min avg.:", np.max(im),np.min(im),np.mean(im)
    im = im[:,:,2]
    print " (testimage) max min avg.:", np.max(im),np.min(im),np.mean(im)
    
    
    item = myenv.envlist['datadir'] + '/mytest1ch.jpg'  
    im = plt.imread(item)
    plt.imshow(im,cmap='Greys',  interpolation='nearest')
    plt.show()
    plt.hist(im)    
    #plt.imshow(im)
    plt.show()

    
def readImage():

    myenv = envParamAlz()   

    traindir = myenv.envlist['datadir'] + '/alz_train_images'  

    filelists = os.listdir(traindir)
    for efilename in filelists[:10]:
        item=os.path.join(traindir,efilename)
        im = plt.imread(item)
        plt.title(efilename)
        plt.imshow(im,cmap='Greys',  interpolation='nearest')
        plt.show()
      

def main():

    '''
    This is the main function
    '''
    # tdir = "/Users/donchan/Downloads/NNData/iris_caffe/"
    #readH5Data()
    #  readImage()
    readTest3chImage()
    

if __name__ == "__main__":
    main()
    #cProfile.run('main()') # if you want to do some profiling
    
    
