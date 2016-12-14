# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import numpy as np
import os

from caffeBase.envParam import envParamAlz
from AlzheimerClass import AlzheimerClass

import matplotlib.pyplot as plt



def plot_gallary(images,labels):
    
    plt.figure(  figsize = (8,6))

    plt.suptitle("Alzheimer desease MRI:",size=14)
    plt.suptitle(labels[0])
    
    
    for i, (comp,label) in enumerate(    zip(images,labels)   ):
        plt.subplot(8,8,i+1)
        #print comp.shape
        plt.imshow(comp, cmap='Greys',  interpolation='nearest')
        #print "-- max / min of image : ", np.max(comp),np.min(comp)
        
        plt.xticks(())
        plt.yticks(())
        
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.001, hspace=0)    
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
    im1 = im[:,:,1]
    print " (testimage) max min avg.:", np.max(im1),np.min(im1),np.mean(im1)
    im2 = im[:,:,2]
    print " (testimage) max min avg.:", np.max(im2),np.min(im2),np.mean(im2)
    
    
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
    for efilename in filelists[:5]:
        item=os.path.join(traindir,efilename)
        im = plt.imread(item)
        print "-- shape of image :" , im.shape
        plt.title(efilename)
        #plt.imshow(im,cmap='Greys',  interpolation='nearest')
        plt.imshow(im)
        plt.show()
      

def main():

    '''
    This is the main function
    '''
    # tdir = "/Users/donchan/Downloads/NNData/iris_caffe/"
    #readH5Data()
    readImage()
    #readTest3chImage()
    
    #
    """
    alzCls = AlzheimerClass(None,True)
    images = alzCls.getImages()
    labels = alzCls.getLables()
    
    myimages = images[:62]
    mylabels = labels[:62]
    
    plot_gallary(myimages,mylabels)
    """

if __name__ == "__main__":
    main()
    #cProfile.run('main()') # if you want to do some profiling
    
    
