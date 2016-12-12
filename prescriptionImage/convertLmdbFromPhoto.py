import os
import glob
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import patches

from scipy import ndimage
from time import time

from sklearn import decomposition
from sklearn.cluster import KMeans

from PIL import Image

from skimage.feature import match_template
import lmdb
import caffe

def createLmdbFile(images,test=False):
    
    #lmdb_dir = "C:\\VBox\\Prescription"
    if test:
        lmdb_dir="/Users/donchan/Documents/Statistical_Mechanics/caffe/prescriptionImage/testlmdb"
    else:
        lmdb_dir="/Users/donchan/Documents/Statistical_Mechanics/caffe/prescriptionImage/mylmdb"
        
    X = np.array(images)
    print X.shape
    map_size = X.nbytes * 10
    lmdb_env = lmdb.open(lmdb_dir,map_size=map_size)

    N = len(images)
    with lmdb_env.begin(write=True) as txn:
    # txn is a Transaction object
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(i)
            str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

def imresize(im,l,w):

    pilImage = Image.fromarray(np.uint8(im))
    resize_ = pilImage.resize( (l,w)  )
    
    return np.array(resize_)

def readImage(mydir,test=False):
    
    mydir = mydir + "/" + "*.JPG"
    
    myImages = []
    files = glob.glob(mydir)
    for idx,f in enumerate(files):
        print f
        myImage = plt.imread(f)
        l,w,ch = myImage.shape
        if l > w:
            print "l w before rotate", l,w
            myImage = ndimage.rotate(myImage,90)
            l,w,ch = myImage.shape    
            print "l w after rotate", l,w
            
        myImage = imresize(myImage, int(l/70), int(w/70))
        
        myImage = np.transpose(myImage,(2,0,1))
        
        # for confirmation of image transpose
        #im = np.transpose(myImage,(1,2,0))
        #fig = plt.figure()
        #plt.imshow(im)        
        
        myImages.append(myImage)
        
        if idx > 10 and test == True:
            break
        
    
    return myImages

def main():
    
    #mydir = "L:\\Prescription_Photo\\20151102"
    mydir = "/Volumes/myShare/Prescription_Photo/20151102"
    test = False    
    im = readImage(mydir,test)
    createLmdbFile(im,test)


    test = True    
    im = readImage(mydir,test)
    createLmdbFile(im,test)

    
if __name__ == "__main__":
    main()
