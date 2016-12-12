# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

import os

import caffe

from lfw import lfw

from caffeBase.envParam import envParamLFW

from caffeBase.imageProcessClass import imageProcessClass

import sys
import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version



def process():
    
    env = envParamLFW()
    lfwCls = lfw(env) 
    
    facelist = lfwCls.loadFaceTopN(1)
    
    
    top0 = facelist[0]
    
    print "-- target image file", top0 

    return top0
    
def sc():

    imagefile= process()    

    img = plt.imread(imagefile)
    
    imageProcessCls = imageProcessClass()
    img = imageProcessCls.rgb2gray(img)
    
    plt.imshow(img,cmap=plt.cm.gray)
    
    # standarized with 255 to 0-1
    img = img / 255.
    graph = image.img_to_graph(img)    
    print "-- graph shape ", graph.data.shape
    print "-- lfw face image shape", img.shape


    fig = plt.figure(figsize=(8,6))
    #ax1= fig.add_subplot(211)    
    #ax1.plot(graph.data)
    print "-- graph data standarized ", graph.data.std()    
    
    try:
        face = sp.face(gray=True)
    except AttributeError:
        # Newer versions of scipy have face in misc
        from scipy import misc
        face = misc.face(gray=True)
    
    
    # Take a decreasing function of the gradient: an exponential
    # The smaller beta is, the more independent the segmentation is of the
    # actual image. For beta=1, the segmentation is close to a voronoi
    beta = 5
    eps = 1e-6
    graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

    #print "--top10 graph data after exponential", graph.data[:10]
    #ax2= fig.add_subplot(212)    
    #ax2.plot(graph.data)
    
    #plt.plot(graph.data)
    # Apply spectral clustering (this step goes much faster if you have pyamg
    # installed)
    N_REGIONS = 25    
    
    #sys.exit(0)
    
    for assign_labels in ['discretize','kmeans']:
        t0 = time.time()
        labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                 assign_labels=assign_labels, random_state=1)
        t1 = time.time()
        labels = labels.reshape(img.shape)

        plt.figure(figsize=(5, 5))
        plt.imshow(img,cmap=plt.cm.gray)
        for l in range(N_REGIONS):
            plt.contour(labels == l, contours=1,
                    colors=[plt.cm.spectral(l / float(N_REGIONS))])
        plt.xticks(())
        plt.yticks(())
        title = 'Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0))
        print(title)
        plt.title(title) 
    
    plt.show()
    
    #print "face shape", face.shape    
    
def main():
    
    sc()
    
    
if __name__ == "__main__":
    main()