# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

from sklearn import mixture

from webCam.kmeansClassify import kmeansClassify

def gibbs(N=20000,thin=500):
    x=0
    y=0
    xsample_ = []
    ysample_ = []
    print "Iter  x  y"
    for i in range(N):
        for j in range(thin):
            
            shape_ , scale_ = 3, 1.0/(y*y+4)
            x = np.random.gamma(shape_,scale_)
            #x=random.gammavariate(3,1.0/(y*y+4))
            y=np.random.normal(1.0/(x+1),1.0/np.sqrt(x+1))
        
            xsample_.append(x)
            ysample_.append(y)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(xsample_,ysample_,"ro")
    plt.show()
     


def sdNorm(x):
    
    return norm.pdf(x)
    

def process():

    # Construct dataset
    X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=4,
                                 n_classes=2, random_state=1)
    X2, y2 = make_gaussian_quantiles(mean=(3, 3, 3, 3), cov=1.5,
                                 n_samples=300, n_features=4,
                                 n_classes=2, random_state=1)
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, - y2 + 1))

    print "-- test data shape X,y", X.shape,y.shape
    # Create and fit an AdaBoosted decision tree
    print "== start AdaBoost =="    
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

    bdt.fit(X, y)
    
    y_bdtpred = bdt.predict(X)
    
    classif_rate = np.mean(y_bdtpred.ravel() == y.ravel()) * 100
    print("classif_rate for  %f " % (classif_rate))
    
    print "== start GMM =="    
    mygmm = mixture.GMM(n_components=2)
    mygmm.fit(X)
    
    y_predict = mygmm.predict(X)
    #print y_predict
    #print y
    
    print "-- GMM accuracy: ", np.mean(y.ravel() == y_predict.ravel()) * 100



    print "== start Kmeans"
    kmeans = kmeansClassify()
    #x_test, y_label = kmeans.Kmeans(X,2)
    print "-- kmeans accuracy: ", np.mean(y.ravel() == y_label.ravel()) * 100
    

def main():
    process()
    
if __name__ == '__main__':
    main()