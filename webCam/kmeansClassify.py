# -*- coding: utf-8 -*-

import numpy as np

from sklearn import decomposition, svm

from sklearn.cross_validation import train_test_split

from sklearn.cluster import KMeans

from time import time

class kmeansClassify(object):
    
    
    def __init__(self,test=False):
        
        self.test = test
        
    def pca(self,X):
        
        X_train,X_test,y_train,y_test = self.dataSplitter(X)
        
        x, xt = self.decomposition(X_train,X_test)

        return x, xt
                

    def dataSplitter(self,X):
        
        y = range(len(X))
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
        
        return X_train,X_test,y_train,y_test

    def decomposition(self,X_train,X_test):
        
        pca = decomposition.RandomizedPCA(n_components=150, whiten=True)

        pca.fit(X_train)
        
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
    
        return X_train_pca,X_test_pca
        
    def Kmeans(self,X,n_clusters=5):
                
        X_train,X_test,y_train,y_test = self.dataSplitter(X)
        
        print("-- Fitting model on a small sub-sample of the data")
        t0 = time()
        #image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=n_clusters, random_state=64).fit(X_train)
        print("-- done in %0.3fs." % (time() - t0))
        
        labels_test = kmeans.predict(X_test)
        
        print "-- top 10 labels: ", labels_test[:10] 

        return X_test,labels_test
        