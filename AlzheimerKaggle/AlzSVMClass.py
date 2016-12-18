# -*- coding: utf-8 -*-



from AlzBaseClass import AlzBaseClass
from AlzheimerClass import AlzheimerClass
from caffeBase.envParam import envParamAlz

from sklearn.cross_validation import train_test_split

import numpy as np
import collections

from sklearn import svm
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt

import itertools

class AlzSVMClass(AlzheimerClass):

    def __init__(self,env=None,read=False,test=False):
                
        super(AlzSVMClass,self).__init__(env,read,test)
        
        

    def pickupImages(self,size=12400):
        
        images = np.zeros( (size,96,96) )
        labels = np.zeros(size)        
        
        images = self.images[:size]
        labels = self.labels[:size]
        
        l,h,w = images.shape
        
        images = np.reshape(images,(l,-1))
        
        print images.shape,labels.shape
    
        collect_dict = collections.Counter(labels)
        diag_keys = set(collect_dict.keys())

        
        vlists = []
        for k,v in collect_dict.items():
            print k, v
            vlists.append(v)
        
        idx = np.argmin(vlists)
        min_number = collect_dict[idx+1.]
        
        print "-- minimum counts of label --" , min_number

        mask_1 = labels == 1.
        labels_1 = labels[mask_1][:min_number]
        labels_23 = labels[~mask_1]
        
        labels = np.hstack( ( labels_1,labels_23 ) )
        print "-- new labels size", labels.shape

        images_1 = images[mask_1][:min_number]
        images_23 = images[~mask_1]

        images = np.vstack( ( images_1,images_23 ) )
        print "-- new images size", images.shape

        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.5, random_state=0)
        
        collect_dict = collections.Counter(y_train)
        vlists = []
        for k,v in collect_dict.items():
            print " -- after shuffling : ", k, v
            vlists.append(v)

        """
         use tricky logic to have equal balance each labels 
         for best practices of learning....
        """        
        
        
        
        classifier = svm.SVC(gamma=0.001)

        # We learn the digits on the first half of the digits
        classifier.fit(X_train, y_train)    
        print classifier.score(X_test,y_test)
        
        
        
        y_pred = classifier.predict(X_test)
        
        
        print y_pred
        print y_test
        
        # Compute confusion matrix
        #cnf_matrix = confusion_matrix(y_test, y_pred)
        
        #self.plot_confusion_matrix(cnf_matrix, y_test)
        
        #print np.sum(predict_ == y_train) / np.float(len(X_train))
    def plot_confusion_matrix(self,cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
    
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
    



def main():
    
    myenv = envParamAlz()
    
    alzSvmCls = AlzSVMClass(myenv,True,True)
    alzSvmCls.pickupImages()


if __name__ == "__main__":
    main()