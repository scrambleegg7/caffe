# -*- coding: utf-8 -*-

"""

    read Alzheimer data base from ANDI - Kaggle 

    Alz MRI image is NOT normarized.
    Thus, image has to be normalized with 0-255
    
    formula :  myimage *= 255.0/myimage.max()

    image size : 96 x 96 numpy array
    
    
"""

import numpy as np
import os
import cv2
import pandas as pd
import csv
import shutil

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

from caffeBase.envParam import envParamAlz

from caffeBase.myDirClassNocv2 import myDirClass

import platform as pl

from time import time

from imageDataBase.imageDataBase import imageDataBase

import caffe

class AlzheimerClass(object):

    def __init__(self,env=None,read=False,test=False):
        
        if env == None:
            self.env = envParamAlz()
        else:
            self.env = env
        
        self.N = None
        self.X = np.zeros((1,3,96,96))
        self.y = np.zeros(3)
        
        if read:
            print "--[AlzheimerClass]  alz data reading ......"
            self.images, self.validimages = self.readAlzClinicalData()
            self.labels, self.valid_labels = self.readLabels()
        
        if test and read:
            print "--images shape [AlzheimerClass:init] --", self.images.shape
            print "--valid images shape [AlzheimerClass:init] --", self.validimages.shape
            print "--labels shape [AlzheimerClass:init] --", self.labels.shape            
            print "--valid labels shape [AlzheimerClass:init] --", self.valid_labels.shape            
                
    def getImages(self):
        return self.images
    
    def getLables(self):
        return self.labels
                
    def readLabels(self):
    
        if pl.system() == "Linux":
            udir = "/home/hideaki/SynologyNFS/myProgram/pythonProj/KaggleData/Alzheimer"
        else:    
            udir = self.env.envlist['datadir']    
    
        filename = "adni_demographic_master_kaggle.csv"
        fullpath = os.path.join(udir,filename)
    
        demo = pd.read_csv(fullpath)
        print "-- demographic data : \n", demo.head()
    
        trX_subjs = demo[(demo['train_valid_test']==0)]
        trY_diagnosis = np.asarray(trX_subjs.diagnosis)

        vaX_subjs = demo[(demo['train_valid_test']==1)]
        vaY_diagnosis = np.asarray(vaX_subjs.diagnosis)

        train_orig = trY_diagnosis
        valid_orig = vaY_diagnosis    
    
        images_per_sub = 62

        """
        diagnosis x 62 == total images of one Subjects    
        
        """
        trY_all = []
        for n in trY_diagnosis:
            for i in range(images_per_sub):
                trY_all.append(n)

        trY_all = np.asarray(trY_all)
    
        vaY_all = []
        for n in vaY_diagnosis:
            for i in range(images_per_sub):
                vaY_all.append(n)
        
        vaY_all = np.asarray(vaY_all)
        
        print "-- training target :", trY_all,   len(trY_all), list(set(trY_all))
        print "-- validation target :", vaY_all, len(vaY_all), list(set(vaY_all))
        
        """        
        trY_targets = np.zeros((len(trY_all), 3))
        for count, target in enumerate(trY_all):
            trY_targets[count][ target - 1 ]= 1    
        print(trY_targets)
        """
        
        #trainingOneHot =LabelBinarizer().fit_transform(trY_all)
        #validOneHot =LabelBinarizer().fit_transform(vaY_all)
        #print "-- length for traing / valid target flag",  len(trainingOneHot), len(validOneHot)
    
        return trY_all,vaY_all
        #,validOneHot,train_orig,valid_orig
        
    def alzImageData(self):
    
        # trainign directory index
        training_idx = range(1,2)

        myDirCls = myDirClass()
        
        alzDir = self.env.envlist['datadir']
    
        files = []    
        for idx in training_idx:
            print ".... reading imgset_%d : dirctory" % idx
        

            if pl.system() == "Linux":
                udir = "/home/hideaki/SynologyNFS/myProgram/pythonProj/KaggleData/Alzheimer/imgset_%d" % idx
            else:
                udir = alzDir + "/imgset_%d" % idx
            myDirCls.getFiles(udir) 
            trainings = [f for f in myDirCls.getFileList() if f[-3:] == 'npy' ]
        
            files.extend(trainings)

        return files
    
    def alzValidImageData(self):
    
        # trainign directory index
        valid_idx = range(9,11)

        myDirCls = myDirClass()
        
        alzDir = self.env.envlist['datadir']
    
        files = []    
        for idx in valid_idx:
            print ".... reading imgset_%d : dirctory" % idx
        

            if pl.system() == "Linux":
                udir = "/home/hideaki/SynologyNFS/myProgram/pythonProj/KaggleData/Alzheimer/imgset_%d" % idx
            else:
                udir = alzDir + "/imgset_%d" % idx
            myDirCls.getFiles(udir) 
            trainings = [f for f in myDirCls.getFileList() if f[-3:] == 'npy' ]
        
            files.extend(trainings)

        return files

    
    def StackImages(self,files):

        prev_data = np.array([])
        s_image = np.array([])
    
        for idx, f in enumerate(files):
            print "- reading image npy files ...", f
            img_data = np.load(f)
        
            if idx > 0:
                print "-- images stacked No. %d" % (idx+1)
                s_image = np.vstack((prev_data,img_data))
                prev_data = s_image
            else:
                prev_data = img_data
                    
        return s_image    


    def makeAlzLMDB(self,test_size=0.4):

        db = imageDataBase(self.env)
        
        alzimages = self.images.copy()
        print "-- reading image file from genderfile ..."
        t0 = time()        
        l, h, w = alzimages.shape
        X = np.zeros( ( l,1,h,w) )
        for n,f in enumerate(alzimages):
            #im = plt.imread(f)
            im = np.reshape(f, (1,h,w))
            #im = np.transpose(im,(2,0,1))
            X[n] = im            
            
        print "-- reading time :  %.4f sec.... " % (time() - t0)

        #n = range(X.shape[0])
        n = np.random.permutation(l)
        top5000 = n[:5000]
        
        #print "-- top 5000 :", top5000

        print "-- extract 5000 shuffle data from X ....."
        X = X[top5000]     
        print "-- after extracted X size ....", X.shape[0]
        
        print "-- label size ....", len(self.labels)        
        
        #label = np.array(genders)
        label = self.labels[top5000]
        X_, Xt_, label_, labelt_= train_test_split(X, label, test_size=test_size, random_state=42)
        print "-- shape of training data: ",X_.shape,label_.shape
        if test_size > 0.0:        
            print "-- shape of test data: ",Xt_.shape,labelt_.shape
        
        print "-- writing training lmdb data ....."
        db.dataWritelmdb(X_,label_)

        print "-- writing testing lmdb data ....."
        db.dataWritelmdb(Xt_,labelt_,True)

    def makeImagelistFile3ch(self,toprank=10000,testsize=0.3,channel=1):

        alzimages = self.images.copy()
        l, h, w = alzimages.shape
        n = np.random.permutation(l)
        
        print "-- just pick up one image for test --"        
        testimage = alzimages[  n[0] ]
        
        
        print "-- confirm image size:", testimage.shape
        print " (testimage) max min avg.: ", np.max(testimage),np.min(testimage),np.mean(testimage)
        # testimage /= 255.0     
        testimage *= 255.0/testimage.max()     

        print " (testimage) max min avg.: ", np.max(testimage),np.min(testimage),np.mean(testimage)

        h,w = testimage.shape
        testimage = testimage.reshape(h,w,1)
        
        X = np.zeros((h,w,3))
        X[:,:,0] = testimage[:,:,0]
        
        # change BGR 
        print " ** change BGR ** "
        X = X[:,:,::-1]
        
        
        #print X[:,:,0]
        
        print "-- confirm X image size after modifying 3 channel:", X.shape
        
        output_file = self.env.envlist['datadir'] + "/mytest3ch.jpg"
        cv2.imwrite(output_file, X)
        
        print "** file saved on ..", output_file

        output_file = self.env.envlist['datadir'] + "/mytest1ch.jpg"
        cv2.imwrite(output_file, testimage)
        
        print "** file saved on ..", output_file


    def makeImagelistFile(self,toprank=10000,testsize=0.3,channel=1):
        
        alzimages = self.images.copy()
        l, h, w = alzimages.shape
        
        print "***** color mode -- 3 channel is used to save image ******"
        print "\n** step1 ** -- extracting randomly training data --\n"
        n = np.random.permutation(l)
        
        
        top5000 = n[:toprank]

        livedatas = int(toprank * (1-testsize))
        #split train and test size 
        train_idx = top5000[:livedatas]
        test_idx = top5000[livedatas:]

        print ".. training  size .. %d " % len(train_idx)

        #X = alzimages[top5000,:,:]
        X_ = alzimages[train_idx,:,:]
        
        for n,myimage in enumerate(X_):
            #im = plt.imread(f)
            #im = np.transpose(im,(2,0,1))
        
            # normalized with 255
            myimage *= 255.0/myimage.max()     
            X_[n] = myimage            
        
        #label = self.labels[top5000]
        label_ = self.labels[train_idx]
        
        #X_, Xt_, label_, labelt_= train_test_split(X, label, test_size=testsize, random_state=42)
        #X_ = X.copy()
        #label_ = label.copy()

        cropdir = self.env.envlist['datadir'] + "/alz_train_images"

        training = []
        t0 = time()        
        print "-- writing images on ", cropdir
        
        filelists = os.listdir(cropdir)
        for efilename in filelists:
            item=os.path.join(cropdir,efilename)
            if os.path.isfile(item):
            # remove if file existed
                try:
                #print "remove file if exists ......... "
                    os.remove(item)            
                except OSError:
                    pass    
                
        print "-- %d training imagefiles are deleted from %s" % (len(filelists),cropdir)
        
        for idx,train in enumerate(X_):
            
            if idx % 1000 == 0:
                print "-- %d images saved ..." % idx
                
            target = label_[idx]
            file_idx = train_idx[idx]
            filename = "ALZ_" + str(file_idx) + ".jpg"
            training.append([  filename  , int(target)  ]  )
                        
            output_file = os.path.join(cropdir,filename)
            # remove if file existed
            #try:
                #print "remove file if exists ......... "
            #    os.remove(output_file)            
            #except OSError:
            #    pass       

            if channel == 3:
                train1 = train.reshape(h,w,1)
                train3 = np.zeros((h,w,3))
                train3[:,:,0] = train1[:,:,0]
                # change BGR 
                train = train3[:,:,::-1]

            # if channel 1,  black and white 
            # if channel 3,  color mode but saved only 1 chaanel      
            cv2.imwrite(output_file, train)
            
        print "-- all %d training images has been saved....." % (idx+1)
        print "-- writing time (for your ref.)  :  %.4f sec.... " % (time() - t0)

        rootdir = self.env.envlist['datadir']
        outfile = os.path.join(rootdir, "train_alzheimer_mri.dat")
        f = open(outfile,'w')
        writer = csv.writer(f,delimiter='\t')
        writer.writerows(training)
        print "-- training image txt file  has been saved on ....." , outfile
        
        
        print "\n** Step 2 **  ...... validation data starting .........\n"
        print "     max of validataion image data is 226970    \n"
        
        cropdir = self.env.envlist['datadir'] + "/alz_test_images"

        training = []
        t0 = time()        
        print "-- all testing (valid) images will be erased from ", cropdir
        
        filelists = os.listdir(cropdir)
        for efilename in filelists:
            item=os.path.join(cropdir,efilename)
            if os.path.isfile(item):
            # remove if file existed
                try:
                    #print "remove file if exists ......... "
                    os.remove(item)            
                except OSError:
                    pass            
        print "-- %d testing (valid) files have been deleted from %s" % (len(filelists),cropdir)
        

        alzvalid_images = self.validimages.copy()
        l, h, w = alzvalid_images.shape
        
        n = np.random.permutation(l)
         
        #top5000 = n[:toprank]

        livedatas = int(l * (1-testsize))
        #split train and test size 
        #train_idx = top5000[:livedatas]
        test_idx = n[:livedatas]
        
        print "-- extracted top %d randomly from valid image out of %d" % (len(test_idx), l)
        

        Xt_ = alzvalid_images[test_idx,:,:]
        print "--Xt length : ", Xt_.shape[0]
        
        for n,myimage in enumerate(Xt_):
            #im = plt.imread(f)
            #im = np.transpose(im,(2,0,1))
        
            # normalized with 255
            myimage *= 255.0/myimage.max()     
            Xt_[n] = myimage            
        
        #label = self.labels[top5000]
        label_ = self.valid_labels[test_idx]
        print "--label length : ", len(label_)
    
        for idx,train in enumerate(Xt_):
            
            if idx % 1000 == 0:
                print "-- %d images saved ..." % idx
                
            target = label_[idx]
            file_idx = test_idx[idx]
            filename = "ALZ_" + str(file_idx) + ".jpg"
            training.append([  filename  , int(target)  ]  )
                        
            output_file = os.path.join(cropdir,filename)
            # remove if file existed
            #try:
                #print "remove file if exists ......... "
            #    os.remove(output_file)            
            #except OSError:
            #    pass        
            if channel == 3:
                train1 = train.reshape(h,w,1)
                train3 = np.zeros((h,w,3))
                train3[:,:,0] = train1[:,:,0]
                # change BGR 
                train = train3[:,:,::-1]

            cv2.imwrite(output_file, train)
            
        print "-- all %d test (valid) images has been saved....." % (idx+1)
        print "-- writing time :  %.4f sec.... " % (time() - t0)

        rootdir = self.env.envlist['datadir']
        outfile = os.path.join(rootdir, "test_alzheimer_mri.dat")
        f = open(outfile,'w')
        writer = csv.writer(f,delimiter='\t')
        writer.writerows(training)
        print "-- test (valid) image txt file  has been saved on ....." , outfile

    
    def makeH5Data(self,toprank=10000):
        
        db = imageDataBase(self.env)
        
        alzimages = self.images.copy()
        
        t0 = time()        
        l, h, w = alzimages.shape
        X = np.zeros( ( l,1,h,w) )
        
        #X = np.zeros( ( len(files),3,250,250) )
        for n,myimage in enumerate(alzimages):
            #im = plt.imread(f)
            #im = np.transpose(im,(2,0,1))
            myimage *= 255.0/myimage.max()
            im = np.reshape(myimage, (1,h,w))            
            X[n] = im            
        
        print "-- total images into X ", n, X.shape
        print "-- reading time :  %.4f sec.... " % (time() - t0)

        n = np.random.permutation(l)
        top5000 = n[:toprank]
        
        print "-- extract 5000 shuffle data from X ....."
        X = X[top5000]     
        print "-- after extracted X size ....", X.shape[0]
        
        
        print "-- label size ....", len(self.labels)        
        #label = np.array(genders)
        label = self.labels[top5000]
        X_, Xt_, label_, labelt_= train_test_split(X, label, test_size=0.3, random_state=42)
        print "** shape of training data from top 5000: ",X_.shape,label_.shape
                
        print "--- writing train h5 db ......"
        #labels = {}
        #labels["label"] = label_
        # shuffle data for saving into image database
        db.dataWriteh5_v2(X_,label_)

        print "shape of testing data: ",Xt_.shape,labelt_.shape
        
        print "--- writing test h5 db ......"
        #labelst = {}
        #labelst["label"] = labelt_
        # shuffle data for saving into image database
        db.dataWriteh5_v2(Xt_,labelt_,True)        
            
    def readAlzClinicalData(self):
        
        files = self.alzImageData()
        images = self.StackImages(files)
        
        vfiles = self.alzValidImageData()
        validimages = self.StackImages(vfiles)
                
        return images,validimages
        
    def train(solver_prototxt_filename):
        '''
        Train the ANN
        '''
        caffe.set_mode_cpu()
        solver = caffe.get_solver(solver_prototxt_filename)
        solver.solve()

    def readH5Data(self,labelname=["label"]):
        
        db = imageDataBase(self.env)
        images,label = db.dataReadh5_v2()
        #y = label["label"]        
        
        return np.array(images),np.array(label)
        
        


