import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from django.http import request
from os import rename
import requests
import shutil

import re
import csv

from sklearn.cross_validation import train_test_split

from caffeBase.envParam import envParamLFW
from caffeBase.imageProcessClass import imageProcessClass

from imageDataBase.imageDataBase import imageDataBase

from time import time

class lfw(object):

    def __init__(self,env):
        self.lfw_ = False
        self.direct = None
        self.env = env
        self.N = None
        self.X = np.zeros((1,3,250,250))
        self.y = np.zeros(10)
        
        self.genderfile = os.path.join(self.env.envlist['rootdir'],"genderfile.csv")
        self.fNfile = os.path.join(self.env.envlist['rootdir'],"firstName.csv")

        self.genderFolds = os.path.join(self.env.envlist['rootdir'],"LFW-gender-folds.txt")        
        
        self.firstName = []
        
        self.genderDict = {}
        
    def readLFWgenderfile(self):
        f = open(self.genderFolds, 'r')
        
        reader = csv.reader(f,delimiter='\t')
        for r in reader:
            name = r[0].split("/")
            gender =  r[-1:][0]
            #print name[0],gender
            person_name = name[0]
            
            if not person_name in self.genderDict:
                self.genderDict[person_name] = gender
                
        #print self.genderDict

    def loadFaceDataTopN(self,topN=10,test=False):
        
        facelist = self.loadFaceData(test)
        
        return facelist[:topN]

    def loadFaceData(self,test=False):
        
        mydir = self.env.envlist['datadir']
        
        pattern = r"jpg"
        filenamelist = []
                
        self.images = []
        for r,dirs,files in os.walk(mydir):
            for f in files:
                ob = re.search(pattern,f)
                obg = ob.group() if ob != None else False
                if obg:
                    
                    person_name = r.split("/")[-1:]
                    person_name = person_name[0]
                    #print person_name                    
                    fullname = os.path.join(r,f)
                    #print fullname
                    filenamelist.append(fullname)
                    
        return filenamelist

    def setImages(self,test=False):

        mydir = self.env.envlist['datadir']
        
        # set gender file 
        self.readLFWgenderfile()
        
        i = 0
        pattern = r"jpg"
        genderlist = []
                
        self.images = []
        for r,dirs,files in os.walk(mydir):
            for f in files:
                ob = re.search(pattern,f)
                obg = ob.group() if ob != None else False
                if obg:
                    
                    person_name = r.split("/")[-1:]
                    person_name = person_name[0]
                    #print person_name                    
                    fullname = os.path.join(r,f) 
                    self.images.append(fullname)
                    
                    sex = self.checkSex2(person_name)
                    #findex = self.checkFirstName(f)
                    
                    genderlist.append([fullname,sex])
            
            if i > 5 and test:
                break
            i += 1
            
            if i % 100 == 0:
                print "--- %d images has been processed --" % i
        
        with open(self.genderfile, 'w') as f:
            writer = csv.writer(f)            
            writer.writerows(genderlist) 
            
        flist = [ (idx,fn) for idx,fn in enumerate(self.firstName)]
        with open(self.fNfile, 'w') as f:
            writer = csv.writer(f)            
            writer.writerows(flist) 
        
        
        #self.N = len(self.images)
        #self.X = np.zeros((self.N,3,250,250))
        #self.y = np.zeros(self.N)
        
        #self.imageload()
                    #sex = self.checkSex(f)

    def imageload(self,imagefile=None):
        if imagefile:
            imagedata = os.path.join(self.mydir,imagefile)
            print "image data:",imagedata
            image = plt.imread(imagedata)
            return image
        else:

            for n,imagedata in enumerate(self.images):
                
                fname = imagedata.split("/")[-1:][0]
                fsplit = fname.split("_")
                sex = self.checkSex(fsplit[0])
                im = plt.imread(imagedata)
                im = np.transpose(im,(2,0,1))
                self.X[n] = im
                self.y[n] = sex                
                #print "shape of lfw data:", im.shape
                
    def checkFirstName(self,fname):
        fileSplit = fname.split("_")
        firstName = fileSplit[0]

        if not firstName in self.firstName:
            self.firstName.append(firstName)
            
        return self.firstName.index(firstName)  

    def checkSex2(self,name):
        
        g = 0
        if name in self.genderDict:
            g = 1 if self.genderDict[name] == "M" else 2
        
        return g
  
    def checkSex(self,fname):
        fileSplit = fname.split("_")

        result = requests.get("http://api.genderize.io?name=%s" % fileSplit[0])
        result = result.json()

        try:
            if float(result['probability']) > 0.9:
                if result['gender'] == 'male':
                    print "--- ", result['name'],"male"
                    return 1
                    #shutil.copyfile(f,"%s/%s" % (maleFolder,file))

                elif result['gender'] == 'female':
                    print "--- ", result['name'],"female"
                    return 2
                    #shutil.copyfile(f,"%s/%s" % (femaleFolder,file))
        except Exception as e:
            #if result['name']:
            print "no found sex ...", 
            return 0

    def makeGenderLMDB(self,test_size=0.4):

        db = imageDataBase(self.env)
        
        files,genders = self.readGenderFile()

        print "-- reading image file from genderfile ..."
        t0 = time()        
        X = np.zeros( ( len(files),3,250,250) )
        for n,f in enumerate(files):
            im = plt.imread(f)
            im = np.transpose(im,(2,0,1))
            X[n] = im            
            
        print "-- reading time :  %.4f sec.... " % (time() - t0)

        n = range(X.shape[0])
        np.random.shuffle(n)
        top5000 = n[:5000]
        
        #print "-- top 5000 :", top5000

        print "-- extract 5000 shuffle data from X ....."
        X = X[top5000]     
        print "-- after extracted X size ....", X.shape[0]
        
        print "-- label size ....", len(genders)        
        
        label = np.array(genders)
        label = label[top5000]
        X_, Xt_, label_, labelt_= train_test_split(X, label, test_size=test_size, random_state=42)
        print "-- shape of training data: ",X_.shape,label_.shape
        if test_size > 0.0:        
            print "-- shape of test data: ",Xt_.shape,labelt_.shape
        
        db.dataWritelmdb(X_,label_)
        db.dataWritelmdb(Xt_,labelt_,True)


    def makeH5Data(self):
        
        db = imageDataBase(self.env)
        
        files,genders = self.readGenderFile()
        
        X = np.zeros( ( len(files),3,250,250) )
        for n,f in enumerate(files):
            im = plt.imread(f)
            im = np.transpose(im,(2,0,1))
            X[n] = im            
        
        
        label = np.array(genders)
        X_, Xt_, label_, labelt_= train_test_split(X, label, test_size=0.3, random_state=42)
        print "shape of training data: ",X_.shape,label_.shape
                
        print "--- writing train db ......"
        labels = {}
        labels["label"] = label_
        # shuffle data for saving into image database
        db.dataWriteh5(X_,labels)

        print "shape of testing data: ",Xt_.shape,labelt_.shape
        
        print "--- writing train db ......"
        labelst = {}
        labelst["label"] = labelt_
        # shuffle data for saving into image database
        db.dataWriteh5(Xt_,labelst,True)        
        
    
    def readGenderFile(self):
        
        files = []
        genders = []
        with open(self.genderfile, 'r') as f:
            reader = csv.reader(f)
            
            #row_count = sum(1 for row in reader)
            for r in reader:
                #print r
                files.append(r[0])
                genders.append(r[1])
                
        
        return files,genders
        
        
    def readH5Data(self):
        
        db = imageDataBase(self.env)

        labels = ['label']
        images,hlabels = db.dataReadh5(labels)  
        
        return images, hlabels
        

def main():
    #print "-- stat lfw class"
    
    lfw_ = lfw(envParamLFW())
    #lfw_.readLFWgenderfile()
    
    # set lfw     
    #lfw_.setImages()    
    
    lfw_.makeGenderLMDB()
    # create h5 database    
    #lfw_.makeH5Data()
    #    
    
    #images , hlabels = lfw_.readH5Data()

    #label = hlabels['label']
    #print "--- show image from h5 database......."
    #for i, im in enumerate(images):
    #    imageProc = imageProcess(im)
    #    
    #    im0 = imageProc.convertToCHW()
    #    print "first name:%d" % label[i]
    #    plt.figure()
    #    plt.imshow(im0)        
        
    #    if i > 5: 
    #        break

if __name__ == "__main__":
    main()
