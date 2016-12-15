# -*- coding: utf-8 -*-


"""

   initialize environment parameters 
   a special directory setting is loaded based on class...
   
   
"""


import numpy as np
import os

import platform as pl
        

class envParam(object):
    
    def __init__(self):

        self.flag = False
        self.envlist = {}

        if pl.system() == "Linux":
            self.home = "/home/hideaki/SynologyNFS/myProgram/pythonProj"
            self.myDataDir = self.home
            self.envlist['caffe-public_root'] = '/home/hideaki/cafferoot/caffe'

        else:    
            self.home = "/Users/donchan/Documents"
            self.myDataDir = os.path.join(self.home,"mydata")
            self.envlist['caffe-public_root'] = '/Users/donchan/caffe/caffe-public'
        
        self.setEnv()
        
        self.setCaffemodel()
        self.setSolver()

        self.setTrain()
        self.setTest()      
        self.setDeploy()      
        
        self.setUserCaffeTrain()
        
        
    def setEnv(self):
        envlistStr = ['caffe_root','caffe_model','train','test','solver']
        for k in envlistStr:
            self.envlist[k] = "envparam"
    
    def getCaffeRoot(self):
        return self.envlist['caffe_root']
        
    def setUserCaffeTrain(self):
        self.usercaffetrain = os.path.join(self.envlist['caffe_root'],self.envlist['usercaffetrain'])     
        
    def setCaffemodel(self):
        self.caffemodel = os.path.join(self.envlist['caffe_root'],self.envlist['caffe_model'])
    
    def setTrainedCaffemodel(self,w):
        self.caffemodel = os.path.join(self.envlist['caffe_root'],w)
        
    def setSolver(self):
        self.solver = os.path.join(self.envlist['caffe_root'],self.envlist['solver'])
    
    def setTrain(self):
        self.train = os.path.join(self.envlist['caffe_root'],self.envlist['train'])

    def setTest(self):
        self.test = os.path.join(self.envlist['caffe_root'],self.envlist['test'])

    def setDeploy(self):
        self.deploy = os.path.join(self.envlist['caffe_root'],self.envlist['deploy'])

    def getCaffemodel(self):
        return self.caffemodel
        
    def getSolver(self):
        return self.solver
        
    def getTrain(self):
        return self.train

    def getTest(self):
        return self.test

    def getDeploy(self):
        return self.deploy
        
    def getUserCaffeTrain(self):
        return self.usercaffetrain
        
        
        
class envParamCifar100(envParam):
    
    def __init__(self):
        super(envParamCifar100,self).__init__()
        
    def setEnv(self):
        
        print "----- set environment for python caffe custom cifar 100 ------- "

        self.envlist['caffe_root']='/Users/donchan/Documents/Statistical_Mechanics/caffe/python-caffe-custom-cifar-100-conv-net' # this file should be run from {caffe_root}/examples (otherwise change this line)
        self.envlist['caffe_model'] = 'cnn_snapshot_iter_150000.caffemodel'
        self.envlist['solver'] = 'cnn_solver_rms.prototxt'
        self.envlist['train'] = 'cnn_train.prototxt'
        self.envlist['test'] = 'cnn_test.prototxt'
        #envlist['weights'] = 'iter/cnn100_full_iter_43759.caffemodel.h5'
        self.envlist['usercaffetrain'] = 'user_train_test.prototxt'


class envParamLFW(envParam):

    def __init__(self):
        super(envParamLFW,self).__init__()

    def setEnv(self):
        
        #self.envlist['caffe_root']='/Users/donchan/Documents/Statistical_Mechanics/caffe/lfw'
        self.envlist['caffe_model'] = 'caffenet_train_iter_2000.caffemodel'
        self.envlist['solver'] = 'lfw_quick_solver.prototxt'
        
        self.envlist['caffe_root']='/Users/donchan/caffe/caffe/examples/lfw'
        
        self.envlist['rootdir']='/Users/donchan/caffe/caffe/examples/lfw'
        self.envlist['datadir']='/Users/donchan/caffe/caffe/examples/lfw/lfw_funneled' 
        self.envlist['train_txt'] = 'train.txt'
        self.envlist['test_txt'] = 'test.txt'
        self.envlist['train'] = 'lfw_train_test.prototxt'
        self.envlist['test'] = 'lfw_train_test.prototxt'
        self.envlist['deploy'] = 'lfw_deploy.prototxt'
                
        self.envlist['meanfile'] = 'mean.binaryproto'
        
        self.envlist['usercaffetrain'] = 'user_train_test.prototxt'
        
        self.envlist['traindb'] = 'train.h5'
        self.envlist['testdb'] = 'test.h5'     
        
        self.envlist['train_lmdb'] = "train_lmdb"
        self.envlist['test_lmdb'] = "test_lmdb"
        
class envParamAlz(envParam):

    def __init__(self):
        super(envParamAlz,self).__init__()

    def setEnv(self):
        
        
        
        #self.envlist['caffe_root']='/Users/donchan/Documents/Statistical_Mechanics/caffe/alz'
        self.envlist['caffe_model'] = 'caffenet_train_iter_2000.caffemodel'
        self.envlist['solver'] = 'alz_quick_solver.prototxt'
        
        
        self.envlist['caffe_root']= os.path.join(self.myDataDir,'KaggleData/Alzheimer')

        
        self.envlist['rootdir'] = os.path.join(self.myDataDir,'KaggleData/Alzheimer')
        self.envlist['datadir'] = os.path.join(self.myDataDir,'KaggleData/Alzheimer') 
        self.envlist['train_txt'] = 'train.txt'
        self.envlist['test_txt'] = 'test.txt'
        self.envlist['train'] = 'alz_train_test.prototxt'
        self.envlist['test'] = 'alz_train_test.prototxt'
        self.envlist['deploy'] = 'alz_deploy.prototxt'
                
        self.envlist['meanfile'] = 'mean.binaryproto'
        
        self.envlist['usercaffetrain'] = 'user_train_test.prototxt'
        
        self.envlist['traindb'] = 'train.h5'
        self.envlist['testdb'] = 'test.h5'     
        
        self.envlist['train_lmdb'] = "train_lmdb"
        self.envlist['test_lmdb'] = "test_lmdb"


        
def main():
    env = envParamCifar100()

    print env.getCaffemodel()
    print env.getSolver()
    print env.getTrain()
    print env.getTest()


if __name__ == "__main__":
    main()



    