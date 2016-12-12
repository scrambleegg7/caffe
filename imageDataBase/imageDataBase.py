# -*- coding: utf-8 -*-

from caffeBase.envParam import envParamLFW

import numpy as np
import h5py
import os
import shutil

import lmdb

import caffe

class imageDataBase(object):
    
    def __init__(self,env):
        
        self.flag = False        
        self.env = env
        
        self.train_filename = os.path.join(self.env.envlist['rootdir'],self.env.envlist['traindb'])
        self.train_txt = os.path.join(self.env.envlist['rootdir'],self.env.envlist['train_txt'])

        self.test_filename = os.path.join(self.env.envlist['rootdir'],self.env.envlist['testdb'])
        self.test_txt = os.path.join(self.env.envlist['rootdir'],self.env.envlist['test_txt'])

        
    def dataWritelmdb(self,X,labels,test=False):
        
        if test:
            lmdb_dir = os.path.join(self.env.envlist['rootdir'],self.env.envlist['test_lmdb'])
        else:
            lmdb_dir = os.path.join(self.env.envlist['rootdir'],self.env.envlist['train_lmdb'])


        # remove if file existed
        try:
            print "remove file if exists ......... "
            #os.remove(lmdb_dir)
            shutil.rmtree(lmdb_dir)
        except OSError:
            pass



        print "--  writing lmdb ...... ", lmdb_dir
        
        map_size = X.nbytes * 10
        lmdb_env = lmdb.open(lmdb_dir,map_size=map_size)

        N = X.shape[0]
        with lmdb_env.begin(write=True) as txn:
        # txn is a Transaction object
            for i in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
                datum.label = int(labels[i])
                str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())

    def dataWriteh5_v2(self,X,labels,test=False):

        if not test:
            train_filename = self.train_filename
            train_txt = self.train_txt
        else:
            train_filename = self.test_filename
            train_txt = self.test_txt
            
        print "-- h5 filename ", train_filename
        print "-- TXT file to point h5 data ", train_txt
        
        # remove if file existed
        try:
            print "remove file if exists ......... "
            os.remove(train_filename)
            
        except OSError:
            pass
        
        print "-- save X and labels into image database (h5) ...."
                
        with h5py.File(train_filename, 'w') as f:
            f['data'] = X.astype(np.float32)
            f['label'] = labels.astype(np.float32)
        with open(train_txt, 'w') as f:
            f.write(train_filename + '\n')

    
    def dataWriteh5(self,X,labels,test=False):
        
        comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
        
        if not test:
            train_filename = self.train_filename
            train_txt = self.train_txt
        else:
            train_filename = self.test_filename
            train_txt = self.test_txt
            
        print "-- h5 filename ", train_filename
        print "-- TXT file to point h5 data ", train_txt
        
        # remove if file existed
        try:
            print "remove file if exists ......... "
            os.remove(train_filename)
            
        except OSError:
            pass
            
        with h5py.File(train_filename, 'w') as f:
            print "-- save X into image database (h5) ...."
            f.create_dataset('data', data=X, **comp_kwargs)
            
            for k, v in labels.items():
                
                print "-- labels value : ",k, v.astype(np.int_)
                f.create_dataset(k, data=v.astype(np.int_), **comp_kwargs)
            #f.create_dataset('label_fine', data=y_f.astype(np.int_), **comp_kwargs)
        with open(train_txt, 'w') as f:
            f.write(train_filename + '\n')
        
    def dataReadh5_v2(self,labels="label"):

        train_filename = self.train_filename
        self.images = None
        self.label = {}        
        
        print "-- reading h5 image database ....... "
        hf =  h5py.File(train_filename, 'r')
        self.images = hf.get('data')
        label = hf.get(labels)

        return self.images,label

        
    def dataReadh5(self,labels):
        
        
        train_filename = self.train_filename
        self.images = None
        self.label = {}        
        
        print "-- reading h5 image database ....... "
        hf =  h5py.File(train_filename, 'r')
        self.images = hf.get('data')

        label = {}
        for l in labels:
            print "-- datareadh5 - imageDataBase : label name : ", l
            label[l] = hf.get(l)    
            
        return self.images,label
    
    def readOriginalData(self):
        pass
    
    
    

def main():
    
    env = envParamLFW()
    db = imageDataBase(env)
    
    h5_filedir="/Users/donchan/caffe/caffe/examples/lfw/train.h5"
    #db.dataReadh5(["label_sex"])
    
    #with h5py.File(h5_filedir, 'r') as hf:
    #    images = hf.get('data')
    #    print images[0].shape
        
if __name__ == "__main__":
    main()
    