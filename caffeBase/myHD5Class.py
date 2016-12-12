# -*- coding: utf-8 -*-

import numpy as np
import h5py


class myHD5Class(object):
    
    def __init__(self,filename,test=False):
        self.test = test
        self.filename = filename
        
    
    def readH5DB(self):
        
        with h5py.File(self.filename, 'r') as hf:
            
            datas = hf.get('data')
            labels = hf.get('label')

            print('List of arrays in this file: \n', hf.keys())
        
            print "-- convert np data array from h5py data format"        
            np_data = np.array(datas)
            np_label = np.array(labels)
            print "-- converted array data shape", np_data.shape
            print "-- converted array label shape", np_label.shape
            
        return np_data,np_label

    def writeH5DB(self,data,label):
        
        with h5py.File(self.filename, 'w') as f:
            f['data'] = data.astype(np.float32)
            f['label'] = label.astype(np.float32)

        return True

        