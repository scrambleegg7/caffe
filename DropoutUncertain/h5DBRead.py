# -*- coding: utf-8 -*-

from caffeBase.myHD5Class import myHD5Class

import numpy as np
import os

def process():
    
    
    fdir = "/Users/donchan/caffe/caffe/examples/DropoutUncertaintyCaffeModels/co2_regression/data"
    filename = "test.h5"
    
    h5filename = os.path.join(fdir,filename)
    
    
    h5DBCls = myHD5Class(h5filename)
    
    data,label = h5DBCls.readH5DB()
    
    print "-- data top10:", data[:10]
    print "-- label top10:",label[:10]    
    
def main():
    
    process()
    
if __name__ == "__main__":
    main()