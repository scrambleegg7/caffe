# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from CifarClass import CifarClass

from caffeBase.envParam import envParamCifar100

from caffeBase.imageProcess import imageProcess

def process():
    
    env = envParamCifar100()
    CifarClass_ = CifarClass(env)
    
    net = CifarClass_.setNetWithTrainedWeights()

    data_batch, labels_batch = CifarClass_.getDataAndLabel(net,"label_fine")
    
    
    print "data_batch shape", data_batch.shape
    #data_len = data_batch.shape[0]
    #n = np.random.randint(0,data_len,1)
    #print "target image no. %d" % n
    image = data_batch[20]
    
    print image.shape
    
    


def main():
    process()
    


if __name__ == "__main__":
    main()