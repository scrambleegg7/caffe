# -*- coding: utf-8 -*-

"""

    alzRunFineTune 
    
    first of all, need to set image 227x227 to fit pretrained imagenet model
    

"""

import numpy as np
import caffe
import csv, os

import matplotlib.pyplot as plt

from caffe import layers as L
from caffeBase.envParam import envParamAlz

from AlzFineTuneClass import AlzFineTuneClass


import platform as pl


def recreateAlzNetTxt():
    
    rootdir = envParamAlz().envlist['datadir']

    outfile = os.path.join(rootdir, "train_alzheimer_mri.dat")

    print "-- open train_alzheimer_mri.dat --"
    f = open(outfile,'r')
        
    newdatalist = []
    newdir = os.path.join(rootdir,'alz_train_images')
    
    
    if pl.system() == "Linux":
        reader = csv.reader(f,lineterminator='\n',delimiter=' ')
    else:
        reader = csv.reader(f,delimiter='\t')

    for row in reader:
        filename = row[0]
        newdatalist.append([ os.path.join(newdir, filename), row[1] ] )
    
    f.close()
    
    print "-- open train.txt under finetune "
    newoutfile = os.path.join(rootdir, "finetune/train.txt")    
    f = open(newoutfile,'wb')


    if pl.system() == "Linux":
        writer = csv.writer(f,lineterminator='\n',delimiter=' ')
    else:
        writer = csv.writer(f,delimiter='\t')
        
    writer.writerows(newdatalist)
    
    
    outfile = os.path.join(rootdir, "test_alzheimer_mri.dat")

    print "-- open test_alzheimer_mri.dat --"
    f = open(outfile,'r')
        
    newdatalist = []
    newdir = os.path.join(rootdir,'alz_test_images')
    
    
    if pl.system() == "Linux":
        reader = csv.reader(f,lineterminator='\n',delimiter=' ')
    else:
        reader = csv.reader(f,delimiter='\t')



    for row in reader:
        filename = row[0]
        newdatalist.append([ os.path.join(newdir, filename), row[1] ] )
    
    f.close()
    
    print "-- open test.txt under finetune "
    newoutfile = os.path.join(rootdir, "finetune/test.txt")    
    f = open(newoutfile,'wb')


    if pl.system() == "Linux":
        writer = csv.writer(f,lineterminator='\n',delimiter=' ')
    else:
        writer = csv.writer(f,delimiter='\t')
        
    writer.writerows(newdatalist)



def untrained():
    
    alzFineCls = AlzFineTuneClass()
    
    im, label = alzFineCls.untrained()
    
    print im.shape,label.shape
    
    mindex = 2
    
    im = im[mindex]
    image = alzFineCls.deprocess_net_image(  im   )
    
    im = np.transpose(im,(1,2,0))
    print im
    plt.imshow(im)
    plt.show()

    labels = alzFineCls.setLabels()
    print labels[  label[mindex] - 1 ]
    
def prediction(alzFineCls):
    
    
    im,label = alzFineCls.untrained()
    mindex = 2    
    im = im[mindex]
        
    imagenet_net = alzFineCls.dummyImageSetNet()
    alzFineCls.disp_imagenet_preds(imagenet_net,im)
        
    """
    thisi is an untrained model, therefore 1/3 probabilties are displayed...
    
    """
    untrained_net = alzFineCls.untrained_net()
    alzFineCls.disp_alz_preds(untrained_net,im)
    
    diff = untrained_net.blobs['fc7'].data[0] - imagenet_net.blobs['fc7'].data[0]
    error = (diff ** 2).sum()
    assert error < 1e-8   
    
    print "-- untrained_net fc7 vs imagenet_net fc7 layer difference:", error 
    

def main():
    
    recreateAlzNetTxt()
    #proc1()
    #untrained()
    alzFineCls = AlzFineTuneClass()
    prediction(alzFineCls)



if __name__ == "__main__":
    main()