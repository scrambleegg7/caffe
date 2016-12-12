# -*- coding: utf-8 -*-


import os
import caffe
import numpy as np
import matplotlib.pyplot as plt

from caffe import layers as L
from caffe import params as P

from caffe.proto import caffe_pb2
import cPickle
#from caffe.proto import caffe_pb2

from caffeBase.caffeUserClass import caffeUserClass
from caffeBase.envParam import envParamCifar100


class CifarClass(caffeUserClass):
    
    def __init__(self,env):
        
        super(CifarClass,self).__init__(env)
        
        self.res_blobs = ('loss_c', 'accuracy_c', "loss_f", "accuracy_f")
        
    def runSolver(self,niter=500):

        solver = self.setSolver()                
        disp_interval = 100
    
        """Run solvers for niter iterations,
            returning the loss and accuracy recorded each iteration.
                    `solvers` is a list of (name, solver) tuples."""
        res = {bstr: np.zeros(niter) for bstr in self.res_blobs}
        for it in range(niter):

           solver.step(1)  # run a single SGD step in Caffe
           
           res['loss_c'][it],res['accuracy_c'][it],res['loss_f'][it],res['accuracy_f'][it] = (solver.net.blobs[b].data.copy() for b in self.res_blobs)
           if it % disp_interval == 0 or it + 1 == niter:
               print "* Loop Count ------> %d " % it
               print "* loss c  %.4f" % res['loss_c'][it]
               print "* loss f  %.4f" % res['loss_f'][it]
               print "* accuracy c %.4f" % (100. * res['accuracy_c'][it])
               print "* accuracy f %.4f" % (100. * res['accuracy_f'][it])

        filename = 'weights.%s.caffemodel' % "rms_pretrained"
        self.env.setTrainedCaffemodel(filename)        
        weights = self.env.getCaffemodel()
        
        #self.env.envlist['trained_weights'] = os.path.join(self.env.getCaffeRoot(), filename)
        print "--- pretrained weight generated:", weights
        solver.net.save(weights)
            
        return res
        

    def eval_net(self, net, test_iters=10):
        
        accuracy = 0
        for it in xrange(test_iters):
            accuracy += net.forward()['accuracy_f']
            #print "acuracy f ",  accuracy
        
        accuracy /= test_iters
        return float(accuracy)

def unpickle(nfile):
    print "unpacking file", nfile
    fo = open(nfile, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def labelNames():
    ## meta
    nfile = '/Users/donchan/caffe/caffe/data/cifar-100-python/'
    nfilemeta = nfile + "meta"
    bm = unpickle(nfilemeta)
    clabel_names = bm['coarse_label_names']
    flabel_names = bm['fine_label_names']
    print "meta clabel ", type(clabel_names),len(clabel_names)
    print "meta flabel ", type(flabel_names),len(flabel_names)

    return flabel_names,clabel_names

def imageProcess(image):
    
    image = image.copy()
    image = image[::-1]               # BGR -> RGB
    image = np.transpose(image,(1,2,0))
    
    #print "by 2 x 2", meanarray[:2,:2,:]
    
    #meanarray = meanarray.astype(np.uint8)
    #image = image + meanarray

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image


    

def showTestImageAndProb(net):
    
    
    flabels,clables = labelNames()
    
    net.forward()
    testN = 0
    data_batch = net.blobs['data'].data.copy()
    print data_batch.shape
    labelfine_batch = np.array(net.blobs['label_fine'].data, dtype=np.int32)
    
    im = data_batch[testN]
    plt.imshow(imageProcess(im))
    print "fine label:", flabels[ labelfine_batch[testN] ], labelfine_batch[testN]  
    
    input_blob = net.blobs['data']
    
    net.blobs['data'].data[0, ...] = im
    probs = net.forward(start='conv1')['probs'][0]
    
    print "probs shape", probs.shape

    k = 5
    top_k = (-probs).argsort()[:5]
    
    print "shape of top_k", top_k.shape
    print "top_k from top 5", top_k
    print 'top %d predicted %s labels =' % (k, "cnn test")
    print '\n'.join('\t(%d) %5.2f%% %s  %d' % (i+1, 100*probs[p], flabels[p], p   )
                    for i, p in enumerate(top_k))
    
    return True
    
    
    
    
       
    

def main():
    #process()
    env = envParamCifar100()
    cifar = CifarClass(env)
    #res = cifar.runSolver(100)    
    #print res["loss_c"]
    
    net = cifar.setNetWithTrainedWeights()
    print "accuracy with trained weights %.4f " % (cifar.eval_net(net,50) * 100.)


    
    
if __name__ == "__main__":
    main()