# -*- coding: utf-8 -*-

import os
import caffe
import numpy as np
import matplotlib.pyplot as plt

from caffe import layers as L
from caffe import params as P

from caffe.proto import caffe_pb2
import cv2
#from caffe.proto import caffe_pb2


def setEnvCifar100():

    envlist = {}
    envlist['caffe_root']='/Users/donchan/Documents/Statistical_Mechanics/caffe/python-caffe-custom-cifar-100-conv-net' # this file should be run from {caffe_root}/examples (otherwise change this line)
    #envlist['caffe_root'] = '/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    envlist['caffe_model'] = 'cnn_snapshot_iter_150000.caffemodel'
    envlist['deploy'] = 'cnn_deploy.prototxt'
    envlist['meannpy'] = 'mean.npy'
    envlist['meanfile'] =  '/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/cnn_mean.binaryproto'
    envlist['solver'] = 'cnn_solver_h5.prototxt'
    envlist['train'] = 'cnn_train.prototxt'
    envlist['test'] = 'cnn_test.prototxt'
    envlist['weights'] = 'iter/cifar100_nin_iter_6000.caffemodel'
    
    return envlist


def setEnv():

    envlist = {}
    envlist['caffe_root'] = '/Users/donchan/caffe/caffe/examples/cifar10/quick_'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    envlist['caffe_model'] = 'cifar10_quick_iter_5000.caffemodel.h5'
    envlist['deploy'] = 'deploy.prototxt'
    envlist['meannpy'] = 'mean.npy'
    envlist['solver'] = 'cifar10_quick_solver.prototxt'
    envlist['train'] = 'cifar10_quick_train_test.prototxt'
    envlist['untrain'] = 'cifar10_quick_untrained.prototxt'
    
    envlist['weights'] = 'cifar10_quick_iter_5000.caffemodel.h5'
    envlist['meanfile'] = '/Users/donchan/caffe/caffe/examples/cifar10/mean.binaryproto'
    #envlist['imagefile'] = lmdb_img # after loading image with caffe io load_image

    return envlist
    
def meanbinary(envlist):
    
    # read mean binary image 
    
    mean_blob = caffe_pb2.BlobProto()
    with open(envlist['meanfile']) as f:
        mean_blob.ParseFromString(f.read())
        
    print "----mean blob data shape:" , np.asarray(mean_blob.data,dtype=np.float32).shape 
    print "----mean blob h w c num", mean_blob.height,mean_blob.width,mean_blob.channels,mean_blob.num
    #print mean_blob.data
    # convert array into h x w x channel
    mean_array_new = caffe.io.blobproto_to_array( mean_blob )
    print "mean_array_new shape with blobproto_to_array", mean_array_new.shape

    mean_array = np.transpose( mean_array_new[0],(1,2,0) ) 
    #mean_array = np.asarray(
    #    mean_blob.data,
    #    dtype=np.float32).reshape(
    #    (mean_blob.height,
    #     mean_blob.width,mean_blob.channels ))

    return mean_array

def imageProcess(image,meanarray):
    
    image = image.copy()
    image = image[::-1]               # BGR -> RGB
    image = np.transpose(image,(1,2,0))
    
    #print "by 2 x 2", meanarray[:2,:2,:]
    
    #meanarray = meanarray.astype(np.uint8)
    image = image + meanarray

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image


def process(envlist,test=False):
    
    meanarray = meanbinary(envlist)    
    
    # load untrained proto txt file (lr_mult = 0)
    train = os.path.join(envlist['caffe_root'],envlist['test'])
    weights = os.path.join(envlist['caffe_root'],envlist['caffe_model'])
    print "--- train file ---",train
    print "--- weight file ---", weights    
    
    
    trained_style_net = caffe.Net(train,weights,caffe.TEST)
    
    trained_style_net.forward()
    
    
    style_data_batch = trained_style_net.blobs['data'].data.copy()
    labels = trained_style_net.blobs['label_fine'].data.copy()
    labels = np.array(labels,dtype=np.int32)

    print "image shape style_data_batch", style_data_batch.shape


    batch_index = 20
    image = style_data_batch[batch_index]
    
    print "image shape before imageProcess", image.shape
    
    image = imageProcess(image,meanarray)

    print "blob data image shape", image.shape
    print "label category",labels[batch_index]
    print "length of labels:",len(labels)
    
    if test:
        #print image
        plt.title("no. %d image read from lmdb ........" % batch_index)
        plt.imshow(image)
        plt.show()
        
def visualDsiplayFilter(data):
    
    max_ = data.max()
    min_ = data.min()
    
    data = (data - min_) / (max_ - min_)
    
    dim_ = data.shape[0]
    n = int(np.ceil( np.sqrt( dim_ )    )   )
    
    padding =   ( ( ( 0, n ** 2 - dim_), (0,1),(0,1) ) + ((0,0), ) * (data.ndim - 3 ) )
    
    data = np.pad(data,padding,mode='constant',constant_values=1)
    #tile filter into image 
    data = data.reshape((n,n) + data.shape[1:]).transpose( (0,2,1,3) + tuple( range(4, data.ndim + 1))) 
    data = data.reshape(( n * data.shape[1], n * data.shape[3]) + data.shape[4:] )
    
    plt.imshow(data); plt.axis('off')
    plt.show()
    

def run_SingleStepSolver(envlist):

    envlist['solver'] = 'cifar100_full_solver.prototxt'    
    style_solver_filename = os.path.join(envlist['caffe_root'],envlist['solver'])
    
    weights = os.path.join(envlist['caffe_root'],envlist['weights'])
    
    print "[ style solver name]", style_solver_filename
    solver = caffe.get_solver(style_solver_filename)
    net = solver.net

    print "---- check blobs & params"
    print "---- conv structure information"    
    for k,v in net.blobs.items():
        print k,v.data.shape
    
    for k,v in net.params.items():
        print k, "length of conv parameters:",len(v)
        print "v data shape", v[0].data.shape,v[1].data.shape
    
    #print net.blobs['conv1'].data.shape
    
    net.forward()
    solver.test_nets[0].forward()
    
    #print "param0 of conv1", net.params['conv1'][0].data.shape
    #print "param1 of conv1", net.params['conv1'][1].data.shape
    #print "param0 of conv1", net.params['conv1'][0].data
    #print "param1 of conv1", net.params['conv1'][1].data
        
    # now image display
    meanarray = meanbinary(envlist)        
    batch_size = solver.test_nets[0].blobs['data'].num
    print "batch size:", batch_size    
    
    batch_data = net.blobs['data'].data.copy()
    test_batch_data = solver.test_nets[0].blobs['data'].data.copy()

    print "batch data shape",batch_data.shape
    print "pick up no.20"
    image = batch_data[20]
    t_image = test_batch_data[20]    
    # catch up mean image    
    image = imageProcess(image,meanarray)
    t_image = imageProcess(t_image,meanarray)

    print "blob data image shape", image.shape
    #print "label category",labels[batch_index]
    #fig = plt.figure()
    #ax1 = fig.add_subplot(121)
    #ax1.title('image from data param')
    #ax1.imshow(image)
    #ax2 = fig.add_subplot(122)
    #ax2.title('test image from data param')
    #ax2.imshow(t_image)
    
    #plt.show()
    
    
    print "************ stepping solver *****************"
    solver.step(1)
    #
    filters = net.params['conv1'][0].data
    visualDsiplayFilter(filters.transpose(0,2,3,1))
    
    # no1 of pool3 filters
    filters = net.blobs['pool3'].data[0]
    visualDsiplayFilter(filters)    
    
    # pickup first image feature
    features = net.blobs['ip_f'].data[0]
    plt.subplot(211)    
    plt.plot(features.flat)
    plt.subplot(212)
    _ = plt.hist(features.flat[features.flat > 0],bins=100 )
    plt.show()

def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss_f', 'accuracy_f')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            
            #print "loss",name,s.net.blobs["loss"].data.copy()             
            #print "accuracy",name,s.net.blobs["accuracy"].data.copy()             
            
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
            
    weight_dir = '/Users/donchan/caffe/caffe/examples/cifar10/quick_'
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    
    return loss, acc, weights


def startSolver(envlist):
    

    style_solver_filename = os.path.join(envlist['caffe_root'],envlist['solver'])

    #caffe.set_device(0)
    #scaffe.set_mode_cpu()
    solver = caffe.SGDSolver(style_solver_filename)

    solver.net.forward()
    solver.test_nets[0].forward()
    solver.step(1)

    niter = 2000
    test_interval = 10
    train_loss = np.zeros(niter)
    test_acc = np.zeros(int(np.ceil(niter * 1.0 / test_interval)))
    print len(test_acc)
    output = np.zeros((niter, 120, 100))

    # The main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        
        train_loss[it] = solver.net.blobs['loss_f'].data
        
        solver.test_nets[0].forward(start='data')
        
        output[it] = solver.test_nets[0].blobs['ip_f'].data.copy()
        
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            data = solver.test_nets[0].blobs['ip_f'].data.copy()
            loss_f = solver.test_nets[0].blobs['loss_f'].data.copy()
            acc_f = solver.test_nets[0].blobs['accuracy_f'].data.copy()
            
            print loss_f, (acc_f * 100)
            
    # Train and test done, outputing convege graph
    

def solver_process(envlist):
    
    style_solver_filename = os.path.join(envlist['caffe_root'],envlist['solver'])
    weights = os.path.join(envlist['caffe_root'],envlist['weights'])
    
    print "[ style solver name]", style_solver_filename
    style_solver = caffe.get_solver(style_solver_filename)
    #style_solver.net.copy_from(weights)    

    # For reference, we also create a solver that isn't initialized from
    # the pretrained ImageNet weights.
    #scratch_style_solver_filename = solver(style_net(train=True),True)
    #scratch_style_solver = caffe.get_solver(scratch_style_solver_filename)

    niter = 100
    print 'Running solvers for %d iterations...' % niter
    solvers = [('pretrained', style_solver)]
    loss, acc, weights = run_solvers(niter, solvers)
    print 'Done.'
    
    
def main():

    envlist = setEnvCifar100()
    #envlist = setEnv()
    
    test=False
    weights = process(envlist,test)    
    
    #run_SingleStepSolver(envlist)
    #solver_process(envlist)
    #startSolver(envlist)
    
if __name__ == "__main__":
    main()    
    
