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


weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
       
    envlist = setEnvCifar100()
    filename = os.path.join(envlist['caffe_root'],'mytrain.prototxt')    

    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
        
    with open(filename, 'w') as f:
        f.write(str(n.to_proto()))
        return f.name

def style_net(train=True, learn_all=False, subset=None):
    
    envlist = setEnvCifar100()
    caffe_root = envlist['caffe_root']
    
    if subset is None:
        subset = 'train' if train else 'test'
    source = caffe_root + '/data/flickr_style/%s.txt' % subset
    
    transform_param = dict(mirror=train, crop_size=227,
        mean_file=caffe_root + '/cnn_mean.binaryproto')

    #transform_param = dict(mirror=train, crop_size=227,
    #    mean_file=caffe_root + '/data/ilsvrc12/imagenet_mean.binaryproto')


    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=50, new_height=256, new_width=256, ntop=2)


    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=100,
                    classifier_name='fc8_flickr',
                    learn_all=learn_all)

def setEnvCifar100():

    envlist = {}
    envlist['caffe_root'] = '/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    envlist['caffe_model'] = 'cnn_snapshot_iter_150000.caffemodel'
    envlist['deploy'] = 'cnn_deploy.prototxt'
    envlist['meannpy'] = 'mean.npy'
    envlist['meanfile'] =  '/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/cnn_mean.binaryproto'
    envlist['solver'] = 'cnn_solver_h5.prototxt'
    envlist['train'] = 'cnn_train.prototxt_'
    envlist['weights'] = 'iter/cnn100_full_iter_43759.caffemodel.h5'
    envlist['bvlc_train'] = "untrained_bvlc.prototxt_"
    envlist['bvlc_root'] = '/Users/donchan/caffe/caffe/models/bvlc_reference_caffenet'     
    envlist['bvlc_model'] = 'bvlc_reference_caffenet.caffemodel'
    

    return envlist

def setCaffeNet(envlist):
    
    
    bvlc_caffemodel = os.path.join(envlist['bvlc_root'],envlist['bvlc_model'])
    bvlc_untrained = os.path.join(envlist['caffe_root'],envlist['bvlc_train'])

    bvlc_untrained = os.path.join(envlist['caffe_root'],'mytrain.prototxt')
    

    print "bvlc_caffemodel",bvlc_caffemodel
    print "bvlc_untrained",bvlc_untrained


    #net = caffe.Net(bvlc_caffemodel,bvlc_untrained,caffe.TEST)
    
    return net


def untrained(weights,imagenet_labels=None,style_labels=None):
    envlist = setEnvCifar100()
    
    #
    # bvlc 256x256 , cifar 32 x 32
    # thefore fine tune is not used due to different image size
    # 
    # copy train_val.prototxt --> then make new full connection layer (fc xxxx)
    # then start to generate learning.
    #
    #
    #caffenet_filename = style_net(train=False, subset='train_osx')
    
    caffenet_filename = os.path.join(envlist['caffe_root'],"untrained_bvlc.prototxt_")
    print "[ untrained caffe net ]", caffenet_filename    
    untrained_style_net = caffe.Net(caffenet_filename,
                                weights, caffe.TEST)
    untrained_style_net.forward()


def process():

    envlist = setEnvCifar100()
    bvlc_caffemodel = os.path.join(envlist['bvlc_root'],envlist['bvlc_model'])

    #untrained(bvlc_caffemodel)    
    #net = setCaffeNet(envlist)
    #filename = style_net(train=False, subset='train')
    #print "untrained filename",filename
    #print style_net(train=False, subset='train')
    
    dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
    print caffenet(data=dummy_data, train=False)

def main():
    process()
    
if __name__ == "__main__":
    main()
    