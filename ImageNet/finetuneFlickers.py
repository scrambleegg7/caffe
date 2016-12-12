# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
import caffe
import numpy as np
import matplotlib.pyplot as plt

from caffe import layers as L
from caffe import params as P

from caffe.proto import caffe_pb2



weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2



def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image
    
def setEnvCifar100():

    envlist = {}
    envlist['caffe_root'] = '/Users/donchan/caffe/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    envlist['caffe_model'] = 'cnn_snapshot_iter_150000.caffemodel'
    envlist['deploy'] = 'cnn_deploy.prototxt'
    envlist['meannpy'] = 'mean.npy'
    envlist['solver'] = 'cnn_solver_h5.prototxt'
    envlist['meanfile'] =  '/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/mean.binaryproto'
    #envlist['imagefile'] = lmdb_img # after loading image with caffe io load_image

    envlist['weights'] = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    envlist['imagenet_label_file'] = 'data/ilsvrc12/synset_words.txt'
    envlist['style_name'] = 'examples/finetune_flickr_style/style_names.txt'
    envlist['train'] = 'models/bvlc_reference_caffenet/train.prototxt'
    
    return envlist

# -*- coding: utf-8 -*-

import os
import caffe
import numpy as np
import matplotlib.pyplot as plt

from caffe import layers as L
from caffe import params as P

from caffe.proto import caffe_pb2



weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2



def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image
    
def setEnvCifar100():

    envlist = {}
    envlist['caffe_root'] = '/Users/donchan/caffe/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    envlist['caffe_model'] = 'cnn_snapshot_iter_150000.caffemodel'
    envlist['deploy'] = 'cnn_deploy.prototxt'
    envlist['meannpy'] = 'mean.npy'
    envlist['solver'] = 'cnn_solver_h5.prototxt'
    envlist['meanfile'] =  '/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/mean.binaryproto'
    #envlist['imagefile'] = lmdb_img # after loading image with caffe io load_image

    envlist['weights'] = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    envlist['imagenet_label_file'] = 'data/ilsvrc12/synset_words.txt'
    envlist['style_name'] = 'examples/finetune_flickr_style/style_names.txt'
    envlist['train'] = 'models/bvlc_reference_caffenet/train.prototxt'
    envlist['flickersolver'] = 'models/finetune_flickr_style/solver.prototxt'
    envlist['myflickersolver'] = 'models/finetune_flickr_style/mysolver.prototxt'
    
    return envlist

def loadWeightsBvlc(envlist):
    w_file = os.path.join(envlist['caffe_root'],envlist['weights'])    
    print "[ weights file ]",w_file
    assert os.path.exists(w_file)
    return w_file

def loadSolverFlicker(envlist):
    
    
    solv = os.path.join( envlist['caffe_root'], envlist['flickersolver'] )
    print "[ flicker solver file ]", solv
    return solv

def loadMySolverFlicker(envlist):
    
    
    solv = os.path.join( envlist['caffe_root'], envlist['myflickersolver'] )
    print "[ flicker solver file ]", solv
    return solv


def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            
            print "loss",name,s.net.blobs["loss"].data.copy()             
            print "accuracy",name,s.net.blobs["acc"].data.copy()             
            
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weight_dir = '/Users/donchan/caffe/caffe/models/finetune_flickr_style'
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel_' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    
    return loss, acc, weights


def main():
    envlist = setEnvCifar100()
    weights = loadWeightsBvlc(envlist)

    style_solver_filename = loadMySolverFlicker(envlist)
    
    style_solver = caffe.get_solver(style_solver_filename)
    #style_solver.net.copy_from(weights)
    #style_solver.set_phase_test()


    niter = 100
    print 'Running solvers for %d iterations...' % niter
    solvers = [('pretrained', style_solver)]
    loss, acc, weights = run_solvers(niter, solvers)
    print 'Done.'


    
if __name__ == "__main__":
    main()    