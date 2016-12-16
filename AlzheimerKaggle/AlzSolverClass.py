# -*- coding: utf-8 -*-

import numpy as np
import os

from caffeBase.envParam import envParamAlz

import platform as pl

from time import time

import caffe

from caffe.proto import caffe_pb2

from AlzheimerClass import AlzheimerClass
from AlzFineTuneClass import AlzFineTuneClass
from AlzBaseClass import AlzBaseClass

from caffe import layers as L
from caffe import params as P


class AlzSolverClass(AlzBaseClass):

    def __init__(self,env=None,test=False):
        
        super(AlzSolverClass,self).__init__(env,test)
        
        self.alzFinetuneCls = AlzFineTuneClass(env,test)
        
        
        self.finetunedir = os.path.join(self.datadir,"finetune")
        self.weight_dir = self.finetunedir
        
        if test:
            print self.finetunedir
            print self.weight_dir
        
        
        
    def createSolver(self,train_net_path, test_net_path=None, base_lr=0.0001,filename=None):
        
        s = caffe_pb2.SolverParameter()

        # Specify locations of the train and (maybe) test networks.
        s.train_net = train_net_path
    
        if test_net_path is not None:
            s.test_net.append(test_net_path)
            s.test_interval = 1000  # Test after every 1000 training iterations.
            s.test_iter.append(100) # Test on 100 batches each time we test.

        # The number of iterations over which to average the gradient.
        # Effectively boosts the training batch size by the given factor, without
        # affecting memory utilization.
        s.iter_size = 1
    
        s.max_iter = 100000     # # of times to update the net (training iterations)
    
        # Solve using the stochastic gradient descent (SGD) algorithm.
        # Other choices include 'Adam' and 'RMSProp'.
        s.type = 'SGD'

        # Set the initial learning rate for SGD.
        s.base_lr = base_lr

        # Set `lr_policy` to define how the learning rate changes during training.
        # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
        # every `stepsize` iterations.
        s.lr_policy = 'step'
        s.gamma = 0.1
        s.stepsize = 20000

        # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
        # weighted average of the current gradient and previous gradients to make
        # learning more stable. L2 weight decay regularizes learning, to help prevent
        # the model from overfitting.
        s.momentum = 0.9
        s.weight_decay = 5e-4

        # Display the current training loss and accuracy every 1000 iterations.
        s.display = 1000

        # Snapshots are files used to store networks we've trained.  Here, we'll
        # snapshot every 10K iterations -- ten times during training.
        s.snapshot = 10000
        s.snapshot_prefix = os.path.join(self.finetunedir,'snapshot/finetune_alzheimer')
    
        # Train on the GPU.  Using the CPU to train large networks is very slow.
        #s.solver_mode = caffe_pb2.SolverParameter.GPU
        s.solver_mode = caffe_pb2.SolverParameter.CPU
    
        # Write the solver to a temporary file and return its filename.
    
        #rootdir = os.path.join(self.env.envlist['datadir'],"finetune")
        outfile = os.path.join(self.finetunedir, filename)
        f = open(outfile,'w')
        #with tempfile.NamedTemporaryFile(delete=False) as f:
        
        f.write(str(s))
        
        
        return f.name
        
    def run_solvers(self,niter, solvers, disp_interval=10):
    
        """Run solvers for niter iterations,
        returning the loss and accuracy recorded each iteration.
        `solvers` is a list of (name, solver) tuples."""
        blobs = ('loss', 'acc')

        loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    
        for it in range(niter):
            for name, s in solvers:
                s.step(1)  # run a single SGD step in Caffe
                loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        
            if it % disp_interval == 0 or it + 1 == niter:
                loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            
                print '%3d) %s' % (it, loss_disp)     
    
        # Save the learned weights from both nets.
        
        
        weights = {}
        for name, s in solvers:
            
            filename = 'weights.%s.caffemodel' % name
            
            weights[name] = os.path.join(self.weight_dir, filename)

            s.net.save(weights[name])
    
        return loss, acc, weights
    
    
    def stepRunSolver(self):

        train_path = self.alzFinetuneCls.alz_net(train=True,filename="fulltrain.prototxt")
        weights = self.alzFinetuneCls.getBvlcRefWeights()

        sfilename = "solver.prototxt"
        solver_filename = self.createSolver(train_net_path=train_path,filename=sfilename)
        
        style_solver = caffe.get_solver(solver_filename)
        
        style_solver.net.copy_from(weights)
        
  

        sfilename = "scratch_solver.prototxt"
        scratch_solver_filename = self.createSolver(train_net_path=train_path,filename=sfilename)
        scratch_style_solver = caffe.get_solver(scratch_solver_filename)
        

        niter = 200
        print 'Running solvers for %d iterations...' % niter

        solvers = [('pretrained', style_solver),
                   ('scratch', scratch_style_solver)]
                   
                   
        loss, acc, weights = self.run_solvers(niter, solvers)
        print 'Done.'

        train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
        train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
        style_weights, scratch_style_weights = weights['pretrained'], weights['scratch']
    
    
    def eval_alzNet_acc(self):

        train_path = self.alzFinetuneCls.alz_net(train=False,filename="non_train.prototxt")
        
        
        weightslist = [ 'pretrained',  'scratch' ]
        test_iters = 10
        
        for name in weightslist:
            filename = 'weights.%s.caffemodel' % name
        
            weights = os.path.join(self.weight_dir, filename)

            test_net = caffe.Net(train_path, weights, caffe.TEST)
            
            accuracy = 0
            for it in xrange(test_iters):
                accuracy += test_net.forward()['acc']
            
            accuracy /= test_iters
    
    
            
            print "weightname:%s  accuracy: %.4f" % (name,accuracy) 


        
        
        
        
        
        
        