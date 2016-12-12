# -*- coding: utf-8 -*-

import numpy as np

import caffe

from caffe import layers as L
from caffe import params as P

from envParam import envParam

import matplotlib.pyplot as plt


from imageProcessClass import imageProcessClass

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]


class caffeUserClass(object):
    
    def __init__(self,env):
        
        self.env = env

        self.learned_param = None        
        self.setweightParam()
        
        self.imageClass = imageProcessClass()
        
    def netblobsChecker(self,net): 
        for layer_name, blob in net.blobs.iteritems():
            print layer_name + '\t' + str(blob.data.shape)
        
    def netParamsChecker(self,net):
        for layer_name, param in net.params.iteritems():
            print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
    
    def setweightParam(self):
        
        self.weight_param = dict(lr_mult=1, decay_mult=1)
        self.bias_param   = dict(lr_mult=2, decay_mult=0)
        self.learned_param = [self.weight_param, self.bias_param]

        self.frozen_param = [dict(lr_mult=0)] * 2

    def conv_relu(self, bottom, ks, nout, stride=1, pad=0, group=1,
             param=learned_param,
             weight_filler=dict(type='gaussian', std=0.01),
             bias_filler=dict(type='constant', value=0.1)):

        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
                         
        return conv, L.ReLU(conv, in_place=True)
        
    def fc_relu(self,bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):

        fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
        
        return fc, L.ReLU(fc, in_place=True)

    def max_pool(self,bottom, ks, stride=1):
        return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

    
    def caffenet(self,data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
        """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
        n = caffe.NetSpec()
        
        n.data = data

        frozen_param = self.frozen_param
        max_pool = self.max_pool
        learned_param = self.learned_param

        param = learned_param if learn_all else frozen_param
        n.conv1, n.relu1 = self.conv_relu(n.data, 11, 96, stride=4, param=param)
        n.pool1 = max_pool(n.relu1, 3, stride=2)
        n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2, n.relu2 = self.conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
        n.pool2 = max_pool(n.relu2, 3, stride=2)
        n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
        n.conv3, n.relu3 = self.conv_relu(n.norm2, 3, 384, pad=1, param=param)
        n.conv4, n.relu4 = self.conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
        n.conv5, n.relu5 = self.conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
        n.pool5 = max_pool(n.relu5, 3, stride=2)
        n.fc6, n.relu6 = self.fc_relu(n.pool5, 4096, param=param)

        if train:
            n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
        else:
            fc7input = n.relu6
        n.fc7, n.relu7 = self.fc_relu(fc7input, 4096, param=param)
        
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
        with open(self.env.getUserCaffeTrain(), 'w') as f:
        #with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(str(n.to_proto()))
        
        return f.name
    
    
    def style_net(self,train=True, learn_all=False, subset=None):
        
        if subset is None:
            subset = 'train' if train else 'test'
        
        caffe_root = '/Users/donchan/caffe/caffe/'        
        source = caffe_root + 'data/flickr_style/%s.txt' % subset
        transform_param = dict(mirror=train, crop_size=227,
        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
        style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=50, new_height=256, new_width=256, ntop=2)

        NUM_STYLE_LABELS = 1000        
        
        return self.caffenet(data=style_data, label=style_label, train=train,
                    num_classes=NUM_STYLE_LABELS,
                    classifier_name='fc8_flickr',
                    learn_all=learn_all)
        
    def setSolver(self):

        solver_filename = self.env.getSolver()
        weights = self.env.getCaffemodel()        
        
        solver = caffe.get_solver(solver_filename)
        solver.net.copy_from(weights)

        return solver

    def runSolver(self,niter=500):
        solver = self.setSolver()                

        
    def setNetWithTrainedWeights(self):
                        
        net = caffe.Net(self.env.getTest(), self.env.getCaffemodel(), caffe.TEST)
        return net

    def setNetWithDeploydWeights(self):
                        
        net = caffe.Net(self.env.getDeploy(), self.env.getCaffemodel(), caffe.TEST)
        return net


    def getDataAndLabel(self,net,labelname):
        net.forward()
        data_batch = net.blobs['data'].data.copy()
        label_batch = np.array(net.blobs[labelname].data, dtype=np.int32)
        
        return data_batch,label_batch

    def predictFromImageDB(self,labelname):
        
        net = self.setNetWithTrainedWeights()
        
        d, l = self.getDataAndLabel(net,labelname)
        
        targetN = 11
        image = d[targetN]
        label_ = l[targetN]
        
        net.blobs['data'].data[0, ...] = image
        probs = net.forward(start='conv1')['probs'][0]
        k = 1
        top_k = (-probs).argsort()[:k]
    
        print 'top %d predicted %s labels =' % (k, labelname)
        print 'probability',probs

    def eval_net(self, net, test_iters=10):
        
        accuracy = 0
        for it in xrange(test_iters):
            accuracy += net.forward()['accuracy']
        
        accuracy /= test_iters

        return float(accuracy)
        
    def getTransformer(self,net,test=False):
        
        meanarray = self.imageClass.meanArray(self.env)
        mu = meanarray.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

        if test:        
            print '- mean-subtracted values:', zip('BGR', mu)
            print "- net blobs data shape ", net.blobs['data'].data.shape
            print "- mu shhap - ", mu.shape, mu
        
        # create transformer for the input called 'data'
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

        transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
        transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

        return transformer


        
