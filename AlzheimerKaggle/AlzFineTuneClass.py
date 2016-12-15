# -*- coding: utf-8 -*-



"""

    FineTune Alzheimer data model - Kaggle 

    based on bvlc model
    
    image size : 96 x 96 numpy array
    
    
"""

import numpy as np
import os

from caffeBase.envParam import envParamAlz

import platform as pl

#from time import time

import caffe

#from AlzheimerClass import AlzheimerClass

from caffe import layers as L
from caffe import params as P

class AlzFineTuneClass(object):

    def __init__(self,env=None,test=False):
        
        if env == None:
            self.env = envParamAlz()
        else:
            self.env = env

        self.test = test
        
        self.caffe_public_root = self.env.envlist['caffe-public_root']

        self.caffe_root = self.env.envlist['caffe_root']
        
        self.setInitialParams()
        self.setImageSetLabels()
        self.weightsFile = self.setupWeights()
        
    def getBvlcRefWeights(self):
        
        return self.weightsFile
        
        
    def setInitialParams(self):
        
        self.weight_param = dict(lr_mult=1, decay_mult=1)
        self.bias_param   = dict(lr_mult=2, decay_mult=0)
        self.learned_param = [self.weight_param, self.bias_param]

        self.frozen_param = [dict(lr_mult=0)] * 2

        
    def deprocess_net_image(self, image):
        
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

    def setImageSetLabels(self):

        imagenet_label_file = self.caffe_public_root + '/data/ilsvrc12/synset_words.txt'
        self.imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))
            
    
    def setupWeights(self):
        
        models = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
        models_file = os.path.join(self.caffe_public_root,models)
        
        return models_file
        
    def setLabels(self):
        
        disease_labels = ['Normal','MC','AD']
        return disease_labels
        
    def conv_relu(self,bottom, ks, nout, stride=1, pad=0, group=1,
              param="learned",
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    
        if param == "learned":
            param = self.learned_param
            
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
        return conv, L.ReLU(conv, in_place=True)

    def fc_relu(self, bottom, nout, param="learned",
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
        
        if param == "learned":
            param = self.learned_param
        
        fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
                        
        return fc, L.ReLU(fc, in_place=True)

    def max_pool(self,bottom, ks, stride=1):
        return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

    def caffenet(self,data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False,filename=None):
        """Returns a NetSpec specifying CaffeNet, following the original proto text
        specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
        n = caffe.NetSpec()
        
        n.data = data
                
        param = self.learned_param if learn_all else self.frozen_param
        
        n.conv1, n.relu1 = self.conv_relu(n.data, 11, 96, stride=4, param=param)
        n.pool1 = self.max_pool(n.relu1, 3, stride=2)
        n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
        
        n.conv2, n.relu2 = self.conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
        n.pool2 = self.max_pool(n.relu2, 3, stride=2)
        n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)

        n.conv3, n.relu3 = self.conv_relu(n.norm2, 3, 384, pad=1, param=param)

        n.conv4, n.relu4 = self.conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)

        n.conv5, n.relu5 = self.conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
        n.pool5 = self.max_pool(n.relu5, 3, stride=2)

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
        fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=self.learned_param)
        # give fc8 the name specified by argument `classifier_name`
        n.__setattr__(classifier_name, fc8)

        if not train:
            n.probs = L.Softmax(fc8)

        if label is not None:
            n.label = label
            n.loss = L.SoftmaxWithLoss(fc8, n.label)
            n.acc = L.Accuracy(fc8, n.label)
        # write the net to a temporary file and return its filename
            
        if filename == None:
            filename = "dummy.prototxt" 
        rootdir = os.path.join(self.env.envlist['datadir'],"finetune")
        outfile = os.path.join(rootdir, filename)
        f = open(outfile,'w')
        
        #with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        
        
        return f.name
        
    def alz_net(self,train=True, learn_all=False, subset=None,filename=None):

        if subset is None:
            subset = 'train' if train else 'test'
    
        source = self.caffe_root + '/finetune/%s.txt' % subset

        mean_file = os.path.join(self.caffe_public_root,'data/ilsvrc12/imagenet_mean.binaryproto')

        transform_param = dict(mirror=train, crop_size=227,mean_file= mean_file)
        
        # alzheimer original data size is 96x96..
        # then need to resize 227x227 if it applied to imagenet model...

        
        #transform_param = dict(mirror=train, crop_size=96,mean_file= mean_file)
        
        style_data, style_label = L.ImageData(
            transform_param=transform_param, source=source,
            batch_size=62, new_height=256, new_width=256, ntop=2)
        
        # num style is changed to 3
        
        if filename == None:
            filename_="alz_net.prototxt"
        else:
            filename_= filename
            
        return self.caffenet(data=style_data, label=style_label, train=train,
                    num_classes=3,
                    classifier_name='fc8_flickr',
                    learn_all=learn_all,filename=filename_)
    
    
    def untrained(self):
    
        
        """
        train is set to false......
        for untraining ...
        
        
            learn_all = False = Frozen 
            lr_mult = 0.0  except fc8 
            
            fc7input = n.relu6
            fc8input = n.relu7
            
            not dropping out relu6 / relu7
            
            n.probs = L.Softmax(fc8)
                        
        """
        untrained_style_net = caffe.Net(self.alz_net(train=False, subset='train'),
                                self.weightsFile, caffe.TEST)
    
        untrained_style_net.forward()
    
        alz_data_batch = untrained_style_net.blobs['data'].data.copy()

        alz_label_batch = np.array(untrained_style_net.blobs['label'].data, dtype=np.int32)

        return alz_data_batch,alz_label_batch
    
    def untrained_net(self):

        untrained_style_net = caffe.Net(self.alz_net(train=False, subset='train'),
                                self.weightsFile, caffe.TEST)
        
        return untrained_style_net
        
    
    def dummyImageSetNet(self):
        
    
        #dummy_data = L.DummyData(shape=dict(dim=[1, 3, 96, 96]))
        dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))


        imagenet_net_filename = self.caffenet(data=dummy_data, train=False)
        #print imagenet_net_filename
    
        imagenet_net = caffe.Net(imagenet_net_filename,self.weightsFile,caffe.TEST)

        return imagenet_net
        
    def disp_preds(self, net, image, labels, k=5, name='ImageNet'):

        #input_blob = net.blobs['data']
        
        net.blobs['data'].data[0, ...] = image
        probs = net.forward(start='conv1')['probs'][0]
        top_k = (-probs).argsort()[:k]
        print 'top %d predicted %s labels =' % (k, name)
        print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))

    def disp_imagenet_preds(self,net,image):
        
        imagenet_labels = self.imagenet_labels
    
        self.disp_preds(net, image, imagenet_labels, name='ImageNet')

    def disp_alz_preds(self,net, image):
        
        alz_labels = self.setLabels()

        self.disp_preds(net, image, alz_labels, k=3, name='alzheimer')    
        
        
    def trained(self):
        pass





        
        