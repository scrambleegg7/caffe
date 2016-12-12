# -*- coding: utf-8 -*-

from caffeBase.caffeUserClass import caffeUserClass

from caffeBase.imageProcessClass import imageProcessClass

import matplotlib.pyplot as plt

import numpy as np


class lfwCaffeClass(caffeUserClass):
    def __init__(self,env):
        
        super(lfwCaffeClass,self).__init__(env)
        
        # env is set to self.env
        
        self.gender= {}  
        self.loadGender()
        
        self.res_blobs = ('loss', 'accuracy')
    
    def loadGender(self):
        self.gender[0] = "Male"
        self.gender[1] = "Female"
    
    def predictFromImageDB(self,labelname):
        
        net = self.setNetWithTrainedWeights()
        
        d, l = self.getDataAndLabel(net,labelname)
        
        targetN = 12
        image = d[targetN]
        
        print "test image shape", image.shape

        imageClass = imageProcessClass()
        meanarray = imageClass.meanbinary(self.env)
        im0 = imageClass.convertToCHW_Mean(image,meanarray,True)
        
        plt.imshow(im0)
        plt.show()
        
        net.blobs['data'].data[0, ...] = image
        probs = net.forward(start='conv1')['probs'][0]
        
        top_k = (-probs).argsort()
        #print top_k
        print 'top %d predicted %s labels =' % (2, labelname)
        #print 'probability',probs
        
        for i, p in enumerate(top_k):
            print " (%d) : %s   %.4f" % (i, self.gender[p], probs[p])
    
    def netblobAndParamChecker(self):

        net = self.setNetWithTrainedWeights()
        
        self.netblobsChecker(net)
        self.netParamsChecker(net)
            
    def classify(self,image):
        
        net = self.setNetWithDeploydWeights()

        transformer = self.getTransformer(net,True)
        transformed_image = transformer.preprocess('data', image)
                            
        #plt.imshow(image)
        #plt.show()
        

        # load mean file from caffemodel used on training
        #imageClass = imageProcessClass()
        #meanarray = imageClass.meanbinary(self.env)

        #uint8_image_ = np.transpose(image,(2,0,1)) 
        
        #net.blobs['data'].reshape(50,        # batch size
        #                  3,         # 3-channel (BGR) images
        #                  227, 227)  # image size is 227x227
        
        #out = net.forward_all(data=np.asarray([uint8_image_]) - meanarray)
        
        
        
        net.blobs['data'].data[...] = transformed_image

        ### perform classification
        output = net.forward()

        output_prob = output['probs'][0]  # the output probability vector for the first image in the batch

        print 'output probs:', output_prob
        print 'predicted class is:', output_prob.argmax()



        
    
    
        
        




