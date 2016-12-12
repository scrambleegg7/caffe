# -*- coding: utf-8 -*-

import cv2
import numpy as np

from scipy.stats import multivariate_normal

class FicherClass(object):
    
    def __init__(self,flist,N=10,test=False):
        
        self.test = test
        self.flist = flist
        self.N = N
        
        self.desc = []

        
    def process(self):

        descriptors = self.sift()
        if self.test:        
            print "-- concatenate shape --", descriptors.shape

        means,covs,weights = self.calc(descriptors)
        
        th = 1.0 / self.N
        means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
        covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
        weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])

        #np.save("means.gmm", means)
        #np.save("covs.gmm", covs)
        #np.save("weights.gmm", weights)
        return means, covs, weights        
        
    def sift(self):
        
        self.desc = []
        for f in self.flist:
            img = cv2.imread(f,0)
    
            _ , descriptor = cv2.SIFT().detectAndCompute(img, None)
            self.desc.append(descriptor)
        
        return np.concatenate(self.desc)
        
    
    def calc(self,descriptors):

        N=10
        means, covs, weights = self.dictionary(descriptors, N)
        
        return means,covs,weights
        
    
    def dictionary(self,descriptors, N):
        em = cv2.EM(N)
        em.train(descriptors)

        return np.float32(em.getMat("means")), \
		np.float32(em.getMatVector("covs")), np.float32(em.getMat("weights"))[0]
  
    def likelihood_statistics(self,samples, means, covs, weights):
        gaussians, s0, s1,s2 = {}, {}, {}, {}
        samples = zip(range(0, len(samples)), samples)
	
        g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]
        for index, x in samples:
            gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

        for k in range(0, len(weights)):
            s0[k], s1[k], s2[k] = 0, 0, 0
            for index, x in samples:
                probabilities = np.multiply(gaussians[index], weights)
                probabilities = probabilities / np.sum(probabilities)
                
                s0[k] = s0[k] + self.likelihood_moment(x, probabilities[k], 0)
                s1[k] = s1[k] + self.likelihood_moment(x, probabilities[k], 1)
                s2[k] = s2[k] + self.likelihood_moment(x, probabilities[k], 2)

        return s0, s1, s2
        
    def likelihood_moment(self,x, ytk, moment):	
        
        x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
        return x_moment * ytk
        
    def fisher_vector_weights(self,s0, s1, s2, means, covs, w, T):
        return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

    def fisher_vector_means(self,s0, s1, s2, means, sigma, w, T):
        return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

    def fisher_vector_sigma(self,s0, s1, s2, means, sigma, w, T):
        return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])
  

    def fisher_vector(self,samples, means, covs, w):
        s0, s1, s2 =  self.likelihood_statistics(samples, means, covs, w)
        T = samples.shape[0]
        covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
        a = self.fisher_vector_weights(s0, s1, s2, means, covs, w, T)
        b = self.fisher_vector_means(s0, s1, s2, means, covs, w, T)
        c = self.fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
        fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
        fv = self.normalize(fv)
        return fv


    def normalize(self,fisher_vector):
        v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
        return v / np.sqrt(np.dot(v, v))
        
    def get_fisher_vectors_from_folder(self,folder, gmm):
        #files = glob.glob(folder + "/*.jpg")
        
        return np.float32([self.fisher_vector(image_descriptors(file), *gmm) for file in self.flist])

    def fisher_features(self,folder, gmm):
        folders = glob.glob(folder + "/*")
        features = {f : get_fisher_vectors_from_folder(f, gmm) for f in folders}
        return features


        