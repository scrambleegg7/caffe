# -*- coding: utf-8 -*-

from caffeBase.envParam import envParamAlz



class AlzBaseClass(object):

    def __init__(self,env=None,test=False):
        
        if env == None:
            self.env = envParamAlz()
        else:
            self.env = env

        self.test = test
        
        self.caffe_public_root = self.env.envlist['caffe-public_root']

        self.caffe_root = self.env.envlist['caffe_root']
        
        self.datadir = self.env.envlist['datadir']

        
