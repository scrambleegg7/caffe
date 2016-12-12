# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import os

import caffe

from lfw import lfw

from caffeBase.envParam import envParamLFW

from caffeBase.imageProcessClass import imageProcessClass

import sys
import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version

from caffeBase.FicherClass import FicherClass



def process():
    
    env = envParamLFW()
    lfwCls = lfw(env) 
    
    facelist = lfwCls.loadFaceDataTopN(1000)
    print "-- target image file", facelist[-10:]
    facelist = facelist[-10:]
    return facelist


def main():
    flist = process()
    
    ficherCls = FicherClass(flist,5,True)
    ficherCls.process()


if __name__ == "__main__":
    main()
