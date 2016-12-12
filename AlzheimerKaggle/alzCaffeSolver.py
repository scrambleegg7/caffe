# -*- coding: utf-8 -*-


import numpy as np
import os

from caffeBase.envParam import envParamAlz

from caffeBase.myDirClassNocv2 import myDirClass

from AlzheimerClass import AlzheimerClass

from time import time

import matplotlib.pyplot as plt


import caffe



def train(solver_prototxt_filename):
    '''
    Train the ANN
    '''
    caffe.set_mode_cpu()
    solver = caffe.get_solver(solver_prototxt_filename)
    solver.solve()



def main():

    '''
    This is the main function
    '''
    # tdir = "/Users/donchan/Downloads/NNData/iris_caffe/"
    tdir = "./"
    # Set parameters    
    #solver_prototxt_filename = tdir + 'lenet_alz_solver.prototxt'
    solver_prototxt_filename = tdir + 'alz_lmdb_quick_solver.prototxt'

    train(solver_prototxt_filename)
    #readH5Data()

if __name__ == "__main__":
    main()
    #cProfile.run('main()') # if you want to do some profiling
    
    