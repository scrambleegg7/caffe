# -*- coding: utf-8 -*-

import caffe


def process():
    solverf = "/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/solver_hdf5.prototxt"

    solver = caffe.get_solver(solverf)

    print("Layers' features:")
    for k, v in solver.net.blobs.items():
        print k, v.data.shape
    
    
    print("Parameters and shape:")
    #print solver.net.params.items()
    for k, v in solver.net.params.items():
        print k,v[0].data.shape
        
    #sgdsolver = caffe.SGDSolver(solverf)

def main():

    process()


if __name__ == "__main__":
    main()
    