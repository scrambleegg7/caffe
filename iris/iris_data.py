# -*- coding: utf-8 -*-

import subprocess
import platform
import copy

from sklearn.datasets import load_iris
import sklearn.metrics 
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import h5py
import caffe
import caffe.draw

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score



import google.protobuf

import os

def print_network(prototxt_filename, caffemodel_filename):
    '''
    Draw the ANN architecture
    '''
    _net = caffe.proto.caffe_pb2.NetParameter()
    f = open(prototxt_filename)
    google.protobuf.text_format.Merge(f.read(), _net)
    caffe.draw.draw_net_to_file(_net, prototxt_filename + '.png' )
    print('Draw ANN done!')

def load_data():
    '''
    Load Iris Data set
    '''
    data = load_iris()
    print(data.data)
    print(data.target)
    targets = np.zeros((len(data.target), 3))
    for count, target in enumerate(data.target):
        targets[count][target]= 1    
    #print(targets)

    new_data = {}
    #new_data['input'] = data.data
    new_data['input'] = np.reshape(data.data, (150,1,1,4))
    new_data['output'] = np.array(data.target)
    #print(new_data['input'].shape)
    #new_data['input'] = np.random.random((150, 1, 1, 4))
    #print(new_data['input'].shape)   
    #new_data['output'] = np.random.random_integers(0, 1, size=(150,3))    
    
    #print(new_data['input'])

    print "-- data shape : ", new_data['input'].shape
    print "-- label shape : ", data.target.shape
    return new_data


def save_data_as_hdf5(hdf5_data_filename, data):
    '''
    HDF5 is one of the data formats Caffe accepts
    '''
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data['input'].astype(np.float32)
        f['label'] = data['output'].astype(np.float32)
    

def read_hdf5_data(h5_filedir):
    
    with h5py.File(h5_filedir, 'r') as hf:
        datas = hf.get('data')
        labels = hf.get('label')
        
        print('List of arrays in this file: \n', hf.keys())
        print "-- read data shape", datas.shape
        #print type(datas)
        
        print "-- convert np data array from h5py data format"        
        np_data = np.array(datas)
        np_label = np.array(labels)
        print "-- converted array shape label : ", np_label.shape

    return np_data,np_label
    

def train(solver_prototxt_filename):
    
    caffe.set_mode_cpu()
    solver = caffe.get_solver(solver_prototxt_filename)
    solver.solve()


def get_predicted_outputs(deploy_prototxt_filename, caffemodel_filename, inputs):
    '''
    Get several predicted outputs
    '''
    outputs = []
    net = caffe.Net(deploy_prototxt_filename,caffemodel_filename, caffe.TEST)
    
    for data_item in inputs:
        
        out = net.forward(data=data_item)
        
        #print out[net.outputs[0]]
        outputs.append(copy.deepcopy(out[net.outputs[0]]))
        #print(input)
    
        #outputs.append(copy.deepcopy(get_predicted_output(deploy_prototxt_filename, caffemodel_filename, input, net)))
    return outputs    


def SaveAndTrain(new_data,tdir):
    
    hdf5_data_filename = "/Users/donchan/Documents/mydata/iris_data.hdf5"
    save_data_as_hdf5(hdf5_data_filename,new_data)
    
    datas,labels = read_hdf5_data(hdf5_data_filename)
    
    # Set parameters
    solver_prototxt_filename = os.path.join(tdir , 'iris_caffe_solver.prototxt')
    
    train(solver_prototxt_filename)        
    
    

def main():

    new_data = load_data()
    tdir = "/Users/donchan/Documents/Statistical_Mechanics/caffe/iris"
    
    train_test_prototxt_filename = os.path.join( tdir , 'iris_caffe_train_test.prototxt')
    
    
    # train(solver_prototxt_filename)    
    deploy_prototxt_filename  = os.path.join(tdir , 'iris_caffe_deploy.prototxt')
    
    caffe_model = os.path.join( tdir, "iris_caffe_iter_20000.caffemodel"  )


    print_network(train_test_prototxt_filename,caffe_model)
    

    iris_data = new_data['input']  
    iris_label = new_data['output']
    
    
    X_train,X_test,label_train,label_test = train_test_split(iris_data,iris_label)    
    
    results = get_predicted_outputs(deploy_prototxt_filename,caffe_model,X_test)

    #print "-- top 10 results", results[:10]
    predict_label = [np.argmax(l) for l in results]
    print "-- top 10 predicted label:", predict_label
    
    print "-- accuracy score", accuracy_score(label_test,predict_label)

    
    #number_of_samples = iris_data.shape[0]
    #number_of_outputs = iris_data.shape[1]
    #threshold = 0.0 # 0 if SigmoidCrossEntropyLoss ; 0.5 if EuclideanLoss

    #for output_number in range(number_of_outputs):
    #    predicted_output_binary = []
    #    for sample_number in range(number_of_samples):
            #print(predicted_outputs)
    #        print results[sample_number][0][output_number]

if __name__ == "__main__":
    main()
    