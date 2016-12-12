# -*- coding: utf-8 -*-

import caffe
import os
import numpy as np

import sys
from caffe.proto import caffe_pb2

def classify():
    
    pass
    
def main():


    net_path = 'prescript_quick.prototxt'
    model_path = 'prescript_quick_iter_4000.caffemodel.h5'
    mean_path = 'prescription_mean.binaryproto'
    
    map_ = {}
    for i in range(50):
        map_[i] = i
        
    mean_blob = caffe_pb2.BlobProto()
    with open(mean_path) as f:
        mean_blob.ParseFromString(f.read())

    mean_array = np.asarray(
                    mean_blob.data,
                    dtype=np.float32
                    ).reshape(
                    (mean_blob.channels, mean_blob.height, mean_blob.width)
                    )

    classifier = caffe.Classifier(
        net_path,
        model_path,
        mean=mean_array,
        raw_scale=255)

    # sys.argv[0] is script name
    # sys.argv[1] is image file
    image = caffe.io.load_image(sys.argv[1])

    # predict(target_image, oversample=True|False)
    # oversample's default value is True
    predictions = classifier.predict([image], oversample=False)
    answer = np.argmax(predictions) # get max value's index

    # RESULT
    print("====================================")
    print("possibility of each categoly")
    for index, prediction in enumerate(predictions[0]):
        print (str(index)+"("+ str(map_[index]) +"): ").ljust(15) + str(prediction)
    print("====================================")
    print("I guess this image is [" + str(map_[answer]) + "]")

    
if __name__ == "__main__":
    main()
