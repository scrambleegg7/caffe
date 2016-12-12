# -*- coding: utf-8 -*-

import caffe
#import leveldb
from caffe.proto import caffe_pb2
import numpy as np


testfile = 'test_mean.binarryproto'

# write 
mean_array = np.zeros((1,2,48,48))
mean_blobproto = caffe.io.array_to_blobproto(mean_array)

mean_blobproto.height = mean_array.shape[2]
mean_blobproto.channels = mean_array.shape[1]
mean_blobproto.width = mean_array.shape[3]
mean_blobproto.num = mean_array.shape[0]



f = open(testfile, 'wb')
f.write(mean_blobproto.SerializeToString())
f.close()

# read
mean_blobproto_new = caffe_pb2.BlobProto()
f = open(testfile, 'rb')
mean_blobproto_new.ParseFromString( f.read() )

x = np.array(mean_blobproto_new.data)
print "----shape with np array from mean_blobproto_new", x.shape
print "----mean blob data shape:" , np.asarray(mean_blobproto_new.data,dtype=np.float32).shape 

print "----h w c num", mean_blobproto_new.height,mean_blobproto_new.width,mean_blobproto_new.channels,mean_blobproto_new.num


mean_array_new = caffe.io.blobproto_to_array( mean_blobproto_new )

print type(mean_array_new)
print "----shape",mean_array_new.shape


f.close()