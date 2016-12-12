# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from caffeBase.envParam import envParamAlz

import os

import caffe
import lmdb
import PIL.Image
from StringIO import StringIO
import numpy as np
import matplotlib.pyplot as plt

def read_lmdb(lmdb_file):

    lmdb_cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()


    idx = 0
    for key, value in lmdb_cursor:


        datum.ParseFromString(value)

        label = datum.label
        data = caffe.io.datum_to_array(datum)
        im = data.astype(np.uint8)

        print "-- im shape", im.shape, im.ndim
        # for colored
        #im = np.transpose(im, (2, 1)) # original (dim, col, row)
        im = im[0,:,:]
        print "-- im shape after change .... ", im.shape, im.ndim


        print "label ", label

        plt.imshow(im,cmap='gray',  interpolation='nearest')
        plt.show()

        idx += 1
        if idx > 5:
            break

    """
    for _, value in cursor:
        datum.ParseFromString(value)
        s = StringIO()
        s.write(datum.data)
        s.seek(0)

        yield np.array(PIL.Image.open(s)), datum.label
    """
def main():

    myenv = envParamAlz()
    lmdb_dir = os.path.join(myenv.envlist['datadir'], "alz_train_lmdb")
    read_lmdb(lmdb_dir)


if __name__ == "__main__":
    main()
