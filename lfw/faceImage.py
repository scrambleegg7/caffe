import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from django.http import request
from os import rename
import requests
import shutil

import csv


class lfw(object):

    def __init__(self,direct):
	self.lfw_ = False
    	self.direct = None

	self.images = []
	self.genderData = []

	self.genders = self.loadGender()
	self.setDirectory(direct)

    def trainingSplit(self,test_rate = 0.3):

	np.random.shuffle(self.genderData)

	train_size = int(5000. * (1. - test_rate))
	test_size =  int(5000. * test_rate)

	print "** training size ** ", train_size
	print "** testing size ** ", test_size

	tr_ = self.genderData[:train_size]
	te_ = self.genderData[train_size:5000]

	return tr_,te_

    def writeGenderFile(self):

	rootdir = "/Users/donchan/caffe/caffe/examples/lfw"

	train_,test_ = self.trainingSplit()

	outfile = os.path.join(rootdir, "train_gender.dat")
	f = open(outfile,'w')
	writer = csv.writer(f,delimiter='\t')
	writer.writerows(train_)
  
	outfile = os.path.join(rootdir, "test_gender.dat")
	f = open(outfile,'w')
	writer = csv.writer(f,delimiter='\t')
	writer.writerows(test_)

    def setDirectory(self,direct):
	self.mydir = direct


	for r,dirs,files in os.walk(self.mydir):
	    for f in files:
		fullfname = os.path.join(r,f)
		self.images.append(fullfname)

		fname = r.split("/")[-1]
		if fname in self.genders:
		    g = self.genders[fname]
		    path_ = fname + "/" + f
		    self.genderData.append([path_,g])

    def imageload(self,imagefile=None):
	if imagefile:
	    imagedata = os.path.join(self.mydir,imagefile)
	    print "image data:",imagedata
	    image = plt.imread(imagedata)
	    return image
	else:

	    for imagedata in self.images[:1000]:
		im = plt.imread(imagedata)
		#print "shape of lfw data:", im.shape


    def loadGender(self):

	genders = {}
	rootdir = "/Users/donchan/caffe/caffe/examples/lfw"
	gfilename = os.path.join(rootdir, "LFW-gender-folds.dat")
	f = open(gfilename,'r')
	reader = csv.reader(f,delimiter='\t')
	for r in reader:
	    fnames = r[0].split("/")
		
	    gender = 0 if r[2] == "M" else 1

	
	    if not fnames[0] in genders:
		genders[fnames[0]] = gender
	    	#print fnames[0],r[2]
	
	return genders

    def checkGender(self,name):

	pass

    def checkSex(self,fname):
	fileSplit = fname.split("_")

	result = requests.get("http://api.genderize.io?name=%s" % fileSplit[0])
        result = result.json()

	try:
            if float(result['probability']) > 0.9:
            	if result['gender'] == 'male':
		    print "--- ", result['name'],"male"
                    #shutil.copyfile(f,"%s/%s" % (maleFolder,file))
                elif result['gender'] == 'female':
		    print "--- ", result['name'],"female"
                    #shutil.copyfile(f,"%s/%s" % (femaleFolder,file))
        except Exception as e:
       	    print result['name']

    def makeH5Data(self):
	pass
	


def main():
    print "-- stat lfw class"

    lfw_ = lfw("/Users/donchan/caffe/caffe/examples/lfw/lfw_funneled")
    lfw_.writeGenderFile()


if __name__ == "__main__":
    main()
