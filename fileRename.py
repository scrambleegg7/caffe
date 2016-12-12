# -*- coding: utf-8 -*-

import os
import numpy as np
import glob
import shutil



def process(test=False):

    if os.name != "nt":
        mydir = "/Volumes/synology/caffe/data/ImageNet/wnid"
        mydir = "/Users/donchan/Documents/Statistical_Mechanics/caffe/ImageNet/wnid"
        myNewdir = "/Volumes/synology/caffe/data/ImageNet/wnidnew"


    mydir = mydir + "/" + "*.txt"
    myFiles = []
    files = glob.glob(mydir)
    for idx,fd in enumerate(files):
        print "old",fd
        
        file_name = fd.split('/')[-1]
        file_name = file_name.split('¥')[0]

        new_file = myNewdir + "/" +file_name + ".txt"
        print "new",new_file
        #os.rename(fd,new_file)
        shutil.copyfile(fd,new_file)        
        if idx > 2 and test:
            break
    
    
    
def main():
    test = False
    process(test)



    

if __name__ == "__main__":
    main()