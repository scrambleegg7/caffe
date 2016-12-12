# -*- coding: utf-8 -*-


import os
import numpy as np
import csv
import pandas as pd

import urllib2
import urllib
import Queue
import glob

import threading

from os.path import exists

class DownloadThread(threading.Thread):
    def __init__(self, queue, skip_existing):
        super(DownloadThread, self).__init__()
        self.queue = queue
        self.skip_existing = skip_existing

    def run(self):
        while True:
            #grabs url from queue
            url, path = self.queue.get()

            if self.skip_existing and exists(path):
                # skip if requested
                self.queue.task_done()
                continue

            try:
                urllib.urlretrieve(url, path)
            except IOError:
                print "Error downloading url '%s'." % url

            #signals to queue job is done
            self.queue.task_done()


def download_urls(url_and_path_list, num_concurrent, skip_existing):
    # prepare the queue
    queue = Queue.Queue()
    for url_and_path in url_and_path_list:
        queue.put(url_and_path)

    # start the requested number of download threads to download the files
    threads = []
    for _ in range(num_concurrent):
        t = DownloadThread(queue, skip_existing)
        t.daemon = True
        t.start()

    queue.join()


def process(url,targetDir,filename):

    #testurl = "http://img.photobucket.com/albums/v165/cottonmanifesto/Jef2/riverway06140602.jpg"
    #wget_ = "wget -c "
    #file_name = testurl.split('/')[-1]
    #fileDir = "/Volumes/synology/caffe/data/ImageNet/train/"
    file_name = targetDir + "/" + filename
    print "url",url
    print "filename",file_name
    try :
        urllib.urlretrieve (url, file_name)
    
    except (ValueError, RuntimeError), e:
        print "Error: %s :%s" % (e, url)


def readFiles(mydir,test=False):
    
    mydir = mydir + "/" + "*.txt"
    myFiles = []
    files = glob.glob(mydir)
    for idx,fd in enumerate(files):
        
        file_name = fd.split('/')[-1]
        file_name = file_name.split('¥')[0]
        
        #print "-- wnid filename : ",file_name
        directory = "/Volumes/synology/caffe/data/ImageNet/train/" + file_name        
        if not os.path.exists(directory):
            #print "%s dir not found" % directory
            os.makedirs(directory)
        with open(fd, 'r') as f:
            reader = csv.reader(f)
            queue = Queue.Queue()            
            for i,row in enumerate(reader):
                for r in row:
                    #print r
                    if r is not None:
                        nfile_name = file_name + "_" + str(i) + ".JPEG"
                        #print nfile_name
                        #os.chdir(directory)
                        path = directory + "/" + nfile_name
                        
                        queue.put( (r,path) )
                        #process(r,directory,nfile_name)
        
        

    #for url_and_path in url_and_path_list:
        

    # start the requested number of download threads to download the files
            threads = []
            num_concurrent = 3
            for _ in range(num_concurrent):
                skip_existing = True
                t = DownloadThread(queue, skip_existing)
                t.daemon = True
                t.start()

            queue.join()

        # for confirmation of image transpose
        #im = np.transpose(myImage,(1,2,0))
        #fig = plt.figure()
        #plt.imshow(im)        
        
        #myFiles.append(df)
        
        if idx > 2 and test == True:
            break
        
    

def main():
    
    mydir = "/Users/donchan/Documents/Statistical_Mechanics/caffe/ImageNet/wnid"
    readFiles(mydir,True)
    #process()


if __name__ == "__main__":
    main()
    

    