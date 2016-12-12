# -*- coding: utf-8 -*-

import os
import numpy as np
import csv
import pandas as pd

import urllib2
import urllib

def process():
    
    wordtxt = "imagenet.synset.obtain_synset_list.txt"
    
    df= pd.read_table(wordtxt)
    df.columns = ["wnid"]
    #print df.head()
    
    winds = list(df["wnid"])
    totals = len(winds)
    for idx,w in enumerate(winds):
        
        
        data = {}
        data["wnid"] = str(w)
        url_values = urllib.urlencode(data)
        url = "http://www.image-net.org/api/text/imagenet.synset.geturls"
        full_url = url + '?' + url_values

        #file_name = url.split('=')[-1]
        response = urllib2.urlopen(full_url)
        
        print  "%d of %d : %s"   %  (idx , totals, response.geturl() )       
        the_page = response.read()
        #f = open(file_name, 'wb')
        #meta = u.info()
        #file_size = int(meta.getheaders("Content-Length")[0])
        #print "Downloading: %s Bytes: %s" % (file_name, file_size)
        #print the_page

        f = open("./wnid/" +  str(w) + "¥.txt", "w" )
        f.write(the_page)
        f.close()

        #if idx > 2:
        #    break

def main():
    process()
    
    
if __name__ == "__main__":
    main()