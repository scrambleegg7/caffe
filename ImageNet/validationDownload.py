# -*- coding: utf-8 -*-



#wget -c http://www.image-net.org/challenges/LSVRC/2012/nonpub/ILSVRC2012_img_val.tar


import os
import numpy as np
import csv
import pandas as pd

import urllib2
import urllib

def process():



    geturl = "wget -c http://www.image-net.org/challenges/LSVRC/2012/nonpub/ILSVRC2012_img_val.tar"
    
    os.system(geturl)
    
    
def main():
    process()
    

if __name__ == "__main__":
    main()