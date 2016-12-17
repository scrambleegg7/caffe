# -*- coding: utf-8 -*-


import numpy as np
import os

import matplotlib.pyplot as plt


from AlzheimerClass import AlzheimerClass

from caffeBase.envParam import envParamAlz

def proc1():
    
 
    myenv = envParamAlz()
    
    read = True
    test = True

    AlzCls = AlzheimerClass(myenv,read,test)
    
    #print os.path.basename(__file__)
    
    #
    print "-- making Alzheimer image data [ proc1 ]........."    
#    AlzCls.makeH5Data(20000)
    AlzCls.makeImagelistFile(6200,0.3,3)
    #AlzCls.makeImagelistFile3ch(100,0.3,3)

    print "-- flatfile done........."

    return AlzCls
    
def main():
    AlzCls = proc1()
    
    
if __name__ == "__main__":
    main()