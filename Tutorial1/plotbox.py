# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from scipy.stats import norm

import matplotlib.pyplot as plt

def proc1():
    
    
    data = [1.30,1.48,1.50,1.48,1.45,1.44,1.53,1.55,1.49,1.40,
            1.57,1.52,1.52,1.65,1.58,1.20,1.69,1.54,1.38,1.03]
            
    #plt.hist(data)
            
    
    mu = np.mean(data)
    dev = np.std(data)
    print "mu:",mu
    print "median:",np.median(data)
    print np.percentile(data,25)
    print np.percentile(data,75)
    
    boxband = np.percentile(data,75) - np.percentile(data,25)
    print boxband * 1.5 + np.percentile(data,75)
    print np.percentile(data,25) - boxband * 1.5 
    
    print norm.ppf([0.25,0.75],mu,dev)
    
    
    
    plt.boxplot(data)
    plt.show()
    
    
    
def main():
    proc1()




if __name__ == "__main__":
    main()