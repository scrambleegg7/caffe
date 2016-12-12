# -*- coding: utf-8 -*-

import numpy as np

from caffeBase import caffeUserClass


class testA(object):
    
    def __init__(self,ustr):
        print "init testA"
        self.ustr = ustr
    
#    def __new__(self):        
#        print "new testA"

    def proc(self,name):
        print name

    def printstr(self):
        print self.ustr        

class testB(testA):
    
    def __init__(self,ustr):
        super(testB,self).__init__(ustr)


        print "init testB"
 
    def setName(self):
        self.proc("B name set")
#    def __new__(self):
#        print "new testB"

def main():

    b = testB("rome")
    b.setName()
    b.printstr()
            
    
if __name__ == "__main__":
    
    main()    
    
    