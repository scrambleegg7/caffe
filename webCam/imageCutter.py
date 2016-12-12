# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:24:20 2016

@author: donchan
"""

import cv2
import matplotlib.pyplot as plt

import os

import numpy as np


def process():

    YYYYMMDD = "20160705"
    
    photoDir = "/Volumes/myShare/Prescription_Photo"
    
    myPhotoDir = photoDir + "/" + YYYYMMDD    
    myPhotoDir = "/Users/donchan/Documents/Statistical_Mechanics/caffe/webCam/images"
    
    testfile = "IMG_3382.JPG"    
    
    imageFile = os.path.join(myPhotoDir,testfile)
    img = plt.imread(imageFile)    
    
    print img.shape
    height_, width_, ch = img.shape

    rectsize_ = 256

    # center of image 256 256 rectange    
    starty_ = height_ / 2 - (rectsize_ / 2)
    startx_ = width_ / 2 - (rectsize_ / 2)
    endxy_ = (startx_ + rectsize_, starty_ + rectsize_)

    rectxy = np.zeros((2,2))
    rectxy[0:,] = (startx_,starty_)
    rectxy[1:,] = (endxy_[0],endxy_[1])
    
    h = []
    m = []
    h.append(rectxy)
    m.append(rectxy)
    
    hrectxy = rectxy
    mrectxy = rectxy

    while True:
        
        startxy_ = tuple(rectxy[0,].astype(int))        
        endxy_ = tuple(rectxy[1,].astype(int))
        
        #cv2.rectangle(img,startxy_,endxy_,(255,0,0),2)
        
        hrectxy = hrectxy - 256.
        mrectxy = mrectxy + 256.
        
        if len( hrectxy[  hrectxy > 0.0  ]    ) > 3:
            h.append(hrectxy)
        else:
            break
        
	if mrectxy[1,0] > height_ or mrectxy[1,1] > width_:
	    break
	else:
	    m.append(mrectxy)
    
    print h[-1][0,]
    print m[-1][1,]

    topxy =  h[-1][0,]
    bottomxy = m[-1][1,]

    hx = int( len(h) ) * 2  - 1
    wy = int( len(m) ) * 2  - 1

    print hx,wy

    leftx = topxy[0]
    topy = topxy[1]

    rightx = bottomxy[0]
    bottomy = bottomxy[1]

    for hxx_ in range(hx):

        stx = leftx + ( hxx_ * 256)
        edx = stx + 256

        sty = topy 

        for wyy_ in range(wy):

            sty = topy + (wyy_ * 256)
            edy = sty + 256
	
            cv2.rectangle(img,(int(stx),int(sty)),(int(edx),int(edy)),(255,0,255),2)
            
            print stx,sty,edx,edy


    
    plt.imshow(img)
    plt.show()

    #cv2.imshow("testmage",img)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    cv2.destroyAllWindows()
    #width_ = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    #height_ = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))  
    # Capture frame-by-frame
        # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    
    
def main():
    process()




if __name__ == "__main__":
    main()
