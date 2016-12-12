# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import cv2.cv as cv

import os

def process():
    
    
    cap = cv2.VideoCapture(0)

    width_ = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    height_ = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))  

    print "mac webcam width %d , hight : %d " % (width_, height_)
    
    rectsize_ = 256
    starty_ = height_ / 2 - (rectsize_ / 2)
    startx_ = width_ / 2 - (rectsize_ / 2)
    endxy_ = (startx_ + rectsize_, starty_ + rectsize_)

    
    print "x: %d y: %d " % (startx_, starty_)
    
    imdir = "/Users/donchan/Documents/Statistical_Mechanics/caffe/webCam/images"
    
    i = 0
    while(True):

    # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame,(startx_,starty_),endxy_,(255,0,0),2)

        dst = frame[starty_:starty_+rectsize_,startx_:startx_+rectsize_]
        filename = "camimage" + str(i) + ".jpg"
        
        if (i % 20) == 0:

            upath = os.path.join(imdir,filename)
            print "-- save file frame %d on %s" % (i,upath)
            cv2.imwrite(upath,dst)
        #pos    = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        #length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        print i
        #print "-- no of frames and fpsq : %d  %d" % (length, fps)
        
        cv2.imshow('frame',frame)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
def main():
    process()




if __name__ == "__main__":
    main()