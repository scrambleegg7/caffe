# -*- coding: utf-8 -*-


import cv2
import matplotlib.pyplot as plt
import urllib

import numpy as np
import urllib2



def process():


    stream = urllib.urlopen('http://192.168.1.154/frame.mjpg')
    bytes = ""
    while True:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow('i', i)
            if cv2.waitKey(1) == 27:
                exit(0)

# When everything done, release the capture
    #cap.release()
    cv2.destroyAllWindows()
    
    
def main():
    process()




if __name__ == "__main__":
    main()