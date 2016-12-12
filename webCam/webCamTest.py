# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt



def process():
    
    
    cap = cv2.VideoCapture(0)
    
    cascade_dir = "/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades"
    haars_file = cascade_dir + '/haarcascade_frontalface_default.xml'
    haars_eye_file = cascade_dir + '/haarcascade_eye.xml'
    
    face_cascade = cv2.CascadeClassifier(haars_file)
    eye_cascade = cv2.CascadeClassifier(haars_eye_file)

    i = 0
    while(True):

    # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            print "Face -- x:%d y:%d w:%d h:%d" % (x,y,w,h)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                print "Eye -- x:%d y:%d w:%d h:%d" % (x,y,w,h)
            
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            #cv2.imshow('img',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        # Display the resulting frame
#        if i % 100 == 0:
            #print i
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