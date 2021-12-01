import numpy as np
import cv2

face_detector = cv2.CascadeClassifier('../lbp/cascade.xml')
cap = cv2.VideoCapture(0)

while 1:
    ret, image = cap.read()	
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.5, 3 )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),1)    

    cv2.imshow('stream',image)
    
    key = cv2.waitKey(1) 
    if key == ord('q'): 
        break


cap.release()
cv2.destroyAllWindows()
