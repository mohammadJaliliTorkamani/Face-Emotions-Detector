import numpy as np
import cv2
import glob
from shutil import copy

face_detector = cv2.CascadeClassifier('../lbp/cascade.xml')



true_positives=0
false_positives=0
true_negatives=0
false_negatives=0

total_images=0
fnames = glob.glob('../Dataset/Cascade_Detector/mixed_images/*')
total_images=len(fnames)
fnames.sort()
for fname in fnames:
    image = cv2.imread(fname)
    image2 = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 2)
    cv2.putText(image2,'Do you see any face (y/n) ? ',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2) 
    cv2.imshow('question',image2)
    key = cv2.waitKey(0) 
    if key == ord('y'):
        if len(faces) >0 :
            true_positives=true_positives+1
            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.imwrite('../Dataset/Cascade_Detector/filter_result/true_positives/'+fname.split("/")[-1],image)
        else:
            false_negatives=false_negatives+1
            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)    
            print("false negative detected : "+fname)
            cv2.imwrite('../Dataset/Cascade_Detector/filter_result/false_negatives/'+fname.split("/")[-1],image)
    elif key == ord('n'):
        if len(faces) >0 :
            false_positives=false_positives+1
            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)    
            print("false positive detected : "+fname)
            cv2.imwrite('../Dataset/Cascade_Detector/filter_result/false_positives/'+fname.split("/")[-1],image)
        else:
            true_negatives=true_negatives+1
            cv2.imwrite('../Dataset/Cascade_Detector/filter_result/true_negatives/'+fname.split("/")[-1],image)

print("number of false positives : "+str(false_positives))
print("number of false negatives : "+str(false_negatives))
print("number of true positives  : "+str(true_positives))
print("number of true negatives  : "+str(true_negatives))
print("Total images : "+str(total_images))
