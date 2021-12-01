import glob
import os
import cv2
from shutil import copy
a=0
for directory in [x[0] for x in os.walk("./Dataset/Cascade_Detector/101_ObjectCategories/")]:
    fnames = glob.glob(directory+"/*.jpg")
    for fName in fnames:
        image=cv2.imread(fName)
        if image.shape[0]>=192 and image.shape[1]>=168:
            fileName=fName.split("/")[5];
            print(fileName)
            a=a+1
            cv2.imwrite("./Dataset/Cascade_Detector/negative_images/"+str(a)+fileName,image[0:192,0:168])
