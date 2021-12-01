import glob
import os
import cv2
from shutil import copy

for directory in [x[0] for x in os.walk("./Dataset/Cascade_Detector/CroppedYale/")]:
    print(directory)
    fnames = glob.glob(directory+"/*.pgm")
    for fName in fnames:
        print(fName+"   "+str(cv2.imread(fName).shape))
        copy(fName,"./Dataset/Cascade_Detector/positive_images/")
