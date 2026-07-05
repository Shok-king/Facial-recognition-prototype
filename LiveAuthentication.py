import cv2
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import numpy as np 
from Reg_NEWUSER import get_face, new_face
from Reg_NEWUSER import file

database = open(file, 'r')
database = np.loadtxt(database, delimiter= ",")
facedetect = False
while not facedetect:
    get_face()
    for compare_data in new_face:
        for stored_data in database:
            if compare_data == stored_data:
                print("Face matched")
                facedetect = True
                import time
                seconds = 3
                if time.time() > seconds:
                    facedetect = False
            else:
                print("face not matched")


