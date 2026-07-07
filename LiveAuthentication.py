import cv2
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import numpy as np 
import time
from Reg_NEWUSER import get_face

face = get_face()
def similaritycheck():
    file = open("database.txt", "r")
    for everyline in file:
