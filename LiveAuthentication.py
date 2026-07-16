import cv2
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import numpy as np 
import time
from Reg_NEWUSER import get_face

import time

def hamming(stored_position, recieved_position):
    counter = 0
    for x, y in zip(stored_position, recieved_position):
        if x != y:
            counter += 1
    return counter

def conversion(everyline):
        parts = everyline.split(":")
        NumberVAR = parts[1].split(",")
        convert = []
        for v in NumberVAR:
            v = int(v)
            convert.append(v)
        return parts[0], convert

while True:
    print("Warming up camera module...")
    time.sleep(2)
    liveface = get_face()
    Matched = False
    file = open("database.txt", "r")
    for everyline in file:
        ID_name, convert = conversion(everyline)
        counter = hamming(convert, liveface)
    
        if counter <= 500:
            print(f"face matched welcome {ID_name}")
            Matched = True
            break
    else:
        print("Match not found rejected.")
     
    file.close()
