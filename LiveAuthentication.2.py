import cv2
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import numpy as np 
import time
from Reg_NEWUSER import get_face

def hamming(stored_position, recieved_position):
    counter = 0
    for x, y in zip(stored_position, recieved_position):
        if x != y:
            counter += 1
    return counter

def conversion(everyline):
    parts = everyline.strip().split(":") # Added .strip() to clean line breaks
    NumberVAR = parts[1].split(",")
    convert = []
    for v in NumberVAR:
        if v.strip(): # Ensure the value isn't an empty space
            convert.append(int(v))
    return parts[0], convert
timer = True
while timer == True:
    print("Warming up camera module...")
    time.sleep(2)
    liveface = get_face()
    
    file = open("database.txt", "r")
    found_match = False
    
    for everyline in file:
        if not everyline.strip(): continue # Skip empty lines
        ID_name, convert = conversion(everyline)
        counter = hamming(convert, liveface)
        
        # FIX: Added print to see your score
        print(f"Checking {ID_name}: Score = {counter}") 
        
        # FIX: Increased threshold to 1200
        if counter <= 800: 
            print(f"face matched welcome {ID_name}")
            found_match = True
            break
        timer = False
    time.sleep(5)
    timer = True

            
    file.close()
    
    if not found_match:
        print("Match not found rejected.")