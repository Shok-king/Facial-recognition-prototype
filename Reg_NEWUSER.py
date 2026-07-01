from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import cv2 
import numpy as np

# Everything here will be a one time set up as they are the constructor method that needs to be initialised before the loop starts
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
New = False
#==========================
#The while loop will constantly check for new frames from the kinect2 sensor, when that new frame is available it converts that picture into grey scale for image processing using haars cascade classifier.
while not New:
    if kinect.has_new_color_frame():
        raw_data = kinect.get_last_color_frame()
        raw_data = raw_data.reshape((1080, 1920, 4))
        image_BGR = cv2.cvtColor(raw_data, cv2.COLOR_BGRA2BGR)
        image_GREY = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)

    face = cascade.detectMultiScale(image_GREY)
    New = True
for i[x, y, w, h] in face(0, 3):
    print(i)