from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import cv2 
import numpy as np

# Everything here will be a one time set up as they are the constructor method that needs to be initialised before the loop starts
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
#==========================
#The while loop will constantly check for new frames from the kinect2 sensor, when that new frame is available it converts that picture into grey scale for image processing using haars cascade classifier.
def get_face():

    while True:
        if kinect.has_new_color_frame():
            raw_data = kinect.get_last_color_frame()
            raw_data = raw_data.reshape((1080, 1920, 4))
            image_BGR = cv2.cvtColor(raw_data, cv2.COLOR_BGRA2BGR)
            image_GREY = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)

            face = cascade.detectMultiScale(image_GREY)
            if len(face) > 0:
                    break
    new_face = []
    for i in face:
        x, y, w, h = i
        face_cropped = image_GREY[y:y+h, x:x+w]
        thresholded = cv2.resize(face_cropped, (64, 64))
        thresholded = cv2.adaptiveThreshold(thresholded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        new_face.append(thresholded)
    return new_face

new_face = get_face()
file = open("database.txt", "a")
for trueVAL in new_face:
     np.savetxt(file, trueVAL, delimiter=",")
     file.write("------------------------\n")
file.close()