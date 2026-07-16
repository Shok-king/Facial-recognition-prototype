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
            raw_data = raw_data.reshape((1080, 1920, 4))#This restructures the raw data fetched into a grid of 1080 pixel rows by 1920 pixels column where each pixel holds 4 values RGBA
            image_BGR = cv2.cvtColor(raw_data, cv2.COLOR_BGRA2BGR)
            image_GREY = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)

            face = cascade.detectMultiScale(image_GREY)#Checks the image for any facial contrast patterns using haar's like features which then creates an invisible box around those regions where coordinates for them are fetched in the form of x y and width height of those boxes.
            if len(face) > 0:
                    break
#This part when a face is detected it starts to crop out the background until only the face is there and then does image processing to reduce the data down to only the essential   
    new_face = []
    for i in face:
         x, y, w, h = i 
         face_cropped = image_GREY[y:y+h, x:x+w]#Crops out the background using the coordinate where it should only be the face that's shown. To get the second x y coordinates we add x and width, y and height to get our second x y coordinate pairs
         thresholded = cv2.resize(face_cropped, (64, 64))#Resizes the cropped image ina fixed 64 pixel row by 64 pixel column regardless of how big/small the original image was. This ensures that we can do a comparison between stored values with newly recieved values later on.
         thresholded = cv2.adaptiveThreshold(thresholded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)#Adaptive threshold used for lighting normalisation and to fetch only the essential data where pixels becomes either 0 or 255.
         for Firstlevel in thresholded:#in order to flatten a 2d array information to 1D array.
              for secondlevel in Firstlevel:
                   new_face.append(secondlevel)
    return new_face
#Way of storing the data and link it to an ID for easy identification when doing comparison.
def stored_biometric():
    dictionary = {}
    prompt = str(input("Create ID for new user:\n"))
    dictionary[prompt] = get_face()
    dictionary_stored_array = []
    for cleaneddata in dictionary[prompt]:
        cleaneddata = str(cleaneddata)
        dictionary_stored_array.append(cleaneddata)
     

    file = open("database.txt", "a")
    file.write(prompt + ":" + ",".join(dictionary_stored_array) + "\n")
    file.close()
if __name__ == "__main__":
    stored_biometric()
