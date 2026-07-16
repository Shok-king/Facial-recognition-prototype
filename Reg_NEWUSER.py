from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import cv2 
import numpy as np

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face():
    while True:
        if kinect.has_new_color_frame():
            raw_data = kinect.get_last_color_frame()
            raw_data = raw_data.reshape((1080, 1920, 4))
            image_BGR = cv2.cvtColor(raw_data, cv2.COLOR_BGRA2BGR)
            image_GREY = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)
            
            cv2.imshow("Kinect Feed", image_GREY)
            cv2.waitKey(1)

            face = cascade.detectMultiScale(image_GREY)
            
            if len(face) > 0:
                # Face detected, now process it immediately
                new_face = []
                for (x, y, w, h) in face:
                    face_cropped = image_GREY[y:y+h, x:x+w]
                    thresholded = cv2.resize(face_cropped, (64, 64))
                    thresholded = cv2.adaptiveThreshold(thresholded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
                    
                    # Flatten the 2D array into 1D
                    new_face = thresholded.flatten().tolist()
                
                cv2.destroyAllWindows()
                return new_face

def stored_biometric():
    prompt = input("Create ID for new user:\n")
    data = get_face()
    
    # Save to file
    with open("database.txt", "a") as file:
        file.write(prompt + ":" + ",".join(map(str, data)) + "\n")
    print("User saved successfully!")

if __name__ == "__main__":
    stored_biometric()