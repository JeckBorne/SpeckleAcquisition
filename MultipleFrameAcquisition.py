# This code is to save a multiple frames of video feed
# for analysis and creates an histogram of pixel intensity

import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Sets up a VideoCapture object
video_feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Checks if the camera is accesible
if not video_feed.isOpened():
    raise IOError("Cannot connect to camera")

video_feed.set(15, -8) # sets exposure time to 2^{-8} seconds
begin = time.time()
filename = input("Under what name should the images be saved?: ") # The name of the image
while(True):
    return_value, frame = video_feed.read() # Assigns the current frame to "frame" variable
    if return_value == True:
        cv2.imshow("Frame", frame) # Displays current frame
        end = time.time()
        if end-begin>= 5:  # When 5 seconds have passed
            begin = end
            moment = datetime.now().strftime("%H-%M-%S")
            relative_path = f"Figure/TestVIdeofeed/{filename}_{moment}.png" # Where the image will be saved and what format
            relative_path_histo = f"Figure/TestVIdeofeed/{filename}Histogram_{moment}.png" # Where the histogram will be saved and what format
            cv2.imwrite(relative_path, frame) # Saves frame
            print(f"The image was successfully saved as {relative_path}")
            plt.hist(frame.flatten(), bins=256, range=(0, 255)) # Creates histogram of pixel intensity
            plt.savefig(relative_path_histo) # Saves the histogram
            plt.clf()
            print(f"The histogram was successfully saved as {relative_path_histo}")
        if cv2.waitKey(1) & 0xFF == ord("q"): # Hold "q" to stop video feed
            break
    else:
        break
video_feed.release()
cv2.destroyAllWindows()