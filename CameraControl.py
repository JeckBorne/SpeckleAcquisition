# This code is to save a single frame of video feed
# for analysis

import cv2


# Sets up a VideoCapture object
video_feed = cv2.VideoCapture(0)
# Checks if the camera is accesible
if not video_feed.isOpened():
    raise IOError("Cannot connect to camera")

video_feed.set(15, -8) # sets exposure time to 2^{-8} seconds

while(True):
    ret, frame = video_feed.read() # Assigns the current frame to "frame" variable
    if ret == True:
        cv2.imshow("Frame", frame) # Displays current frame
        if cv2.waitKey(1) & 0xFF == ord("s"): # Press "s" to save frame
            filename = input("Under what name should the image be saved?  ") # The name of the image
            relative_path = f"Figure/{filename}.png" # Where the image will be saved and what format
            cv2.imwrite(relative_path, frame) # Saves frame
            print(f"The image was successfully saved as {relative_path}")
            break
        if cv2.waitKey(1) & 0xFF == ord("q"): # Press "q" to stop video feed
            break
    else:
        break
video_feed.release()
cv2.destroyAllWindows()