# This code is to save a single frame of video feed
# for analysis and creates an histogram of pixel intensity

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from datetime import datetime
import time
from os.path import isdir as folder_exists
from os import makedirs as create_folder


class SpeckleAcquisition():

    def __init__(self, filename):
        self.filename = filename

    def captureFrame(self):
        # Sets up a VideoCapture object
        video_feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Checks if the camera is accesible
        if not video_feed.isOpened():
            raise IOError("Cannot connect to camera")
        video_feed.set(15, -3) # sets exposure time to 2^{-8} seconds
        while(True):
            return_value, frame = video_feed.read() # Assigns the current frame to "frame" variable
            if return_value == True:
                cv2.imshow("Frame", frame) # Displays current frame
                if cv2.waitKey(1)%256 == 32: # Press "SPACE" to save frame
                    relative_path = f"FigureTest/{self.filename}.tiff" # Where the image will be saved and what format
                    cv2.imwrite(relative_path, frame) # Saves frame
                    print(f"The image was successfully saved as {relative_path}")
                if cv2.waitKey(1)%256 == 27: # Hold "ESC" to stop video feed
                    break
            else:
                break
        video_feed.release()
        cv2.destroyAllWindows()

    def captureImages(self, folder, delta_t):
        # Sets up a VideoCapture object
        video_feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # Checks if the camera is accesible
        if not video_feed.isOpened():
            raise IOError("Cannot connect to camera")
        video_feed.set(15, -3) # sets exposure time to 2^{-8} seconds
        begin = time.time()
        while(True):
            return_value, frame = video_feed.read() # Assigns the current frame to "frame" variable
            if return_value == True:
                cv2.imshow("Frame", frame) # Displays current frame
                end = time.time()
                if end-begin>= delta_t:  # When 5 seconds have passed
                    begin = end
                    moment = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    if not folder_exists(f"FigureTest/{folder}/"):
                        create_folder(f"FigureTest/{folder}/")
                    relative_path = f"FigureTest/{folder}/{self.filename}_{moment}.tiff" # Where the image will be saved and what format
                    cv2.imwrite(relative_path, frame) # Saves frame
                    print(f"The image was successfully saved as {relative_path}")
                if cv2.waitKey(1)%256 == 27: # Hold "ESC" to stop video feed
                    break
            else:
                break
        video_feed.release()
        cv2.destroyAllWindows()


class SpeckleAnalysis():
    """
    This class contains the functions required to analyse the intensity distribution
    of spekle patterns after subtracting the background of said speckle pattern. 
    """
    

    def getImage(ImagePath):
        image = tiff.imread(ImagePath)[:,:,0]
        return np.int32(image)

    def imageSubstraction(speckle, uniform):
        difference = speckle - uniform
        minPixel = np.amin(difference)
        difference -= minPixel
        return difference

    def histogram(SpeckleImage, filename = None):
        plt.hist(SpeckleImage.flatten(), bins=256, range=(0, 255)) # Creates histogram of pixel intensity
        if filename:
            plt.savefig(f"FigureTest/{filename}Histogram.png") # Saves the histogram
        plt.show() # Shows the histogram

