import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from datetime import datetime
import time
import os
from PIL import Image
import re
from IPython.display import clear_output
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy import *
import pandas as pd

class SpeckleAnalysis():
    """This class is used to analyse the speckle patterns
    """    

    def getImage(ImagePath):
        """This fonction is used to get the pixel intensity of an image

        Args:
            ImagePath (string): The file path of the image

        Returns:
            float32: An array of pixel intensity
        """
        image = tiff.imread(ImagePath)   #[:,:,0]
        return np.float32(image)

    def correlation(folderpath, time_seperation=1, time_scale="seconds", start=0, end=1000, expo_fit=False):
        """Calculates the correlation between the first image of a folder and the follwing images. Shows the correlation graph.
    
        Args:
            folderpath (string): The path of the folder where the images are saved
            time_seperation (int, optional): The time seperation between the images. Defaults to 1.
            time_scale (string, optional): The time scale of the time seperation. Defaults to "seconds".
            start (int, optional): The start time of the correlation. Defaults to 0.
            end (int, optional): The end time of the correlation. Defaults to 30.
        """
        allfiles=os.listdir(folderpath)
        corr = []
        imlist= [filename for filename in allfiles if filename[-5:]==".tiff"]
        initialSpeckle = SpeckleAnalysis.getImage(f"{folderpath}/{imlist[start]}")[:,0:-1]
        Xstd, Xmean = np.std(initialSpeckle), np.mean(initialSpeckle)
        for image in SpeckleAnalysis.sortedAlphanumeric(imlist)[start:end]:
            clear_output(wait=True)
            speckleImageY = SpeckleAnalysis.getImage(f"{folderpath}/{image}")[:,0:-1]
            Ystd, Ymean = np.std(speckleImageY), np.mean(speckleImageY)
            corr.append(np.mean((initialSpeckle-Xmean)*(speckleImageY-Ymean))/(Xstd*Ystd))
            print(f"{len(corr)}/{len(allfiles[start:end])}")
        fig = plt.figure()
        fig.patch.set_facecolor("white")
        time = np.arange(0,len(corr)*time_seperation, time_seperation)
        if expo_fit == True:
            def exponential(x,a,b,c):
                return a * np.exp(-b * x) + c
    
            try:
                popt, pcov = curve_fit(exponential, time, corr)
                plt.plot(time, exponential(time, *popt), 'r-', label='Exponential Fit')
                print(f"The exponential fit parameters are: a={popt[0]}, b={popt[1]}, c={popt[2]}")
            except RuntimeError:
                print("Error - curve_fit failed")
        plt.title("Correlation through time of speckle pattern")
        plt.xlabel(f"Time ({time_scale})")
        plt.ylabel("Correlation coefficient (a.u.)")
        plt.scatter(time, corr, color="black")
        plt.show()
        
SpeckleAnalysis.getImage(specklePath)