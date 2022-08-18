# This code is to save a single frame of video feed
# for analysis and creates an histogram of pixel intensity


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


class SpeckleAcquisition():

    def __init__(self, filename):
        self.filename = filename

    def captureFrame(self):
        """This function allows the capture of a single videoframe
        """
        # Sets up a VideoCapture object
        video_feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Checks if the camera is accesible
        if not video_feed.isOpened():
            raise IOError("Cannot connect to camera")
        video_feed.set(15, -4) # sets exposure time to 2^{-8} seconds
        video_feed.set(3,2592) # set the width and height
        video_feed.set(4,1944)
        print(f"Exposure Time = {video_feed.get(15)}\nResolution = {video_feed.get(3)}x{video_feed.get(4)}")
        while(True):
            return_value, frame = video_feed.read() # Assigns the current frame to "frame" variable
            if return_value == True:
                cv2.imshow("Frame", frame) # Displays current frame
                if cv2.waitKey(1)%256 == 32: # Press "SPACE" to save frame
                    relative_path = f"FigureTest\TestingExposureTime\TestingOpenCVExposureTime\{self.filename}.tiff" # Where the image will be saved and what format
                    cv2.imwrite(relative_path, frame) # Saves frame
                    print(f"The image was successfully saved as {relative_path}")
                    break
                if cv2.waitKey(1)%256 == 27: # Hold "ESC" to stop video feed
                    break
            else:
                break
        video_feed.release()
        cv2.destroyAllWindows()

    def captureImages(self, folder, delta_t):
        """This function allows to save many frames of a videofeed at a certain time interval

        Args:
            folder (string): Name of the folder where the pictures will be saved, the function will create a new file if the name provided does not exist
            delta_t (float): Time interval between each picture saved

        Raises:
            IOError: The code does not appear to be able to connect to the camera
        """
        # Sets up a VideoCapture object
        video_feed = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        # Checks if the camera is accesible
        if not video_feed.isOpened():
            raise IOError("Cannot connect to camera")
        video_feed.set(15, -9) # sets exposure time to 2^{-8} seconds
        video_feed.set(3,2592) # set the width and height
        video_feed.set(4,1944)
        begin = time.time()
        count = 0
        while count < 100:
            return_value, frame = video_feed.read() # Assigns the current frame to "frame" variable
            if return_value == True:
                cv2.imshow("Frame", frame) # Displays current frame
                end = time.time()
                if end-begin>= delta_t:  # When 5 seconds have passed
                    begin = end
                    count += 1
                    moment = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    if not os.path.isdir(f"FigureTest/{folder}/"):
                        os.makedirs(f"FigureTest/{folder}/")
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
    
    def rescale(image):
        """Rescales the array between 0 and 1

        Args:
            image (array): The array of the image pixel intensity

        Returns:
            array: The array of the image pixel intensity rescaled between 0 and 1
        """
        return np.float64(image/np.max(image))

    def meanFolder(folderpath):
        """The function calculates the individual mean of every tiff file in the specified folder

        Args:
            folderpath (string): The path of the folder where all the images are saved
        """
        allfiles=os.listdir(folderpath)
        imlist= [filename for filename in allfiles if filename[-5:]==".tiff"]
        MeanIC = {}
        for image in SpeckleAnalysis.sortedAlphanumeric(imlist):
            MeanIC[image]=np.mean(SpeckleAnalysis.getImage(f"{folderpath}/{image}"))
        print(f"The mean of each picture is: {MeanIC}")

    def medianFolder(folderpath):
        """The function calculates the individual median of every tiff file in the specified folder

        Args:
            folderpath (string): The path of the folder where all the images are saved
        """
        allfiles=os.listdir(folderpath)
        imlist= [filename for filename in allfiles if filename[-5:]==".tiff"]
        MedianIC = {}
        for image in SpeckleAnalysis.sortedAlphanumeric(imlist):
            MedianIC[image]=np.median(SpeckleAnalysis.getImage(f"{folderpath}/{image}"))
        print(f"The median of each picture is: {MedianIC}")

    def imageSubstraction(first, second):
        """This function is used to subtract one image to another, and adding an offset to assure that the minimum intensity is 0

        Args:
            first (float64): The numpy array of first image
            second (float64): The numpy array of the other image

        Returns:
            float: The numpy array of the difference of the images with the offset
        """
        difference = first - second
        minPixel = np.amin(difference)
        difference -= minPixel
        return difference

    def intensityDistribution(SpeckleImage, filepath = None):
        """Creates the histogram of the intensity distribution of the image

        Args:
            SpeckleImage (float64): Image of the speckle pattern
            filepath (string, optional): Path used to save the figure. Defaults to None.
        """
        fig = plt.figure()
        fig.patch.set_facecolor("white")
        plt.hist(SpeckleImage.flatten(), bins=256, color = "black") # Creates histogram of pixel intensity
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        if filepath:
            plt.savefig(filepath) # Saves the histogram
        plt.show() # Shows the histogram

    def sortedAlphanumeric(data):
        """Sorts the provided list

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanumKey = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanumKey)

    def NormalizeSpeckles(speckleImage, CorrectionImage):
        """This fonction uses the correction image to correct the gaussian distribution of the laser

        Args:
            speckleImage (float64): The image of the speckle pattern
            CorrectionImage (float64): The image used for the correction

        Returns:
            float64: Speckle pattern without the gaussian distribution
        """
        Normalized = speckleImage/CorrectionImage
        Normalized = np.float64(Normalized/np.max(Normalized))
        return Normalized
    
    def correlation(folderpath, correctionPath, start=0, end=1000):
        """Calculates the correlation between the first image of a folder and the follwing images. Prints the correlation\n
        values and shows the decorelation graph.

        Args:
            folderpath (string): The path of the folder where the images are saved
            correctionPath (string): The path to the correction image
        """
        allfiles=os.listdir(folderpath)
        correctionImage = SpeckleAnalysis.getImage(correctionPath)
        corr = []
        imlist= [filename for filename in allfiles if filename[-5:]==".tiff"]
        initialSpeckle = SpeckleAnalysis.NormalizeSpeckles(SpeckleAnalysis.getImage(f"{folderpath}/{imlist[start]}")[:,0:-1], correctionImage)
        Xstd, Xmean = np.std(initialSpeckle), np.mean(initialSpeckle)
        for image in SpeckleAnalysis.sortedAlphanumeric(imlist)[start:end]:
            clear_output(wait=True)
            speckleImageY = SpeckleAnalysis.NormalizeSpeckles(SpeckleAnalysis.getImage(f"{folderpath}/{image}")[:,0:-1],correctionImage)
            Ystd, Ymean = np.std(speckleImageY), np.mean(speckleImageY)
            corr.append(np.mean((initialSpeckle-Xmean)*(speckleImageY-Ymean))/(Xstd*Ystd))
            print(f"{len(corr)}/{len(allfiles[start:end])}")
        fig = plt.figure()
        fig.patch.set_facecolor("white")
        time = np.arange(0,len(corr)*5, 5)
        plt.title("Correlation through time of speckle pattern")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Correlation coefficient (a.u.)")
        plt.scatter(time, corr, color="black")
        plt.show()

    def correctionImage(folderpath):
        """Creates a correction image with all the images in a file

        Args:
            folderpath (string): Path of the folder where all the uniform images are saved
        """
        allfiles=os.listdir(folderpath)
        imlist= [SpeckleAnalysis.getImage(f"{folderpath}/{filename}")[:,0:-1] for filename in allfiles if filename[-5:]==".tiff"]
        ref = np.sum(imlist,axis=0)
        ref = np.where(ref < 0.001, 0.001, ref)
        Reference = ref/np.max(ref)
        tiff.imwrite(f"{folderpath}\CorrectionImage.tiff", Reference)
        print(f"The correction image was saved under the following path:\n{folderpath}\CorrectionImage.tiff\n")
        tiff.imshow(Reference)

    def AnalyseCorrelation(folderpath, window_size, time_seperation = 1, save=True, save_name="Data_Correlation"):
        """Analyse the correlation between the images in a folder and saves the data in a excel file.
        The data is in a dataframe that can be saved in a excel file. The excel file is saved in the folder where the images are saved.
        
        folderpath (string): Path of the folder where all the images are saved
        window_size (int): Size of the window used to calculate the correlation
        time_seperation (int): Time unit of seperation between the images. Defaults is 1.
        save (bool): If True, the data is saved in a excel file. Defaults to True.
        save_name (string): Name of the excel file. Defaults to "Data_Correlation".
        
        Returns: Dataframe with the following columns: First Image, Correlation Factor, R_score, Correlation Time"""
        def expo(x, a, b, c):
            """This function is used to calculate the exponential function. It is used for the fit to find the correlation factor."""
            return a * np.exp(-b * x) + c
        def CorrelationTime(a):
            """This function is used to calculate the correlation time. It takes for argument the correlation factor."""
            return 1/a
        allfiles=SpeckleAnalysis.sortedAlphanumeric(os.listdir(folderpath))
        allfiles= [filename for filename in allfiles if filename[-5:]==".tiff"]
        maxi = len(allfiles) - window_size + 1
        correlationFactor = []
        image_index = []
        r_scores = []
        for i in range(maxi):
            try:
                correlationValues = []
                speckleImageX = SpeckleAnalysis.getImage(f"{folderpath}/{allfiles[i]}")
                Xstd, Xmean = np.std(speckleImageX), np.mean(speckleImageX)
                for image in allfiles[i:i+window_size]:
                    clear_output(wait=True)
                    speckleImageY = SpeckleAnalysis.getImage(f"{folderpath}/{image}")
                    Ystd, Ymean = np.std(speckleImageY), np.mean(speckleImageY)
                    correlationValues.append(np.mean((speckleImageX-Xmean)*(speckleImageY-Ymean))/(Xstd*Ystd))
                time = np.arange(0,len(correlationValues), time_seperation)
                popt, pcov = curve_fit(expo, time, correlationValues)
                y_pred = expo(time, *popt)
                r = r2_score(correlationValues, y_pred)
                correlationFactor.append(popt[1])
                image_index.append(i)
                r_scores.append(r)
                print(f"{i+1}/{maxi}")
            except RuntimeError:
                print(f"{i+1}/{maxi}")
        correlationTime = list(map(CorrelationTime, correlationFactor))
        df = pd.DataFrame({"First Image": image_index, "Correlation Factor": correlationFactor, "R_score": r_scores, "Correlation Time": correlationTime})
        if save:
            df.to_excel(f"{folderpath}/{save_name}.xlsx", index=False)
        return df

    def show_correlation_results(filepath, min=0, max=100000, r=0.97, time_scale="seconds"):
        """This function plots the correlation time distribution for the speckle pattern.

        filepath: path to the excel containing the speckle pattern data organized in columns: A:First Image, B:Correlation Factor, C:R_score, and D:Correlation Time.
        min: minimum correlation time to be considered. This is used to remove the faulty data. Default is 0 to take into account all data.
        max: maximum correlation time to be considered. This is used to remove the faulty data. Default is 100000 to take into account all data.
        r: correlation r2_score threshold. This is used to remove the faulty data. Default is 0.97.
        time_scale: time scale of the correlation time. This is used to plot the correlation time distribution. Default is "seconds".

        return: correlation time distribution plot."""
        df = pd.read_excel(filepath, header=0, usecols="A,B,C,D")
        dfa = df.loc[df["Correlation Time"]<max]
        dfb = dfa.loc[dfa["Correlation Time"]>min]
        dfc = dfb.loc[dfb["R_score"]>r]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
        fig.patch.set_facecolor("white")
        ax1.hist(x=dfb["Correlation Time"], bins=25, label="Average/STD: {0:1.3f} +/- {1:1.3f}".format(dfb["Correlation Time"].mean(), dfb["Correlation Time"].std()))
        ax1.set(xlabel=f"Correlation time ({time_scale})", ylabel="Count")
        ax1.legend()
        fig.suptitle(f"Correlation time distribution\nR_score threshold = {r}"+"\n{0:1.3f} % of {1} data is taken into account".format(len(dfc)*100/len(dfb), len(dfb)))
        ax2.hist(x=dfc["Correlation Time"], bins=25, label="Average/STD: {0:1.3f} +/- {1:1.3f}".format(dfc["Correlation Time"].mean(), dfc["Correlation Time"].std()))
        ax2.set(xlabel="Cardboard correlation time (min)", ylabel="Count")
        ax2.legend()
        plt.show()

