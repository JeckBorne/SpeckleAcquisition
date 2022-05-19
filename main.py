from SpeckleClasses import SpeckleAnalysis, SpeckleAcquisition



goal = input("What are you trying to do? [1-Acquisition or 2-Analysis]: ")
if goal == "1":
    filename = input("Under what name should the individual frames be saved?: ")
    speckle = SpeckleAcquisition(filename)
    action = input("How are you trying to acquire images? [A-captureFrame or B-captureImages]: ")
    if action == "A":
        speckle.captureFrame()
    if action == "B":
        folder = input("In which folder should the images be saved?: ")
        delta_t = input("What should be the time intervals between each pictures (in seconds)?: ")
        speckle.captureImages(folder, float(delta_t))
elif goal == "2":
    specklePath = input("Please input the path of the raw speckle image: ")
    uniformPath = input("Please input the path of the uniform speckle image: ")
    speckle = SpeckleAnalysis.getImage(specklePath)
    uniform = SpeckleAnalysis.getImage(uniformPath)
    specklePattern = SpeckleAnalysis.imageSubstraction(speckle, uniform)
    histoName = input("Do you wish to save the histogram? If yes enter the name it should be saved as if not press enter: ")
    if histoName == "":
        SpeckleAnalysis.histogram(specklePattern)
    else:
        SpeckleAnalysis.histogram(specklePattern, histoName)


