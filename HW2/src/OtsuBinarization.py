'''Program to Apply Linear Contrast Stretch to Greyscale Image'''
import os
import sys
import cv2 as cv
from DIPLib import displayImage, histogram, histMean, displayHist, rgb2binary

#Wrapper function for recursive Otsu
def initOtsu(img, channel):
    #Get pixel count of image
    imgHeight, imgWidth = img.shape[:2]
    pixelCount = imgHeight * imgWidth
    
    #Generate histogram of image
    imgHist = histogram(img, 256, channel)
    displayHist(imgHist, 256)
    
    #Calculate image mean
    mean = histMean(imgHist, pixelCount, 256)
    
    #Calculate base values for qOne and variation squared
    qOne = imgHist[0] / pixelCount
    varSquare = qOne * (1 - qOne) * pow(0 - 1, 2)
    
    #Call recursive function
    return recursiveOtsu(imgHist, pixelCount, mean, 1, qOne, 0, varSquare)
    
#Recursive Otsu function
def recursiveOtsu (imgHist, pixelCount, mean, thresh, qOne, uOne, varSquare):
    #Calculate next values for pOne and qOne
    pOnePlus = imgHist[thresh] / pixelCount
    qOnePlus = qOne + pOnePlus
    
    #Calculate the left class mean (with default to 0 if qOnePlus is 0 to avoid divide by zero error)
    if (qOnePlus > 0):
        uOnePlus = ((qOne * uOne) + (thresh * pOnePlus)) / qOnePlus
    else:
        uOnePlus = 0
    
    #Calculate the right class mean
    uTwoPlus = (mean - (qOnePlus * uOnePlus)) / (1 - qOnePlus)
    
    #Calculate the between-class variance with current threshold
    varSquarePlus = qOnePlus * (1 - qOnePlus) * pow(uOnePlus - uTwoPlus, 2)
        
    #If variance decreases, return previous value as it was maximum
    #This is due to the usage of binarization, as the variance will always be an inverted polynomial. With more classes,
    #ending execution early would not be possible
    if (varSquarePlus < varSquare):
        return thresh - 1
    
    #End execution if threshold is maximum value
    if (thresh == 255):
        return thresh
        
    #Execute for next threshold value
    return recursiveOtsu(imgHist, pixelCount, mean, thresh + 1, qOnePlus, uOnePlus, varSquarePlus)
    

#Get paths for input file and bin number
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])
channel = int(sys.argv[3])

#Read in source image and extract necessary information
img = cv.imread(inPath)

#Get optimal threshold from recursive Otsu
threshold = initOtsu(img, channel)
print(threshold)

#Binarize image
img = rgb2binary(img, channel, threshold)
displayImage(img)

#Save image
cv.imwrite(outPath, img)