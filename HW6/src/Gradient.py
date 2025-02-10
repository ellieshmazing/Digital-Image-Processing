'''Program to apply morphological functions on input image'''
import os
import sys
import cv2 as cv
import numpy as np
import math
from scipy.signal import correlate2d, convolve2d
import matplotlib.pyplot as plt
from DIPLib import displayImage

#Generate histogram of single channel of image
def histogram (img, nBins):
    #Extract size attributes
    imgHeight, imgWidth = img.shape[:2]

    #Create numpy array to hold histogram data
    resultHist = np.zeros(nBins, dtype=np.uint32)


    for y in range(imgHeight):
        for x in range(imgWidth):
            pixelBin = round(img[y][x])
            resultHist[pixelBin] = resultHist[pixelBin] + 1
            
    return resultHist


'''def minMaxLCS(img):
    #Generate histogram of image intensity values
    imgHist = histogram(img, 512)
    
    dropCount = 5000
    
    #Get minimum and maximum intensity values
    minVal = 0
    lowCount = 0
    for x in range(0, 512):
        if (imgHist[x] != 0):
            lowCount += imgHist[x]
            if (lowCount > dropCount):
                minVal = x
                break
            
    maxVal = 0
    highCount = 0
    for x in reversed(range(0, 512)):
        if (imgHist[x] != 0):
            highCount += imgHist[x]
            if (highCount > dropCount):
                maxVal = x
                break
            
    #Calculate new value for each intensity level
    diff = maxVal - minVal
    
    newVals = np.arange(maxVal + 1)
    for x in range(0, maxVal + 1):
        newVals[x] = int(255 * ((x - minVal) / diff))

        
    #Change each pixel to new intensity value
    imgHeight, imgWidth = img.shape[:2]

    for y in range(imgHeight):
        for x in range(imgWidth):
            if (int(img[y][x] + 256) > maxVal):
                img[y][x] = 255
            elif(int(img[y][x] + 256) < minVal):
                img[y][x] = 0
            else:
                img[y][x] = newVals[int(img[y][x] + 256)] 
                
            
    return img'''
    
def minMaxLCS(img):
    #Generate histogram of image intensity values
    imgHist = histogram(img, 301)
    
    #Get minimum and maximum intensity values
    dropCount = 0
    
    #Get minimum and maximum intensity values
    minVal = 0
    lowCount = 0
    for x in range(0, 301):
        if (imgHist[x] != 0):
            lowCount += imgHist[x]
            if (lowCount > dropCount):
                minVal = x
                break
            
    maxVal = 0
    highCount = 0
    for x in reversed(range(0, 301)):
        if (imgHist[x] != 0):
            highCount += imgHist[x]
            if (highCount > dropCount):
                maxVal = x
                break
                
                
    #Calculate new value for each intensity level
    diff = maxVal - minVal
    newVals = np.arange(maxVal + 1)
    for x in range(0, maxVal + 1):
        newVals[x] = int(255 * ((x - minVal) / diff))
        
    #Change each pixel to new intensity value
    imgHeight, imgWidth = img.shape[:2]

    for y in range(imgHeight):
        for x in range(imgWidth):
            if (int(img[y][x]) > maxVal):
                img[y][x] = 255
            elif(int(img[y][x]) < minVal):
                img[y][x] = 0
            else:
                img[y][x] = newVals[int(img[y][x])] 
            
    return img

def gradientX (img):
    #Extract image size information
    imgHeight, imgWidth = img.shape[:2]
    
    #Declare filter
    xFilt = np.zeros((1,3), dtype=float)
    xFilt[0][1] = -1
    xFilt[0][2] = 1
    
    #Convolve image with filter
    gradX = correlate2d(img, xFilt, 'same')
    
    return gradX

def gradientY (img):
    #Extract image size information
    imgHeight, imgWidth = img.shape[:2]
    
    #Declare filter
    yFilt = np.zeros((3,1), dtype=float)
    yFilt[1][0] = -1
    yFilt[2][0] = 1
    
    #Convolve image with filter
    gradY = correlate2d(img, yFilt, 'same')
    
    return gradY

def gradientMagnitude(gradX, gradY):
    #Extract image size information
    imgHeight, imgWidth = gradY.shape[:2]
    imgHeight -= 2
    imgWidth -= 2

    gradM = np.zeros((imgHeight, imgWidth), dtype=np.float64)
    
    for y in range(imgHeight):
        for x in range(imgWidth):
            gradM[y][x] = math.sqrt(gradX[y][x] ** 2 + gradY[y][x] ** 2)
            
    return gradM

def gradientOrientation(gradX, gradY):
    #Extract image size information
    imgHeight, imgWidth = gradX.shape[:2]
    imgHeight -= 2
    imgWidth -= 2
    
    gradTheta = np.zeros((imgHeight, imgWidth), dtype=np.float16)
    
    for y in range(imgHeight):
        for x in range(imgWidth):
            gradTheta[y][x] = math.atan2(gradY[y][x], gradX[y][x])
            gradTheta[y][x] = math.degrees(gradTheta[y][x])
            
            if (gradTheta[y][x] < 0):
                gradTheta[y][x] += 360
                
    return gradTheta

#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

#Read in image
img = cv.imread(inPath)

#Make image grayscale
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Apply Gaussian filter
gradY = cv.GaussianBlur(img, (1, 15), 5)
gradX = cv.GaussianBlur(img, (15, 1), 5)

#Calculate gradient
gradY = gradientY(gradY)
gradX = gradientX(gradX)

#Compute gradient magnitude
gradTheta = gradientOrientation(gradX, gradY)

histTheta, bins = np.histogram(gradTheta, 359, (0, 359), density=False)

fig, ax = plt.subplots()
ax.bar(bins[:-1], histTheta, edgecolor="red", align="edge")
#plt.hist(histTheta, bins=359)
plt.show()

#Save output image
cv.imwrite(outPath, gradTheta)