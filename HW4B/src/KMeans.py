'''Program to implement k-means clustering on input image'''
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from DIPLib import displayImage

#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

#Load input image and extract shape features
img = cv.imread(inPath)
imgHeight, imgWidth = img.shape[:2]

pixelData = np.zeros(((imgHeight * imgWidth), 5), dtype=int)

#Iterate through image and append coordinates to pixel info
pixelCount = 0
for y in range(imgHeight):
    for x in range(imgWidth):
        pixelData[pixelCount][0] = img[y][x][0]
        pixelData[pixelCount][1] = img[y][x][1]
        pixelData[pixelCount][2] = img[y][x][2]
        pixelData[pixelCount][3] = y
        pixelData[pixelCount][4] = x
        
        pixelCount += 1
        
#Convert to float for use with cv.kmean
pixelData = np.float32(pixelData)

#Declare cv.kmean parameters
k = 400
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100000, .000001)

#Apply cv.kmean
returnData, labels, centers = cv.kmeans(pixelData, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

#Convert results to 8-bit format
centers = np.uint8(centers)
print(centers[0])
print(labels[0])

#Map labels to center points
segmentedData = centers[labels.flatten()]

    

#Return data to original shape
pixelCount = 0
for y in range(imgHeight):
    for x in range(imgWidth):
        img[y][x][0] = segmentedData[pixelCount][0]
        img[y][x][1] = segmentedData[pixelCount][1]
        img[y][x][2] = segmentedData[pixelCount][2]
        
        pixelCount += 1

#Display segmented image
displayImage(img)

#Save segmented image
cv.imwrite(outPath, img)