'''Program to Apply Linear Contrast Stretch to Greyscale Image'''
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from DIPLib import histogram, displayHist

#Function to apply Min-Max Linear Contrast Stretch
def linContrastStretch(img):
    return img

#Get paths for input file and bin number
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])

#Read in source image
img = cv.imread(inPath)

#Generate histogram of image intensity values
imgHist = histogram(img, 256, 0)

#Get minimum and maximum intensity values
minVal = 256
maxVal = 0 
for x in range(0, 256):
    if (imgHist[x] != 0):
        if (x < minVal):
            minVal = x
        if (x > maxVal):
            maxVal = x

#Calculate new value for each intensity level
diff = maxVal - minVal
newVals = np.arange(maxVal + 1)
for x in range(0, maxVal + 1):
    newVals[x] = int(255 * ((x - minVal) / diff))

#Change each pixel to new intensity value
imgHeight, imgWidth = img.shape[:2]

for y in range(imgHeight):
    for x in range(imgWidth):
        img[y][x] = newVals[img[y][x]]

imgHist = histogram(img, 256, 0)
minVal = 256
maxVal = 0 
for x in range(0, 256):
    if (imgHist[x] != 0):
        if (x < minVal):
            minVal = x
        if (x > maxVal):
            maxVal = x

print(minVal)
print(maxVal)


cv.imshow("Stretched", img)
cv.waitKey(0)
cv.destroyAllWindows()