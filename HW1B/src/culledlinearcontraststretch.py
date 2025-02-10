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
dropPercent = int(sys.argv[2])

#Read in source image and extract necessary information
img = cv.imread(inPath)
imgHeight, imgWidth = img.shape[:2]
imgPixelCount = imgHeight * imgWidth
dropCount = int(imgPixelCount * dropPercent / 100)

#Generate histogram of image intensity values
imgHist = histogram(img, 256, 0)

#Get minimum and maximum intensity values
minVal = 0
maxVal = 0
lowCount = 0
for x in range(0, 256):
    if (imgHist[x] != 0):
        lowCount += imgHist[x]
        if (lowCount > dropCount):
            minVal = x
            break

highCount = 0
for x in reversed(range(0, 256)):
    if (imgHist[x] != 0):
        highCount += imgHist[x]
        if (highCount > dropCount):
            maxVal = x
            break

print(minVal)
print(maxVal)

#Calculate new value for each intensity level
diff = maxVal - minVal
newVals = np.arange(255)
for x in range(0, 255):
    newVals[x] = int(255 * ((x - minVal) / diff))
    if (newVals[x] < 0):
        newVals[x] = 0
    if (newVals[x] > 255):
        newVals[x] = 255

#Change each pixel to new intensity value
for y in range(imgHeight):
    for x in range(imgWidth):
        img[y][x] = newVals[img[y][x]]


imgHist = histogram(img, 256, 0)
displayHist(imgHist, 256)
cv.imshow("Stretched", img)
cv.waitKey(0)
cv.destroyAllWindows()