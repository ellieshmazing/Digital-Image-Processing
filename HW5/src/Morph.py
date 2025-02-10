'''Program to apply morphological functions on input image'''
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

#Read in image
img = cv.imread(inPath)

#Generate structuring element
kernel = np.ones((3,2), np.uint8)

print(kernel)

openedImg = cv.dilate(img, kernel)
openedImg = cv.erode(openedImg, kernel)


displayImage(openedImg)

cv.imwrite(outPath, openedImg)