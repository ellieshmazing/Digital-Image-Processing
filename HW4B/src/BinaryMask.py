import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from DIPLib import displayImage

def threshold(img, thresh, channel, thresh2, channel2, thresh3, channel3):
    imgHeight, imgWidth = img.shape[:2]
    
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (img[y][x][channel] > thresh and img[y][x][channel2] < thresh2 and img[y][x][channel3] < thresh3):
                img[y][x][0] = 0
                img[y][x][1] = 0
                img[y][x][2] = 0
            else:
                img[y][x][0] = 255
                img[y][x][1] = 255
                img[y][x][2] = 255
                
    return img

#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

#Load input image and extract shape features
img = cv.imread(inPath)
imgHeight, imgWidth = img.shape[:2]

img = threshold(img, 100, 0, 165, 2, 200, 1)

displayImage(img)

cv.imwrite(outPath, img)