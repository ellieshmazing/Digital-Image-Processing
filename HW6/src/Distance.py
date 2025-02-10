'''Program to apply morphological functions on input image'''
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from DIPLib import displayImage

#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

#Read in image
img = cv.imread(inPath)

#Convert to gray for appropriate dtype
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Invert image
img = 255 - img

#Apply distance transform
distTrans = cv.distanceTransform(img, cv.DIST_C, 5)

#Normalize into range {0.0, 1.0}
cv.normalize(distTrans, distTrans, 0, 1.0, cv.NORM_MINMAX)

#Threshold
_, imgThresh = cv.threshold(distTrans, 0.45 * distTrans.max(), 255, 0)

cv.imwrite(outPath, imgThresh)