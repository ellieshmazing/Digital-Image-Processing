'''Program to apply morphological functions on input image'''
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import random as rng
from DIPLib import displayImage

#Generate histogram of single channel of image
def histogram (img, nBins):
    #Extract size attributes
    imgHeight, imgWidth = img.shape[:2]

    #Create numpy array to hold histogram data
    resultHist = np.zeros(nBins)

    #Determine bin size in intensity range
    binWidth = 256 / nBins

    for y in range(imgHeight):
        for x in range(imgWidth):
            if (int(img[y][x] * 256 / binWidth) >= 256):
                pixelBin = 255
            else:
                pixelBin = int(img[y][x] * 256 / binWidth)
            resultHist[pixelBin] = resultHist[pixelBin] + 1

    return resultHist

def minMaxLCS(img):
    #Generate histogram of image intensity values
    imgHist = histogram(img, 256)
    
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
            if (int(img[y][x] * 256) >= 256):
                img[y][x] = newVals[255]
            else:
                img[y][x] = newVals[int(256 * img[y][x])]
            
    return img

def plotCenters(img, markers):
    #Draw circles around found centers
    fig, ax = plt.subplots()
    plt.gray()
    nh, nw = img.shape[:2]
    count = 0

    ax.imshow(img)
    for blob in markers:
        y, x = blob
        c = plt.Circle((x,y), 1, color = 'red')
        ax.add_patch(c)
    
    ax.plot()
    plt.show()
    
def centerImage(img, markers):
    imgHeight, imgWidth = img.shape[:2]
    
    imgCenters = np.zeros((imgHeight, imgWidth), dtype=np.uint8)
    
    for coor in markers:
        y, x = coor
        imgCenters[y][x] = 255
        
    return imgCenters

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
distTrans = cv.distanceTransform(img, cv.DIST_L2, cv.DIST_MASK_PRECISE)

#Normalize into range {0.0, 1.0}
cv.normalize(distTrans, distTrans, 0, 1.0, cv.NORM_MINMAX)

sure_fg = cv.erode(img, None, iterations=2)
sure_bg = cv.dilate(img, None, iterations = 2)
ret, sure_bg = cv.threshold(sure_bg, 1, 128, 1)

combine = cv.add(sure_fg, sure_bg)
combine = cv.cvtColor(combine, cv.COLOR_GRAY2BGR)
combine.convertTo()

displayImage(combine)
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.watershed(img, combine)
m = cv.convertScaleAbs(combine)
displayImage(combine)

#Extract local maxima
markers = peak_local_max(distTrans, 5, threshold_rel=.4)

imgCenters = centerImage(distTrans, markers)
kernel = np.ones((2,2), np.uint8)

unknown = cv.subtract(img, imgCenters)

ret, labels = cv.connectedComponents(imgCenters)

labels = labels + 1

labels[unknown == 255] = 0
        

imgCenters = cv.cvtColor(imgCenters, cv.COLOR_GRAY2BGR)
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)


labels = cv.watershed(img, labels)

img[labels == -1] = [255,0,0]

displayImage(img)

# Generate random colors
colors = []
for contour in range(len(labels)):
    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
print(len(colors))
    
    
# Create the result image
dst = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint16)
# Fill labeled objects with random colors
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        index = labels[i,j]
        if index > 0 and index <= len(labels):
            dst[i][j] = colors[index-1]

displayImage(dst)

#Threshold
#_, imgThresh = cv.threshold(distTrans, 0.45 * distTrans.max(), 255, 0)

cv.imwrite(outPath, dst)