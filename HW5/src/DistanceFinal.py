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

#Generate size count of cells
def cellCount (labels):
    #Create array to hold size in pixels
    cellSizes = np.zeros(np.amax(labels), dtype = np.uint32)
    #Extract size attributes
    imgHeight, imgWidth = labels.shape[:2]

    for y in range(imgHeight):
        for x in range(imgWidth):
            cellSizes[labels[y][x] - 1] += 1
            
    '''#Remove two largest sizes (background and watershed dams)
    maxInd = np.argmax(cellSizes)
    cellSizes = np.delete(cellSizes, maxInd)      
    maxInd = np.argmax(cellSizes)
    cellSizes = np.delete(cellSizes, maxInd)'''

    return cellSizes

def thresholdSize(labels, cellSizes, img, m, l):
    #Create array to hold threshold category
    thresholdCat = np.zeros((np.amax(labels), 3), dtype = np.uint32)
    
    for i in range(len(cellSizes)):
        if (cellSizes[i] < m):
            thresholdCat[i] = [0,0,255]
        elif(cellSizes[i] < l):
            thresholdCat[i] = [0,255,0]
        elif(cellSizes[i] < 1000):
            thresholdCat[i] = [255,0,0]
        else:
            thresholdCat[i] = [0,0,0]
    
    #Extract size attributes
    imgHeight, imgWidth = img.shape[:2]
    
    #Create the result image
    dst = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)

    for y in range(imgHeight):
        for x in range(imgWidth):
            #print(dst[y][x])
            dst[y][x] = thresholdCat[labels[y,x] - 1]
            #print(dst[y][x])
            
    displayImage(dst)



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

#Generate sure background and foreground images
sure_fg = cv.erode(img, None, iterations=2)
sure_bg = cv.dilate(img, None, iterations=2)
#displayImage(sure_bg)

#Extract local maxima
markers = peak_local_max(distTrans, 3, threshold_rel=.4)

kernel = np.zeros((3,3), dtype=np.uint8)
kernel[0][1] = 1
kernel[1][0] = 1
kernel[1][1] = 1
kernel[1][2] = 1
kernel[2][1] = 1

#Place center points at maxima and dilate
imgCenters = centerImage(distTrans, markers)
imgCenters = cv.dilate(imgCenters, None, iterations=2)

#Perform connected components on centers
ret, labels = cv.connectedComponents(imgCenters)

#Subtract known centers 
unknown = cv.subtract(sure_bg, imgCenters)
#displayImage(unknown)

#Increment labels to save 0 for unknown
labels = labels + 1

#Set unknown area in combined image to 0
labels[unknown == 255] = 0
        
#Convert to color
imgCenters = cv.cvtColor(imgCenters, cv.COLOR_GRAY2BGR)
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

#Perform watershed
labels = cv.watershed(img, labels)

#Generate random colors
colors = []
for contour in range(np.amax(labels)):
    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
        
#Create the result image
dst = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint16)
print(labels.shape)
#Fill labeled objects with random colors
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        index = labels[i,j]
        dst[i][j] = colors[index-1]

#Set bg to blue
dst[labels == 1] = [255,0,0]

#Generate size histogram
size = cellCount(labels)
thresholdSize(labels,size, img, 70, 110)

#Save final image
cv.imwrite(outPath, dst)