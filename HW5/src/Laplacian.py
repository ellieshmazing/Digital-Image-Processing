'''Program to detect blobs in input image'''
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from DIPLib import displayImage

def LapLevels(img, minSig, maxSig, step):
    laps = []
    
    stepCount = int((maxSig - minSig) / step)
    
    for stepNum in range(stepCount):
        imgLapped = ndimage.gaussian_laplace(img, sigma=minSig + (stepNum * step), output=np.float64)
        laps.append(imgLapped)
        
    return laps

def blobDetect(imgsLapped, minSig, maxSig, step, threshold, kSize, minBool):
    sqrRootTwo = 1.414
    
    blobCenters = []
    imgHeight, imgWidth = imgsLapped[0].shape[:2]
    
    stepCount = int((maxSig - minSig) / step)

    for y in range(imgHeight):
        print(y)
        for x in range(imgWidth):
            lapResult = []
            for stepNum in range(stepCount):
                areaCheck = []
                sigma = minSig + (stepNum * step)
                powSig = pow(sigma, 2)
                
                centerPix = pow(sigma, 2) * imgsLapped[stepNum][y][x][0]
                areaCheck.append(centerPix)
                
                for width in range(kSize):
                    for height in range(kSize):
                        if (y + height < imgHeight):  
                            if (x + width < imgWidth):
                                areaCheck.append(powSig * imgsLapped[stepNum][y + height][x + width][0])
                            if (x - width >= 0):
                                areaCheck.append(powSig * imgsLapped[stepNum][y + height][x - width][0])
                        if (y - height >= 0):
                            if (x + width < imgWidth):
                                areaCheck.append(powSig * imgsLapped[stepNum][y - height][x + width][0])
                            if (x - width >= 0):
                                areaCheck.append(powSig * imgsLapped[stepNum][y - height][x - width][0])
                        
                        
                        
                if ((minBool and centerPix == min(areaCheck)) or ((not minBool) and centerPix == max(areaCheck))):
                    lapResult.append(centerPix)
                
            if (len(lapResult) > 1):
                if (minBool):
                    imgMax = min(lapResult)
                else:
                    imgMax = max(lapResult)
                    #print(max(lapResult))
            
                if ((minBool and imgMax < threshold) or ((not minBool) and imgMax > threshold)):
                    blobCoor = []
                    blobCoor.append(y)
                    blobCoor.append(x)
                    blobCoor.append(sigma * sqrRootTwo)
                    blobCenters.append(blobCoor)
                
    return blobCenters

#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

img = plt.imread(inPath)

#Apply Laplacians of different sizes
imgsLapped = LapLevels(img, 2, 8, .5)

#Determine blob centers
blobCenters = blobDetect(imgsLapped, 2, 8, .5, 50, 10, False)
print(len(blobCenters))

#Draw circles around found centers
fig, ax = plt.subplots()
plt.gray()
nh, nw = img.shape[:2]
count = 0

ax.imshow(img)
for blob in blobCenters:
    y, x, r = blob
    c = plt.Circle((x,y), 1, color = 'red')
    ax.add_patch(c)
    
'''ax.imshow(img)
for blob in blobCenters:
    y, x, r = blob
    c = plt.Circle((x,y), r, color = 'blue', fill = False)
    ax.add_patch(c)'''
    
ax.plot()
plt.show()

plt.savefig(outPath)

'''#Create array of regional minima
threshold = .6

markers = []
imgHeight, imgWidth = img.shape[:2]

for y in range(1, imgHeight - 1):
    for x in range(1, imgWidth - 1):
        if (distTrans[y][x] > threshold and distTrans[y][x] > distTrans[y][x - 1] and distTrans[y][x] > distTrans[y - 1][x] and distTrans[y][x] > distTrans[y + 1][x] and distTrans[y][x] > distTrans[y][x + 1]):
            coordinate = []
            coordinate.append(y)
            coordinate.append(x)
            
            markers.append(coordinate)
            
print(len(markers))

#Draw circles around found centers
fig, ax = plt.subplots()
plt.gray()
nh, nw = img.shape[:2]
count = 0

ax.imshow(img)
for blob in markers:
    y, x= blob
    c = plt.Circle((x,y), 1, color = 'red')
    ax.add_patch(c)
    
ax.plot()
plt.show()'''