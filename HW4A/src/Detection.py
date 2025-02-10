'''Program to detect blobs in input image'''
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from DIPLib import displayImage, dropBotLCS

def minMaxLCS(img):
    imgHeight, imgWidth = img.shape[:2]
    min = 255
    max = 0
    
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (img[y][x] < min):
                min = img[y][x]
            if (img[y][x] > max):
                max = img[y][x]
                
    for y in range(imgHeight):
        for x in range(imgWidth):
            val = int((((img[y][x] - min) / (max - min)) * 255))
            img[y][x] = val
            
    return img

def absVal(img):
    imgHeight, imgWidth = img.shape[:2]
    
    for y in range(imgHeight):
        for x in range(imgWidth):
            val = abs(img[y][x][0])
            img[y][x][0] = val
            img[y][x][1] = val
            img[y][x][2] = val
            
def threshold(img, thresh):
    imgHeight, imgWidth = img.shape[:2]
    
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (img[y][x] > thresh):
                img[y][x] = 255
            else:
                img[y][x] = 0
                
    return img

#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

img = cv.imread(inPath)
'''for sig in range(1, 4):
    print(4 + (sig * .3))
    imgLapped = ndimage.gaussian_laplace(img, sigma= 4 + (sig * .3))
    imgLapped = minMaxLCS(imgLapped)
    displayImage(imgLapped)
    imgLapped = threshold(imgLapped, 200)

    displayImage(imgLapped)
'''

img = cv.bilateralFilter(img, 10, 45, 10)
displayImage(img)
'''img = cv.imread(inPath)
imgHeight, imgWidth = img.shape[:2]

img = dropBotLCS(img, 0, 20)
img = cv.bilateralFilter(img, 10, 10, 10)

for y in range(imgHeight):
    for x in range(imgWidth):
        if (img[y][x][2] > 220 or img[y][x][1] > 200):
            img[y][x][0] = 255
            img[y][x][1] = 255
            img[y][x][2] = 255
        img[y][x][1] = img[y][x][0]
        img[y][x][2] = img[y][x][0]
        
displayImage(img)
cv.imwrite(outPath, img)'''