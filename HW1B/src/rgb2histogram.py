'''Program to Generate Histogram from Gray Image'''
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Function to generate histogram for input image with certain number of bins
def histogram (img, nBins, channel):
    #Extract size attributes
    imgHeight, imgWidth = img.shape[:2]

    #Create numpy array to hold histogram data
    resultHist = np.zeros(nBins)

    #Determine bin size in intensity range
    binWidth = 256 / nBins

    for y in range(imgHeight):
        for x in range(imgWidth):
            pixelBin = int(img[y][x][channel] / binWidth)
            resultHist[pixelBin] = resultHist[pixelBin] + 1

    return resultHist

#Get paths for input file and bin number
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
nBins = int(sys.argv[2])
channel = int(sys.argv[3])

#Read in source image
img = cv.imread(inPath)

#Generate histogram of image intensity values
imgHist = histogram(img, nBins, channel)

#Display histogram as bar graph
x = np.arange(nBins)
fig, ax = plt.subplots()
ax.bar(x, imgHist, width=1, edgecolor="white", linewidth=0.7)

plt.show()