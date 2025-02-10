'''Program to Convert RGB Input Image to Grey'''
import os
import sys
import cv2 as cv
import numpy as np

#Get paths for input and output images
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

#Read in source image and extract necessary attributes
img = cv.imread(inPath)
imgHeight, imgWidth = img.shape[:2]

#Create numpy array for new image
imgGrey = np.empty([imgHeight, imgWidth], dtype = np.uint8)

#Iterate through new image and calculate value for each pixel
for y in range(imgHeight):
    for x in range(imgWidth):
        imgGrey[y][x] = int(.1140 * img[y][x][0] + .5871 * img[y][x][1] + .2989 * img[y][x][2])

#Display image
cv.imshow("Greyscale Cells", imgGrey)
cv.waitKey(0)
cv.destroyAllWindows()

#Save greyscale image
cv.imwrite(outPath, imgGrey)