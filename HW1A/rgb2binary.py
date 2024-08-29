'''Program to Binarize Image Based on Blue Channel'''
import os
import sys
import cv2 as cv

#Get paths for input and output images
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

#Read in source image and split into color channels
img = cv.imread(inPath)
imgB, imgG, imgR = cv.split(img)

#Apply threshold on blue channel
ret, imgBlueThreshold = cv.threshold(imgB, float(sys.argv[3]), float(255), cv.THRESH_BINARY)

#Display image
cv.imshow("Cells", imgBlueThreshold)
cv.waitKey(0)
cv.destroyAllWindows()

#Save blue binarization image
cv.imwrite(outPath, imgBlueThreshold)