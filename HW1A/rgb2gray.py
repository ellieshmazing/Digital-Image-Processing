'''Program to Convert RGB Input Image to Grey'''
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

#Merge channels into single value through weighted addition
imgGrey = cv.addWeighted(imgB, .1626, imgG, .8374, 0)
imgGrey = cv.addWeighted(imgGrey, .7011, imgR, .2989, 0)

#Display image
cv.imshow("Greyscale Cells", imgGrey)
cv.waitKey(0)
cv.destroyAllWindows()

#Save greyscale image
cv.imwrite(outPath, imgGrey)