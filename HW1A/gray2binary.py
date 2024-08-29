'''Program to Binarize Grey Image'''
import os
import sys
import cv2 as cv

#Get paths for input and output images
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

#Read in source image
img = cv.imread(inPath)

#Apply threshold function
ret, imgGreyThreshold = cv.threshold(img, float(sys.argv[3]), float(255), cv.THRESH_BINARY)

#Display resulting image
cv.imshow("Cells", imgGreyThreshold)
cv.waitKey(0)
cv.destroyAllWindows()

#Save binarized image
cv.imwrite(outPath, imgGreyThreshold)