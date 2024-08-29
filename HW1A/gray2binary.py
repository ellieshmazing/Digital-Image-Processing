'''Program to Binarize Grey Image'''
import os
import sys
import cv2 as cv

#Get paths for input and output images and threshold value
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])
threshold = int(sys.argv[3])

#Read in greyscale source image and get size
img = cv.imread(inPath, 0)
imgHeight, imgWidth = img.shape[:2]

#Iterate through each pixel and modify according to threshold
for y in range(imgHeight):
    for x in range(imgWidth):
        if (img[y][x] > threshold):
            img[y][x] = 255
        else:
            img[y][x] = 0

#Display resulting image
cv.imshow("Binarized Cells", img)
cv.waitKey(0)
cv.destroyAllWindows()

#Save binarized image
cv.imwrite(outPath, img)