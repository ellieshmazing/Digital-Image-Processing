'''Program to Binarize Image Based on Blue Channel'''
import os
import sys
import cv2 as cv

#Get paths for input and output images and threshold value
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])
threshold = int(sys.argv[3])

#Read in source image and get size
img = cv.imread(inPath)
imgHeight, imgWidth = img.shape[:2]

#Iterate through image and modify each pixel by blue threshold
for y in range(imgHeight):
    for x in range(imgWidth):
        if (img[y][x][0] > threshold):
            img[y][x][0] = 255
            img[y][x][1] = 255
            img[y][x][2] = 255
        else:
            img[y][x][0] = 0
            img[y][x][1] = 0
            img[y][x][2] = 0

#Display image
cv.imshow("Cells", img)
cv.waitKey(0)
cv.destroyAllWindows()

#Save blue binarization image
cv.imwrite(outPath, img)